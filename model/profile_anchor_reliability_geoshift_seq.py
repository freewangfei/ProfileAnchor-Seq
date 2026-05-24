import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from util.selective_multimethod_lithofacies import (
    metric_row,
    model_proba,
    posterior_pool,
    selective_rows,
    summarize,
)
from model.spatial_lithofacies_selective_geoshift_seq import margin
from model.spatial_lithofacies_tree_stnet_posterior_fusion import family_posteriors, fit_family_views, normalized
from data.spatial_multimethod_group_benchmark import build_features, load_force, make_model, sample_by_well, split_wells_by_space

try:
    import torch
except Exception:
    torch = None


DEFAULT_COVERAGES = [0.2, 0.1, 0.05, 0.02, 0.01]


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-12, None)
    return x / x.sum(axis=1, keepdims=True)


def disable_hist_gradient_boosting_validation(model):
    if isinstance(model, Pipeline):
        for _, step in model.steps:
            if isinstance(step, HistGradientBoostingClassifier):
                step.set_params(early_stopping=False)
    elif isinstance(model, HistGradientBoostingClassifier):
        model.set_params(early_stopping=False)
    return model


def anchor_vote_share(anchors: list[np.ndarray], n_classes: int) -> np.ndarray:
    preds = np.stack([p.argmax(axis=1) for p in anchors], axis=0)
    return np.array(
        [np.max(np.bincount(preds[:, i], minlength=n_classes)) / preds.shape[0] for i in range(preds.shape[1])],
        dtype=float,
    )


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(p) - np.log(m))).sum(axis=1)
    kl_qm = (q * (np.log(q) - np.log(m))).sum(axis=1)
    return 0.5 * (kl_pm + kl_qm) / np.log(2.0)


def nearest_train_distance(frame: pd.DataFrame, train_wells: set[str]) -> np.ndarray:
    well_xy = frame.groupby("WELL")[["X_LOC", "Y_LOC"]].mean().dropna()
    train_xy = well_xy.loc[[w for w in train_wells if w in well_xy.index]].to_numpy(dtype=float)
    if len(train_xy) == 0:
        return np.zeros(len(frame), dtype=float)
    dist_by_well = {}
    for well, xy in well_xy.iterrows():
        d = np.sqrt(((train_xy - xy.to_numpy(dtype=float)) ** 2).sum(axis=1))
        dist_by_well[well] = float(d.min())
    fill = float(np.median(list(dist_by_well.values()))) if dist_by_well else 0.0
    return frame["WELL"].map(dist_by_well).fillna(fill).to_numpy(dtype=float)


def add_depth_profile_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    grouped = work.groupby("WELL", sort=False)
    dmin = grouped["DEPTH_MD"].transform("min")
    dmax = grouped["DEPTH_MD"].transform("max")
    span = (dmax - dmin).replace(0, np.nan)
    work["DEPTH_PROFILE"] = ((work["DEPTH_MD"] - dmin) / span).fillna(0.5)
    if "Z_LOC" in work.columns:
        zmin = grouped["Z_LOC"].transform("min")
        zmax = grouped["Z_LOC"].transform("max")
        zspan = (zmax - zmin).replace(0, np.nan)
        work["Z_PROFILE"] = ((work["Z_LOC"] - zmin) / zspan).fillna(0.5)
    return work


def prototype_posterior(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    n_classes: int,
    args,
) -> np.ndarray:
    train_p = add_depth_profile_features(train)
    frame_p = add_depth_profile_features(frame)
    proto_features = [col for col in args.prototype_features if col in train_p.columns and col in frame_p.columns]
    if not proto_features:
        return np.full((len(frame), n_classes), 1.0 / n_classes, dtype=float)

    x_train = train_p[proto_features].to_numpy(dtype=float)
    x_frame = frame_p[proto_features].to_numpy(dtype=float)
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_frame = np.where(np.isfinite(x_frame), x_frame, med)
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    x_train = (x_train - center) / scale
    x_frame = (x_frame - center) / scale

    y_train = train_p["TARGET"].to_numpy(dtype=int)
    global_var = np.var(x_train, axis=0) + 1e-3
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    prior = counts + float(args.prototype_prior_smoothing)
    prior = prior / prior.sum()
    if args.prototype_prior_temperature > 0:
        prior = np.power(prior, float(args.prototype_prior_temperature))
        prior = prior / prior.sum()

    logp = np.full((len(frame), n_classes), -1e6, dtype=float)
    min_count = max(2, int(args.prototype_min_class_rows))
    for cls in range(n_classes):
        cls_x = x_train[y_train == cls]
        if len(cls_x) < min_count:
            continue
        mu = cls_x.mean(axis=0)
        cls_var = np.var(cls_x, axis=0) + 1e-3
        var = (
            (1.0 - float(args.prototype_var_shrinkage)) * cls_var
            + float(args.prototype_var_shrinkage) * global_var
        )
        var = np.clip(var, 1e-3, None)
        dist = ((x_frame - mu) ** 2 / var).mean(axis=1)
        complexity = np.log(var).mean()
        logp[:, cls] = -0.5 * float(args.prototype_distance_scale) * dist - 0.5 * complexity + np.log(prior[cls] + 1e-12)
    logp = logp - np.max(logp, axis=1, keepdims=True)
    return normalize_rows(np.exp(logp))


def target_profile_score(
    frame: pd.DataFrame,
    coupled: np.ndarray,
    geoshift: np.ndarray,
    anchor_pool: np.ndarray,
    anchors: list[np.ndarray],
    train_wells: set[str],
    args,
) -> dict[str, np.ndarray]:
    n_classes = coupled.shape[1]
    m_c = margin(coupled)
    m_g = margin(geoshift)
    m_a = margin(anchor_pool)
    vote = anchor_vote_share(anchors, n_classes)
    agree_ga = (geoshift.argmax(axis=1) == anchor_pool.argmax(axis=1)).astype(float)
    min_m = np.minimum.reduce([m_c, m_g, m_a])
    disagreement = js_divergence(geoshift, anchor_pool)

    work = frame.reset_index(drop=True).copy()
    work["_margin"] = m_c
    work["_vote"] = vote
    work["_agree"] = agree_ga
    work["_entropy"] = entropy(coupled)
    well_top_margin = work.groupby("WELL")["_margin"].transform(
        lambda s: s.nlargest(max(1, int(round(args.profile_top_fraction * len(s))))).mean()
    ).to_numpy(dtype=float)
    well_vote = work.groupby("WELL")["_vote"].transform("mean").to_numpy(dtype=float)
    well_agree = work.groupby("WELL")["_agree"].transform("mean").to_numpy(dtype=float)
    well_entropy = work.groupby("WELL")["_entropy"].transform("mean").to_numpy(dtype=float)
    dist = nearest_train_distance(work, train_wells)

    profile = (
        args.well_margin_weight * minmax(well_top_margin)
        + args.well_vote_weight * minmax(well_vote)
        + args.well_agree_weight * minmax(well_agree)
        - args.well_entropy_weight * minmax(well_entropy)
        - args.distance_weight * minmax(dist)
    )

    base = m_c
    robust = 0.45 * m_c + 0.20 * min_m + 0.15 * vote + 0.10 * agree_ga - 0.10 * minmax(disagreement)
    profile_anchor = robust + args.profile_weight * profile
    strict_profile = (
        0.35 * min_m
        + 0.20 * m_c
        + 0.20 * vote
        + 0.15 * agree_ga
        + args.profile_weight * profile
        - 0.10 * minmax(disagreement)
    )
    return {
        "margin": base,
        "robust_anchor_margin": robust,
        "target_profile_anchor": profile_anchor,
        "strict_target_profile_anchor": strict_profile,
    }


def rank_percentile(score: np.ndarray) -> np.ndarray:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(score), endpoint=True)
    return ranks


def rank_fusion_scores(
    coupled: np.ndarray,
    geoshift: np.ndarray,
    anchor_pool: np.ndarray,
    anchors: list[np.ndarray],
) -> dict[str, np.ndarray]:
    n_classes = coupled.shape[1]
    vote = anchor_vote_share(anchors, n_classes)
    agree = (geoshift.argmax(axis=1) == anchor_pool.argmax(axis=1)).astype(float)
    min_m = np.minimum.reduce([margin(coupled), margin(geoshift), margin(anchor_pool)])
    anchor_mean_margin = np.mean([margin(p) for p in anchors], axis=0)
    anchor_min_margin = np.min(np.stack([margin(p) for p in anchors], axis=1), axis=1)
    disagreement = -js_divergence(geoshift, anchor_pool)
    ranks = [
        rank_percentile(margin(coupled)),
        rank_percentile(margin(geoshift)),
        rank_percentile(margin(anchor_pool)),
        rank_percentile(anchor_mean_margin),
        rank_percentile(anchor_min_margin),
        rank_percentile(vote),
        rank_percentile(agree),
        rank_percentile(min_m),
        rank_percentile(disagreement),
    ]
    mean_rank = np.mean(np.stack(ranks, axis=1), axis=1)
    strict_rank = np.minimum.reduce(ranks[:8])
    trimmed_rank = np.sort(np.stack(ranks, axis=1), axis=1)[:, 2:-1].mean(axis=1)
    return {
        "rank_fusion_mean": mean_rank,
        "rank_fusion_trimmed": trimmed_rank,
        "rank_fusion_strict": strict_rank,
    }


def smooth_score_by_well(frame: pd.DataFrame, score: np.ndarray, window: int, blend: float) -> np.ndarray:
    if window <= 1:
        return score
    out = np.zeros_like(score, dtype=float)
    work = frame.reset_index(drop=True)
    for _, group in work.groupby("WELL", sort=False):
        idx = group.index.to_numpy()
        rolled = (
            pd.Series(score[idx])
            .rolling(window=window, min_periods=1, center=True)
            .mean()
            .to_numpy(dtype=float)
        )
        out[idx] = blend * score[idx] + (1.0 - blend) * rolled
    return out


def interval_consistency_scores(frame: pd.DataFrame, scores: dict[str, np.ndarray], args) -> dict[str, np.ndarray]:
    rows = {}
    base_names = [name for name in scores if name.startswith("rank_fusion") or name in {"target_profile_anchor", "strict_target_profile_anchor"}]
    for name in base_names:
        for window in args.score_smooth_windows:
            for blend in args.score_smooth_blends:
                key = f"{name}_interval_w{int(window):02d}_b{int(round(100 * float(blend))):03d}"
                rows[key] = smooth_score_by_well(frame, scores[name], int(window), float(blend))
    return rows


def keep_score_name(score_name: str, args) -> bool:
    include = getattr(args, "score_include", None) or []
    exclude = getattr(args, "score_exclude", None) or []
    if include and not any(pattern in score_name for pattern in include):
        return False
    if exclude and any(pattern in score_name for pattern in exclude):
        return False
    return True


def class_aware_selective_rows(
    y: np.ndarray,
    proba: np.ndarray,
    seed: int,
    split: str,
    method: str,
    coverages: list[float],
    score: np.ndarray,
    n_classes: int,
    gamma: float,
) -> list[dict]:
    pred = proba.argmax(axis=1)
    by_class = {}
    for cls in range(n_classes):
        idx = np.where(pred == cls)[0]
        by_class[cls] = idx[np.argsort(score[idx])[::-1]] if len(idx) else idx

    rows = []
    total = len(y)
    support = np.array([len(by_class[cls]) for cls in range(n_classes)], dtype=float)
    active = support > 0
    weights = np.zeros(n_classes, dtype=float)
    weights[active] = support[active] ** float(gamma)
    if weights.sum() <= 0:
        weights[active] = 1.0
    weights = weights / weights.sum()

    global_order = np.argsort(score)[::-1]
    for coverage in coverages:
        keep_n = max(1, int(round(total * coverage)))
        raw_quota = keep_n * weights
        quota = np.floor(raw_quota).astype(int)
        for cls in np.argsort(raw_quota - quota)[::-1]:
            if quota.sum() >= keep_n:
                break
            if support[cls] > quota[cls]:
                quota[cls] += 1

        keep_parts = []
        for cls in range(n_classes):
            if quota[cls] > 0 and len(by_class[cls]):
                keep_parts.append(by_class[cls][: min(int(quota[cls]), len(by_class[cls]))])
        keep = np.concatenate(keep_parts) if keep_parts else np.array([], dtype=int)
        used = np.zeros(total, dtype=bool)
        used[keep] = True
        if len(keep) < keep_n:
            rest = global_order[~used[global_order]]
            keep = np.concatenate([keep, rest[: keep_n - len(keep)]])
        elif len(keep) > keep_n:
            keep = keep[np.argsort(score[keep])[::-1][:keep_n]]

        row = metric_row(y[keep], pred[keep])
        row.update({"seed": seed, "split": split, "method": method, "coverage": coverage, "kept_rows": len(keep)})
        rows.append(row)
    return rows


def source_prior_budget_rows(
    y: np.ndarray,
    proba: np.ndarray,
    seed: int,
    split: str,
    method: str,
    coverages: list[float],
    score: np.ndarray,
    source_prior: np.ndarray,
    strength: float,
    floor: float,
    quality_power: float,
) -> list[dict]:
    pred = proba.argmax(axis=1)
    n_classes = proba.shape[1]
    total = len(y)
    source_prior = np.asarray(source_prior, dtype=float)
    if source_prior.shape[0] != n_classes or source_prior.sum() <= 0:
        source_prior = np.ones(n_classes, dtype=float) / n_classes
    else:
        source_prior = source_prior / source_prior.sum()

    by_class = {}
    support = np.zeros(n_classes, dtype=float)
    quality = np.zeros(n_classes, dtype=float)
    global_order = np.argsort(score)[::-1]
    for cls in range(n_classes):
        idx = np.where(pred == cls)[0]
        support[cls] = len(idx)
        if len(idx):
            ordered = idx[np.argsort(score[idx])[::-1]]
            by_class[cls] = ordered
            top_n = max(1, int(round(0.10 * len(ordered))))
            quality[cls] = float(np.mean(rank_percentile(score)[ordered[:top_n]]))
        else:
            by_class[cls] = idx

    active = support > 0
    support_prior = np.zeros(n_classes, dtype=float)
    if support[active].sum() > 0:
        support_prior[active] = support[active] / support[active].sum()
    source_active = np.zeros(n_classes, dtype=float)
    if source_prior[active].sum() > 0:
        source_active[active] = source_prior[active] / source_prior[active].sum()
    blended = (1.0 - float(strength)) * support_prior + float(strength) * source_active
    if quality_power > 0:
        blended *= np.power(np.clip(quality, 1e-6, 1.0), float(quality_power))
    if blended.sum() <= 0:
        blended[active] = 1.0
    blended /= blended.sum()

    rows = []
    for coverage in coverages:
        keep_n = max(1, int(round(total * coverage)))
        raw_quota = keep_n * blended
        quota = np.floor(raw_quota).astype(int)
        protected = np.floor(keep_n * float(floor) * source_active).astype(int)
        quota = np.maximum(quota, protected)
        quota = np.minimum(quota, support.astype(int))
        while quota.sum() > keep_n:
            removable = np.where(quota > 0)[0]
            if len(removable) == 0:
                break
            cls = removable[np.argmax(quota[removable] - raw_quota[removable])]
            quota[cls] -= 1
        for cls in np.argsort(raw_quota - np.floor(raw_quota))[::-1]:
            if quota.sum() >= keep_n:
                break
            if support[cls] > quota[cls]:
                quota[cls] += 1

        keep_parts = []
        for cls in range(n_classes):
            if quota[cls] > 0:
                keep_parts.append(by_class[cls][: int(quota[cls])])
        keep = np.concatenate(keep_parts) if keep_parts else np.array([], dtype=int)
        used = np.zeros(total, dtype=bool)
        used[keep] = True
        if len(keep) < keep_n:
            rest = global_order[~used[global_order]]
            keep = np.concatenate([keep, rest[: keep_n - len(keep)]])
        elif len(keep) > keep_n:
            keep = keep[np.argsort(score[keep])[::-1][:keep_n]]

        row = metric_row(y[keep], pred[keep])
        row.update({"seed": seed, "split": split, "method": method, "coverage": coverage, "kept_rows": len(keep)})
        rows.append(row)
    return rows


def soft_class_prior_scores(pred: np.ndarray, score: np.ndarray, betas: list[float]) -> dict[str, np.ndarray]:
    if not betas:
        return {}
    pred = np.asarray(pred)
    counts = pd.Series(pred).map(pd.Series(pred).value_counts()).to_numpy(dtype=float)
    rarity = rank_percentile(1.0 / np.maximum(counts, 1.0))
    base = rank_percentile(score)
    rows = {}
    for beta in betas:
        b = float(beta)
        rows[f"soft_class_prior_b{int(round(1000 * b)):03d}"] = (1.0 - b) * base + b * rarity
    return rows


def minority_consensus_scores(
    pred: np.ndarray,
    score: np.ndarray,
    coupled: np.ndarray,
    geoshift: np.ndarray,
    anchor_pool: np.ndarray,
    anchors: list[np.ndarray],
    source_prior: np.ndarray,
    betas: list[float],
) -> dict[str, np.ndarray]:
    if not betas:
        return {}
    n_classes = coupled.shape[1]
    prior = np.asarray(source_prior, dtype=float)
    if prior.shape[0] != n_classes or prior.sum() <= 0:
        prior = np.ones(n_classes, dtype=float) / n_classes
    prior = prior / prior.sum()
    rarity_by_class = -np.log(np.clip(prior, 1e-8, 1.0))
    rarity_by_class = rarity_by_class / max(float(rarity_by_class.max()), 1e-8)
    rarity = rarity_by_class[np.asarray(pred, dtype=int)]
    vote = anchor_vote_share(anchors, n_classes)
    agree = (
        (pred == geoshift.argmax(axis=1))
        & (pred == anchor_pool.argmax(axis=1))
    ).astype(float)
    conservative_margin = rank_percentile(
        np.minimum.reduce([margin(coupled), margin(geoshift), margin(anchor_pool)])
    )
    consensus = rank_percentile(vote) * agree * conservative_margin
    bonus = rank_percentile(rarity * consensus)
    base = rank_percentile(score)
    rows = {}
    for beta in betas:
        b = float(beta)
        rows[f"minority_consensus_b{int(round(1000 * b)):03d}"] = (1.0 - b) * base + b * bonus
    return rows


def stratigraphic_continuity_scores(
    frame: pd.DataFrame,
    coupled: np.ndarray,
    geoshift: np.ndarray,
    anchor_pool: np.ndarray,
    scores: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    pred = coupled.argmax(axis=1)
    geo_pred = geoshift.argmax(axis=1)
    anchor_pred = anchor_pool.argmax(axis=1)
    base_margin = margin(coupled)
    out_len = np.zeros(len(frame), dtype=float)
    out_center = np.zeros(len(frame), dtype=float)
    out_local = np.zeros(len(frame), dtype=float)
    out_boundary = np.zeros(len(frame), dtype=float)
    work = frame.reset_index(drop=True)
    for _, group in work.groupby("WELL", sort=False):
        idx = group.index.to_numpy()
        p = pred[idx]
        m = base_margin[idx]
        n = len(idx)
        start = 0
        while start < n:
            end = start + 1
            while end < n and p[end] == p[start]:
                end += 1
            run_idx = idx[start:end]
            run_len = end - start
            pos = np.arange(run_len)
            dist_edge = np.minimum(pos + 1, run_len - pos)
            out_len[run_idx] = run_len
            out_center[run_idx] = dist_edge
            out_boundary[run_idx] = dist_edge / max(1, run_len)
            start = end
        for local_pos, global_idx in enumerate(idx):
            lo = max(0, local_pos - 3)
            hi = min(n, local_pos + 4)
            neigh = p[lo:hi]
            out_local[global_idx] = (neigh == p[local_pos]).mean()
        out_len[idx] *= np.clip(pd.Series(m).rolling(9, min_periods=1, center=True).mean().to_numpy(), 0.0, 1.0)

    continuity = (
        0.30 * rank_percentile(out_len)
        + 0.25 * rank_percentile(out_center)
        + 0.20 * rank_percentile(out_local)
        + 0.15 * rank_percentile((pred == geo_pred).astype(float))
        + 0.10 * rank_percentile((pred == anchor_pred).astype(float))
    )
    boundary_safe = 0.55 * continuity + 0.45 * rank_percentile(base_margin)
    rows = {"strat_continuity": continuity, "strat_boundary_safe": boundary_safe}
    if "rank_fusion_mean" in scores:
        rows["rank_fusion_mean_x_strat_continuity"] = 0.65 * rank_percentile(scores["rank_fusion_mean"]) + 0.35 * rank_percentile(continuity)
        rows["rank_fusion_mean_x_strat_boundary"] = 0.60 * rank_percentile(scores["rank_fusion_mean"]) + 0.40 * rank_percentile(boundary_safe)
    if "rank_fusion_trimmed" in scores:
        rows["rank_fusion_trimmed_x_strat_continuity"] = 0.65 * rank_percentile(scores["rank_fusion_trimmed"]) + 0.35 * rank_percentile(continuity)
        rows["rank_fusion_trimmed_x_strat_boundary"] = 0.60 * rank_percentile(scores["rank_fusion_trimmed"]) + 0.40 * rank_percentile(boundary_safe)
    for base_name in ["rank_fusion_trimmed", "rank_fusion_mean"]:
        if base_name not in scores:
            continue
        seg_score = np.zeros(len(frame), dtype=float)
        for _, group in work.groupby("WELL", sort=False):
            idx = group.index.to_numpy()
            p = pred[idx]
            start = 0
            while start < len(idx):
                end = start + 1
                while end < len(idx) and p[end] == p[start]:
                    end += 1
                run_idx = idx[start:end]
                run_score = 0.6 * rank_percentile(scores[base_name][run_idx]).mean() + 0.4 * rank_percentile(boundary_safe[run_idx]).mean()
                seg_score[run_idx] = run_score
                start = end
        rows[f"{base_name}_x_segment_boundary"] = rank_percentile(seg_score)
    return rows


def second_stage_rank_scores(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rows = {}
    pairs = [
        ("rank_fusion_trimmed", "target_profile_anchor"),
        ("rank_fusion_trimmed", "strict_target_profile_anchor"),
        ("rank_fusion_mean", "target_profile_anchor"),
        ("rank_fusion_mean", "robust_anchor_margin"),
    ]
    for left, right in pairs:
        if left not in scores or right not in scores:
            continue
        left_rank = rank_percentile(scores[left])
        right_rank = rank_percentile(scores[right])
        rows[f"second_stage_{left}_x_{right}"] = 0.5 * left_rank + 0.5 * right_rank
        rows[f"second_stage_weighted_{left}_x_{right}"] = 0.65 * left_rank + 0.35 * right_rank
    primary = rows.get("second_stage_rank_fusion_mean_x_robust_anchor_margin")
    if primary is not None:
        primary_rank = rank_percentile(primary)
        components = {"primary": primary_rank}
        for name in [
            "rank_fusion_mean",
            "rank_fusion_trimmed",
            "robust_anchor_margin",
            "margin",
            "rank_fusion_strict",
        ]:
            if name in scores:
                components[name] = rank_percentile(scores[name])
        if {"rank_fusion_mean", "robust_anchor_margin", "margin"}.issubset(components):
            rows["tri_consensus_margin"] = (
                0.45 * components["primary"]
                + 0.25 * components["rank_fusion_mean"]
                + 0.20 * components["robust_anchor_margin"]
                + 0.10 * components["margin"]
            )
        if {"rank_fusion_mean", "robust_anchor_margin", "rank_fusion_trimmed"}.issubset(components):
            rows["tri_consensus_trimmed"] = (
                0.45 * components["primary"]
                + 0.25 * components["rank_fusion_mean"]
                + 0.20 * components["robust_anchor_margin"]
                + 0.10 * components["rank_fusion_trimmed"]
            )
        if {"rank_fusion_mean", "robust_anchor_margin", "rank_fusion_strict"}.issubset(components):
            rows["tri_consensus_strict"] = (
                0.45 * components["primary"]
                + 0.25 * components["rank_fusion_mean"]
                + 0.20 * components["robust_anchor_margin"]
                + 0.10 * components["rank_fusion_strict"]
            )
        if "margin" in scores:
            margin_rank = rank_percentile(scores["margin"])
            for beta in [0.10, 0.20, 0.30, 0.40]:
                rows[f"dual_stage_margin_b{int(round(100 * beta)):03d}"] = (
                    (1.0 - beta) * primary_rank + beta * margin_rank
                )
            rows["balanced_profile_release"] = 0.60 * primary_rank + 0.40 * margin_rank
        if "strat_boundary_safe" in scores:
            boundary_rank = rank_percentile(scores["strat_boundary_safe"])
            for beta in [0.10, 0.20, 0.30, 0.40]:
                rows[f"dual_stage_boundary_b{int(round(100 * beta)):03d}"] = (
                    (1.0 - beta) * primary_rank + beta * boundary_rank
                )
            if "tri_consensus_margin" in rows:
                tri_rank = rank_percentile(rows["tri_consensus_margin"])
                for beta in [0.005, 0.010, 0.020, 0.030, 0.040, 0.050, 0.080, 0.100, 0.120, 0.150]:
                    rows[f"tri_boundary_b{int(round(1000 * beta)):03d}"] = (
                        (1.0 - beta) * tri_rank + beta * boundary_rank
                    )
            if "tri_consensus_margin" in rows and "margin" in scores:
                tri_rank = rank_percentile(rows["tri_consensus_margin"])
                margin_rank = rank_percentile(scores["margin"])
                support_rank = 0.50 * margin_rank + 0.50 * boundary_rank
                for beta in [0.005, 0.010, 0.020, 0.030, 0.040, 0.050, 0.080, 0.100, 0.120, 0.150]:
                    rows[f"tri_support_b{int(round(1000 * beta)):03d}"] = (
                        (1.0 - beta) * tri_rank + beta * support_rank
                    )
        if "margin" in scores and "strat_boundary_safe" in scores:
            margin_rank = rank_percentile(scores["margin"])
            boundary_rank = rank_percentile(scores["strat_boundary_safe"])
            for beta in [0.10, 0.20, 0.30]:
                support_rank = 0.5 * margin_rank + 0.5 * boundary_rank
                rows[f"dual_stage_margin_boundary_b{int(round(100 * beta)):03d}"] = (
                    (1.0 - beta) * primary_rank + beta * support_rank
                )
    return rows


def tail_guard_rank_scores(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "tri_consensus_margin" not in scores or "margin" not in scores:
        return {}
    tri = rank_percentile(scores["tri_consensus_margin"])
    margin_rank = rank_percentile(scores["margin"])
    support_parts = [margin_rank]
    if "strat_boundary_safe" in scores:
        support_parts.append(rank_percentile(scores["strat_boundary_safe"]))
    if "robust_anchor_margin" in scores:
        support_parts.append(rank_percentile(scores["robust_anchor_margin"]))
    support = np.mean(np.stack(support_parts, axis=1), axis=1)

    rows = {}
    for pivot in [0.86, 0.88, 0.90, 0.92, 0.94]:
        for sharpness in [20.0, 35.0, 55.0]:
            gate = 1.0 / (1.0 + np.exp(sharpness * (tri - pivot)))
            for beta in [0.04, 0.06, 0.08, 0.10, 0.12]:
                guarded = tri + beta * gate * (support - tri)
                rows[
                    "tri_tail_guard"
                    f"_p{int(round(100 * pivot)):03d}"
                    f"_s{int(round(sharpness)):02d}"
                    f"_b{int(round(1000 * beta)):03d}"
                ] = guarded
    return rows


def paired_stats(raw: pd.DataFrame, candidates: list[str], baselines: list[str]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        for baseline in baselines:
            for coverage in sorted(raw["coverage"].unique()):
                cand = raw[(raw["method"] == candidate) & (raw["coverage"] == coverage)].set_index("seed")
                base = raw[(raw["method"] == baseline) & (raw["coverage"] == coverage)].set_index("seed")
                common = cand.index.intersection(base.index)
                for metric in ["Accuracy", "F1_weighted", "Balanced Accuracy", "F1_macro", "MCC"]:
                    c = cand.loc[common, metric]
                    b = base.loc[common, metric]
                    diff = c - b
                    rows.append(
                        {
                            "candidate": candidate,
                            "baseline": baseline,
                            "coverage": coverage,
                            "metric": metric,
                            "n": len(diff),
                            "candidate_mean": float(c.mean()) if len(diff) else np.nan,
                            "baseline_mean": float(b.mean()) if len(diff) else np.nan,
                            "delta_mean": float(diff.mean()) if len(diff) else np.nan,
                            "wins": int((diff > 0).sum()),
                            "ties": int((diff == 0).sum()),
                            "losses": int((diff < 0).sum()),
                            "paired_t_p": float(ttest_rel(c, b).pvalue) if len(diff) > 1 else np.nan,
                            "wilcoxon_p": float(wilcoxon(diff).pvalue) if len(diff) > 1 and np.any(diff != 0) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def trace_score_selected(score_name: str, args) -> bool:
    include = getattr(args, "trace_score_include", None) or []
    if not include:
        return keep_score_name(score_name, args)
    return any(pattern in score_name for pattern in include)


def build_trace_frame(
    seed: int,
    ordered: pd.DataFrame,
    y: np.ndarray,
    posterior_suffix: str,
    proba: np.ndarray,
    geoshift: np.ndarray,
    anchor_pool: np.ndarray,
    anchors: list[np.ndarray],
    scores: dict[str, np.ndarray],
    args,
) -> pd.DataFrame:
    n_classes = proba.shape[1]
    pred = proba.argmax(axis=1)
    anchor_vote = anchor_vote_share(anchors, n_classes)
    trace = pd.DataFrame(
        {
            "seed": seed,
            "split": "extrapolation",
            "posterior_suffix": posterior_suffix,
            "row_id": np.arange(len(ordered), dtype=int),
            "well": ordered["WELL"].to_numpy(),
            "depth_md": ordered["DEPTH_MD"].to_numpy(dtype=float),
            "true": y.astype(int),
            "pred": pred.astype(int),
            "correct": (pred == y).astype(int),
            "margin_profile_anchor": margin(proba),
            "margin_geoshift": margin(geoshift),
            "margin_anchor_pool": margin(anchor_pool),
            "entropy_profile_anchor": entropy(proba),
            "anchor_vote_share": anchor_vote,
            "geoshift_anchor_agree": (geoshift.argmax(axis=1) == anchor_pool.argmax(axis=1)).astype(int),
            "js_geoshift_anchor": js_divergence(geoshift, anchor_pool),
        }
    )
    if "X_LOC" in ordered.columns:
        trace["x_loc"] = ordered["X_LOC"].to_numpy(dtype=float)
    if "Y_LOC" in ordered.columns:
        trace["y_loc"] = ordered["Y_LOC"].to_numpy(dtype=float)

    for score_name, score in scores.items():
        if trace_score_selected(score_name, args):
            trace[f"score_{score_name}"] = np.asarray(score, dtype=float)
    return trace


def run_seed(seed: int, args) -> tuple[list[dict], list[pd.DataFrame]]:
    if torch is not None:
        torch.manual_seed(seed)
    np.random.seed(seed)
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, features = build_features(df, include_missing=True)
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    train_wells, _, extra_wells, _ = split_wells_by_space(df, args.train_fraction, args.interp_test_wells, seed)
    train_wells = set(train_wells)
    train = df[df["WELL"].isin(train_wells)].copy()
    n_classes = len(class_names)
    source_prior = (
        train["TARGET"].value_counts(normalize=True)
        .reindex(range(n_classes), fill_value=0.0)
        .to_numpy(dtype=float)
    )

    classes = np.array(sorted(train["TARGET"].unique()))
    class_to_local = {cls: idx for idx, cls in enumerate(classes)}
    y_train = train["TARGET"].map(class_to_local)
    fitted_models = {}
    for name in args.models:
        model = make_model(name, seed, len(classes), args)
        if name == "gbdt":
            y_counts = pd.Series(y_train).value_counts()
            if len(y_counts) and int(y_counts.min()) < 2:
                model = disable_hist_gradient_boosting_validation(model)
        model.fit(train[features], y_train)
        fitted_models[name] = model

    fitted = fit_family_views(df, train_wells, features, seed, n_classes, args)
    frame = df[df["WELL"].isin(extra_wells)].copy()
    ordered, p_tree, p_seq = family_posteriors(fitted, frame, n_classes, args)
    y = ordered["TARGET"].to_numpy(dtype=int)
    geoshift = normalized(args.tree_weight * p_tree + (1.0 - args.tree_weight) * p_seq)
    anchors = [model_proba(fitted_models[name], ordered[features], classes, n_classes) for name in args.anchor_models if name in fitted_models]
    p_anchor = posterior_pool(anchors)
    p_proto = prototype_posterior(train, ordered, n_classes, args)

    rows = []
    traces = []
    for name in ["rf", "cat", "xgb", "lgbm"]:
        if name in fitted_models:
            p = model_proba(fitted_models[name], ordered[features], classes, n_classes)
            rows.extend(selective_rows(y, p, seed, "extrapolation", name, args.coverages))
    rows.extend(selective_rows(y, geoshift, seed, "extrapolation", "geoshift_seq", args.coverages))
    if args.prototype_fusion_weights:
        rows.extend(selective_rows(y, p_proto, seed, "extrapolation", "source_strat_response_proto", args.coverages))

    for weight in args.multi_anchor_weights:
        suffix = f"w{int(round(float(weight) * 100)):03d}"
        base_p = normalized(float(weight) * geoshift + (1.0 - float(weight)) * p_anchor)
        posterior_views = [(suffix, base_p)]
        for proto_weight in args.prototype_fusion_weights:
            proto_suffix = f"{suffix}_proto{int(round(float(proto_weight) * 100)):03d}"
            proto_p = normalized((1.0 - float(proto_weight)) * base_p + float(proto_weight) * p_proto)
            posterior_views.append((proto_suffix, proto_p))

        for posterior_suffix, p in posterior_views:
            scores = target_profile_score(ordered, p, geoshift, p_anchor, anchors, train_wells, args)
            scores.update(rank_fusion_scores(p, geoshift, p_anchor, anchors))
            scores.update(stratigraphic_continuity_scores(ordered, p, geoshift, p_anchor, scores))
            scores.update(second_stage_rank_scores(scores))
            if args.enable_tail_guard:
                scores.update(tail_guard_rank_scores(scores))
            scores.update(interval_consistency_scores(ordered, scores, args))
            if posterior_suffix != suffix:
                proto_vote = (p.argmax(axis=1) == p_proto.argmax(axis=1)).astype(float)
                proto_margin = margin(p_proto)
                scores["prototype_consensus_margin"] = (
                    0.55 * rank_percentile(scores["second_stage_rank_fusion_mean_x_robust_anchor_margin"])
                    + 0.25 * rank_percentile(proto_margin)
                    + 0.20 * rank_percentile(proto_vote)
                )
            soft_base_names = {
                "second_stage_rank_fusion_mean_x_robust_anchor_margin",
                "second_stage_weighted_rank_fusion_mean_x_robust_anchor_margin",
                "rank_fusion_mean",
                "rank_fusion_trimmed",
            }
            pred = p.argmax(axis=1)
            for base_name in sorted(soft_base_names):
                if base_name not in scores:
                    continue
                for suffix_name, soft_score in soft_class_prior_scores(pred, scores[base_name], args.soft_class_prior_beta).items():
                    scores[f"{base_name}_{suffix_name}"] = soft_score
                for suffix_name, mc_score in minority_consensus_scores(
                    pred,
                    scores[base_name],
                    p,
                    geoshift,
                    p_anchor,
                    anchors,
                    source_prior,
                    args.minority_consensus_beta,
                ).items():
                    scores[f"{base_name}_{suffix_name}"] = mc_score
            for score_name, score in scores.items():
                if not keep_score_name(score_name, args):
                    continue
                rows.extend(selective_rows(y, p, seed, "extrapolation", f"profile_anchor_{posterior_suffix}_{score_name}", args.coverages, score=score))
                if "geoshift" in args.decoupled_label_views:
                    rows.extend(
                        selective_rows(
                            y,
                            geoshift,
                            seed,
                            "extrapolation",
                            f"profile_anchor_{posterior_suffix}_{score_name}_label_geoshift",
                            args.coverages,
                            score=score,
                        )
                    )
                if "anchor_pool" in args.decoupled_label_views:
                    rows.extend(
                        selective_rows(
                            y,
                            p_anchor,
                            seed,
                            "extrapolation",
                            f"profile_anchor_{posterior_suffix}_{score_name}_label_anchor_pool",
                            args.coverages,
                            score=score,
                        )
                    )
                if score_name in args.source_prior_budget_scores:
                    for strength in args.source_prior_budget_strength:
                        for floor in args.source_prior_budget_floor:
                            for qpow in args.source_prior_budget_quality_power:
                                rows.extend(
                                    source_prior_budget_rows(
                                        y,
                                        p,
                                        seed,
                                        "extrapolation",
                                        (
                                            f"profile_anchor_{posterior_suffix}_{score_name}"
                                            f"_sourceprior_s{int(round(100 * strength)):03d}"
                                            f"_f{int(round(100 * floor)):03d}"
                                            f"_q{int(round(100 * qpow)):03d}"
                                        ),
                                        args.coverages,
                                        score,
                                        source_prior,
                                        strength,
                                        floor,
                                        qpow,
                                    )
                                )
                if score_name in {
                    "rank_fusion_mean_x_strat_boundary",
                    "rank_fusion_trimmed_x_strat_boundary",
                    "second_stage_weighted_rank_fusion_mean_x_robust_anchor_margin",
                }:
                    for gamma in args.class_quota_gamma:
                        rows.extend(
                            class_aware_selective_rows(
                                y,
                                p,
                                seed,
                                "extrapolation",
                                f"profile_anchor_{posterior_suffix}_{score_name}_classaware_g{int(round(100 * gamma)):03d}",
                                args.coverages,
                                score,
                                n_classes,
                                gamma,
                            )
                        )
            if args.trace_csv is not None:
                traces.append(
                    build_trace_frame(
                        seed,
                        ordered,
                        y,
                        posterior_suffix,
                        p,
                        geoshift,
                        p_anchor,
                        anchors,
                        scores,
                        args,
                    )
                )
    return rows, traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/force2020"))
    parser.add_argument("--target", default="FORCE_2020_LITHOFACIES_LITHOLOGY")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", default=["mlp", "rf", "xgb", "lgbm", "cat"])
    parser.add_argument("--anchor-models", nargs="+", default=["mlp", "rf", "xgb", "lgbm", "cat"])
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--interp-test-wells", type=int, default=10)
    parser.add_argument("--max-rows-per-well", type=int, default=800)
    parser.add_argument("--tree-weight", type=float, default=0.75)
    parser.add_argument("--multi-anchor-weights", nargs="+", type=float, default=[0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--coverages", nargs="+", type=float, default=DEFAULT_COVERAGES)
    parser.add_argument("--profile-top-fraction", type=float, default=0.2)
    parser.add_argument("--profile-weight", type=float, default=0.15)
    parser.add_argument("--score-smooth-windows", nargs="+", type=int, default=[])
    parser.add_argument("--score-smooth-blends", nargs="+", type=float, default=[0.5])
    parser.add_argument("--enable-tail-guard", action="store_true")
    parser.add_argument(
        "--score-include",
        nargs="+",
        default=[],
        help="Only evaluate score names containing at least one of these substrings.",
    )
    parser.add_argument(
        "--score-exclude",
        nargs="+",
        default=[],
        help="Skip score names containing any of these substrings.",
    )
    parser.add_argument("--class-quota-gamma", nargs="+", type=float, default=[])
    parser.add_argument("--soft-class-prior-beta", nargs="+", type=float, default=[])
    parser.add_argument("--minority-consensus-beta", nargs="+", type=float, default=[])
    parser.add_argument(
        "--decoupled-label-views",
        nargs="+",
        choices=["geoshift", "anchor_pool"],
        default=[],
        help="Evaluate source-only label posterior and release ranking as separate components.",
    )
    parser.add_argument("--prototype-fusion-weights", nargs="+", type=float, default=[])
    parser.add_argument(
        "--prototype-features",
        nargs="+",
        default=[
            "DEPTH_PROFILE",
            "Z_PROFILE",
            "DEPTH_MD",
            "Z_LOC",
            "GR_WELL_Z",
            "RHOB_WELL_Z",
            "NPHI_WELL_Z",
            "DTC_WELL_Z",
            "GR",
            "RHOB",
            "NPHI",
            "DTC",
        ],
    )
    parser.add_argument("--prototype-var-shrinkage", type=float, default=0.35)
    parser.add_argument("--prototype-prior-smoothing", type=float, default=2.0)
    parser.add_argument("--prototype-prior-temperature", type=float, default=0.35)
    parser.add_argument("--prototype-distance-scale", type=float, default=1.2)
    parser.add_argument("--prototype-min-class-rows", type=int, default=8)
    parser.add_argument(
        "--source-prior-budget-scores",
        nargs="+",
        default=[],
        help="Score names to evaluate with a weak source-prior class budget.",
    )
    parser.add_argument("--source-prior-budget-strength", nargs="+", type=float, default=[])
    parser.add_argument("--source-prior-budget-floor", nargs="+", type=float, default=[0.0])
    parser.add_argument("--source-prior-budget-quality-power", nargs="+", type=float, default=[0.0])
    parser.add_argument("--well-margin-weight", type=float, default=0.35)
    parser.add_argument("--well-vote-weight", type=float, default=0.20)
    parser.add_argument("--well-agree-weight", type=float, default=0.25)
    parser.add_argument("--well-entropy-weight", type=float, default=0.10)
    parser.add_argument("--distance-weight", type=float, default=0.10)
    parser.add_argument("--view-alpha", type=float, default=0.75)
    parser.add_argument("--tree-model", default="rf")
    parser.add_argument("--target-quantile", type=float, default=0.75)
    parser.add_argument("--max-augmented-multiplier", type=float, default=1.5)
    parser.add_argument("--smote-k", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--gbdt-max-iter", type=int, default=120)
    parser.add_argument("--disable-gbdt-early-stopping", action="store_true")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-max-iter", type=int, default=4000)
    parser.add_argument("--knn-neighbors", type=int, default=15)
    parser.add_argument("--mlp-max-iter", type=int, default=160)
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stnet-class-weight-power", type=float, default=0.5)
    parser.add_argument("--stnet-focal-gamma", type=float, default=0.0)
    parser.add_argument("--stnet-balanced-sampler", action="store_true")
    parser.add_argument("--stnet-sampler-power", type=float, default=0.5)
    parser.add_argument("--out-csv", type=Path, default=Path("results/profile_anchor_reliability_geoshift_seq_3seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/profile_anchor_reliability_geoshift_seq_3seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/profile_anchor_reliability_geoshift_seq_3seed_paired.csv"))
    parser.add_argument(
        "--trace-csv",
        type=Path,
        default=None,
        help="Optional sample-level extrapolation trace with labels, predictions, posterior diagnostics, and selected score columns.",
    )
    parser.add_argument(
        "--trace-score-include",
        nargs="+",
        default=[],
        help="When --trace-csv is set, only trace score names containing at least one of these substrings. Defaults to --score-include/--score-exclude filtering.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = []
    traces = []
    done = set()
    if args.resume and args.out_csv.exists():
        existing = pd.read_csv(args.out_csv)
        rows = existing.to_dict("records")
        done = set(existing["seed"].unique())
    if args.resume and args.trace_csv is not None and args.trace_csv.exists():
        traces = [pd.read_csv(args.trace_csv)]
    for seed in args.seeds:
        if seed in done:
            continue
        seed_rows, seed_traces = run_seed(seed, args)
        rows.extend(seed_rows)
        traces.extend(seed_traces)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        if args.trace_csv is not None and traces:
            args.trace_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(traces, ignore_index=True).to_csv(args.trace_csv, index=False)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    candidate_methods = [m for m in raw["method"].unique() if m.startswith("profile_anchor_")]
    paired = paired_stats(raw, candidate_methods, baselines=["rf", "geoshift_seq"])
    summary.to_csv(args.summary_csv, index=False)
    paired.to_csv(args.paired_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.paired_csv}")


if __name__ == "__main__":
    main()
