
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.preprocessing import LabelEncoder

from data.spatial_multimethod_group_benchmark import make_model
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef


DEFAULT_COVERAGES = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.20]
METRICS = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted"]


def normalized(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-12, None)
    return x / x.sum(axis=1, keepdims=True)


def margin(proba: np.ndarray) -> np.ndarray:
    p = np.sort(np.asarray(proba, dtype=float), axis=1)
    return p[:, -1] - p[:, -2] if p.shape[1] > 1 else p[:, -1]


def rank_percentile(score: np.ndarray) -> np.ndarray:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(score), endpoint=True)
    return ranks


def anchor_vote_share(anchors: list[np.ndarray], n_classes: int) -> np.ndarray:
    preds = np.stack([p.argmax(axis=1) for p in anchors], axis=0)
    return np.array(
        [np.max(np.bincount(preds[:, i], minlength=n_classes)) / preds.shape[0] for i in range(preds.shape[1])],
        dtype=float,
    )


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = normalized(np.clip(p, 1e-12, 1.0))
    q = normalized(np.clip(q, 1e-12, 1.0))
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(p) - np.log(m))).sum(axis=1)
    kl_qm = (q * (np.log(q) - np.log(m))).sum(axis=1)
    return 0.5 * (kl_pm + kl_qm) / np.log(2.0)


def rank_fusion_scores(coupled: np.ndarray, geoshift: np.ndarray, anchor_pool: np.ndarray, anchors: list[np.ndarray]) -> dict[str, np.ndarray]:
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
    rank_matrix = np.stack(ranks, axis=1)
    return {
        "rank_fusion_mean": rank_matrix.mean(axis=1),
        "rank_fusion_trimmed": np.sort(rank_matrix, axis=1)[:, 2:-1].mean(axis=1),
        "rank_fusion_strict": np.minimum.reduce(ranks[:8]),
    }


def second_stage_rank_scores(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rows = {}
    pairs = [
        ("rank_fusion_trimmed", "target_profile_anchor"),
        ("rank_fusion_trimmed", "strict_target_profile_anchor"),
        ("rank_fusion_mean", "target_profile_anchor"),
        ("rank_fusion_mean", "robust_anchor_margin"),
    ]
    for left, right in pairs:
        if left in scores and right in scores:
            left_rank = rank_percentile(scores[left])
            right_rank = rank_percentile(scores[right])
            rows[f"second_stage_{left}_x_{right}"] = 0.5 * left_rank + 0.5 * right_rank
            rows[f"second_stage_weighted_{left}_x_{right}"] = 0.65 * left_rank + 0.35 * right_rank
    primary = rows.get("second_stage_rank_fusion_mean_x_robust_anchor_margin")
    if primary is not None:
        primary_rank = rank_percentile(primary)
        if {"rank_fusion_mean", "robust_anchor_margin"}.issubset(scores):
            rows["tri_consensus_trimmed"] = (
                0.45 * primary_rank
                + 0.25 * rank_percentile(scores["rank_fusion_mean"])
                + 0.20 * rank_percentile(scores["robust_anchor_margin"])
                + 0.10 * rank_percentile(scores.get("rank_fusion_trimmed", scores["rank_fusion_mean"]))
            )
    return rows


def metric_row(y_true: np.ndarray, pred: np.ndarray, labels: np.ndarray | None = None) -> dict[str, float]:
    labels = np.unique(np.concatenate([np.asarray(y_true, dtype=int), np.asarray(pred, dtype=int)])) if labels is None else np.asarray(labels, dtype=int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "MCC": matthews_corrcoef(y_true, pred),
        "F1_macro": f1_score(y_true, pred, average="macro", labels=labels, zero_division=0),
        "F1_weighted": f1_score(y_true, pred, average="weighted", labels=labels, zero_division=0),
    }


def model_proba(model, x: pd.DataFrame, classes: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        local = model.predict_proba(x)
    else:
        pred = model.predict(x)
        local = np.zeros((len(pred), len(classes)), dtype=float)
        for i, cls in enumerate(classes):
            local[:, i] = (pred == i).astype(float)
    out = np.zeros((local.shape[0], n_classes), dtype=float)
    for j, cls in enumerate(classes):
        out[:, int(cls)] = local[:, j]
    return normalized(out)


def posterior_pool(posteriors: list[np.ndarray]) -> np.ndarray:
    if not posteriors:
        raise ValueError("posterior_pool requires at least one posterior")
    return normalized(np.mean(np.stack(posteriors, axis=0), axis=0))


def selective_rows(y: np.ndarray, proba: np.ndarray, seed: int, split: str, method: str, coverages: list[float], score=None) -> list[dict]:
    pred = proba.argmax(axis=1)
    score = margin(proba) if score is None else np.asarray(score, dtype=float)
    order = np.argsort(score)[::-1]
    rows = []
    for coverage in coverages:
        keep_n = max(1, int(round(len(y) * coverage)))
        keep = order[:keep_n]
        row = metric_row(y[keep], pred[keep])
        row.update({"seed": seed, "split": split, "method": method, "coverage": coverage, "kept_rows": keep_n})
        rows.append(row)
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted", "kept_rows"]
    summary = raw.groupby(["method", "split", "coverage"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def add_well_zscore(df: pd.DataFrame, logs: list[str]) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("WELL", sort=False)
    for feature in logs:
        mean = grouped[feature].transform("mean")
        std = grouped[feature].transform("std").replace(0, np.nan)
        out[f"{feature}_WELL_Z"] = (out[feature] - mean) / std
    return out


def load_external(path: Path, well_col: str, depth_col: str, label_col: str, logs: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(path)
    missing = [c for c in [well_col, depth_col, label_col] + logs if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.rename(columns={well_col: "WELL", depth_col: "DEPTH_MD", label_col: "LABEL_RAW"}).copy()
    df = df[df["WELL"].notna() & df["DEPTH_MD"].notna() & df["LABEL_RAW"].notna()].copy()
    df["WELL"] = df["WELL"].astype(str)
    encoder = LabelEncoder()
    df["TARGET"] = encoder.fit_transform(df["LABEL_RAW"].astype(str))
    df = add_well_zscore(df, logs)
    features = ["DEPTH_MD"] + logs + [f"{c}_WELL_Z" for c in logs]
    for feature in list(features):
        miss = f"{feature}_MISSING"
        df[miss] = df[feature].isna().astype(np.float32)
        features.append(miss)
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    return df, features, list(encoder.classes_)


def split_wells(df: pd.DataFrame, seed: int, test_wells: int) -> tuple[set[str], set[str]]:
    wells = np.array(sorted(df["WELL"].unique()))
    if len(wells) <= test_wells:
        raise ValueError(f"Need more wells than test_wells={test_wells}; found {len(wells)}")
    rng = np.random.default_rng(seed)
    held = set(rng.choice(wells, size=test_wells, replace=False))
    return set(wells) - held, held


def fit_models(train: pd.DataFrame, features: list[str], n_classes: int, seed: int, args) -> tuple[dict, np.ndarray]:
    classes = np.array(sorted(train["TARGET"].unique()))
    class_to_local = {cls: idx for idx, cls in enumerate(classes)}
    y_train = train["TARGET"].map(class_to_local)
    models = {}
    for name in args.models:
        model = make_model(name, seed, len(classes), args)
        model.fit(train[features], y_train)
        models[name] = model
    return models, classes


def robust_anchor_margin(coupled: np.ndarray, anchor_pool: np.ndarray, anchors: list[np.ndarray]) -> np.ndarray:
    n_classes = coupled.shape[1]
    vote = anchor_vote_share(anchors, n_classes)
    agree = (coupled.argmax(axis=1) == anchor_pool.argmax(axis=1)).astype(float)
    min_m = np.minimum(margin(coupled), margin(anchor_pool))
    return 0.50 * margin(coupled) + 0.20 * min_m + 0.20 * vote + 0.10 * agree - 0.10 * rank_percentile(js_divergence(coupled, anchor_pool))


def paired_stats(raw: pd.DataFrame, candidate: str, baselines: list[str]) -> pd.DataFrame:
    rows = []
    for baseline in baselines:
        for coverage in sorted(raw["coverage"].unique()):
            cand = raw[(raw["method"] == candidate) & (raw["coverage"] == coverage)].set_index("seed")
            base = raw[(raw["method"] == baseline) & (raw["coverage"] == coverage)].set_index("seed")
            common = cand.index.intersection(base.index)
            for metric in METRICS:
                c = cand.loc[common, metric]
                b = base.loc[common, metric]
                diff = c.to_numpy() - b.to_numpy()
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


def run_seed(df: pd.DataFrame, features: list[str], n_classes: int, seed: int, args) -> tuple[list[dict], dict]:
    train_wells, test_wells = split_wells(df, seed, args.test_wells)
    train = df[df["WELL"].isin(train_wells)].copy()
    test = df[df["WELL"].isin(test_wells)].copy()
    models, classes = fit_models(train, features, n_classes, seed, args)
    y = test["TARGET"].to_numpy(dtype=int)
    anchors = [model_proba(models[name], test[features], classes, n_classes) for name in args.anchor_models if name in models]
    anchor_pool = posterior_pool(anchors)
    coupled = anchor_pool

    rows = []
    for name in args.models:
        proba = model_proba(models[name], test[features], classes, n_classes)
        rows.extend(selective_rows(y, proba, seed, "external", name, args.coverages))
        pred = proba.argmax(axis=1)
        full = metric_row(y, pred)
        full.update({"seed": seed, "split": "external", "method": name, "coverage": 1.0, "kept_rows": len(y)})
        rows.append(full)
    for weight in args.anchor_weights:
        method_suffix = f"w{int(round(weight * 100)):03d}"
        p = normalized(weight * coupled + (1.0 - weight) * anchor_pool)
        scores = rank_fusion_scores(p, coupled, anchor_pool, anchors)
        scores["robust_anchor_margin"] = robust_anchor_margin(p, anchor_pool, anchors)
        scores.update(second_stage_rank_scores(scores))
        scores["external_profile_anchor"] = 0.55 * rank_percentile(scores["rank_fusion_mean"]) + 0.45 * rank_percentile(
            scores["robust_anchor_margin"]
        )
        for score_name, score in scores.items():
            if score_name not in args.release_scores:
                continue
            method = f"profile_anchor_external_{method_suffix}_{score_name}"
            rows.extend(selective_rows(y, p, seed, "external", method, args.coverages, score=score))
            pred = p.argmax(axis=1)
            full = metric_row(y, pred)
            full.update({"seed": seed, "split": "external", "method": method, "coverage": 1.0, "kept_rows": len(y)})
            rows.append(full)

    manifest = {
        "seed": seed,
        "train_wells": ",".join(sorted(train_wells)),
        "test_wells": ",".join(sorted(test_wells)),
        "train_rows": len(train),
        "test_rows": len(test),
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--dataset-name", default="external")
    parser.add_argument("--well-col", default="WELL")
    parser.add_argument("--depth-col", default="DEPTH_MD")
    parser.add_argument("--label-col", default="LITHOLOGY")
    parser.add_argument("--logs", nargs="+", default=["GR", "RHOB", "NPHI", "DTC"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--test-wells", type=int, default=3)
    parser.add_argument("--coverages", nargs="+", type=float, default=DEFAULT_COVERAGES)
    parser.add_argument("--models", nargs="+", default=["rf", "xgb", "lgbm", "cat", "mlp"])
    parser.add_argument("--anchor-models", nargs="+", default=["rf", "xgb", "lgbm", "cat", "mlp"])
    parser.add_argument("--anchor-weights", nargs="+", type=float, default=[0.44])
    parser.add_argument(
        "--release-scores",
        nargs="+",
        default=[
            "external_profile_anchor",
            "rank_fusion_mean",
            "rank_fusion_trimmed",
            "robust_anchor_margin",
            "second_stage_rank_fusion_mean_x_robust_anchor_margin",
            "second_stage_weighted_rank_fusion_mean_x_robust_anchor_margin",
        ],
    )
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
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--paired-csv", type=Path, default=None)
    parser.add_argument("--split-manifest-csv", type=Path, default=None)
    args = parser.parse_args()

    out_base = Path("results")
    args.out_csv = args.out_csv or out_base / f"{args.dataset_name}_profile_anchor_external.csv"
    args.summary_csv = args.summary_csv or out_base / f"{args.dataset_name}_profile_anchor_external_summary.csv"
    args.paired_csv = args.paired_csv or out_base / f"{args.dataset_name}_profile_anchor_external_paired.csv"
    args.split_manifest_csv = args.split_manifest_csv or out_base / f"{args.dataset_name}_split_manifest.csv"

    df, features, class_names = load_external(args.csv, args.well_col, args.depth_col, args.label_col, args.logs)
    rows = []
    manifests = []
    for seed in args.seeds:
        seed_rows, manifest = run_seed(df, features, len(class_names), seed, args)
        rows.extend(seed_rows)
        manifests.append(manifest)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        pd.DataFrame(manifests).to_csv(args.split_manifest_csv, index=False)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    candidates = [m for m in raw["method"].unique() if m.startswith("profile_anchor_external")]
    paired = pd.concat([paired_stats(raw, c, [m for m in args.models if m in raw["method"].unique()]) for c in candidates], ignore_index=True)
    summary.to_csv(args.summary_csv, index=False)
    paired.to_csv(args.paired_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.paired_csv}")
    print(f"Wrote {args.split_manifest_csv}")


if __name__ == "__main__":
    main()
