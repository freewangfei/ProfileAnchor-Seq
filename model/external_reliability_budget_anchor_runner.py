
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.exceptions import UndefinedMetricWarning

from model.external_welllog_profile_anchor_runner import (
    DEFAULT_COVERAGES,
    anchor_vote_share,
    js_divergence,
    margin,
    metric_row,
    model_proba,
    normalized,
    posterior_pool,
    rank_percentile,
    selective_rows,
    split_wells,
    summarize,
)
from data.spatial_multimethod_group_benchmark import make_model


METRICS = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted"]


def add_features(df: pd.DataFrame, logs: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.sort_values(["WELL", "DEPTH_MD"]).copy()
    features: list[str] = []
    base_depth = [c for c in ["DEPTH_MD", "TOP_DEPTH", "BOT_DEPTH"] if c in out.columns]
    features.extend(base_depth)
    if {"TOP_DEPTH", "BOT_DEPTH"}.issubset(out.columns):
        out["INTERVAL_THICKNESS"] = out["BOT_DEPTH"] - out["TOP_DEPTH"]
        out["INTERVAL_MID_DEPTH"] = 0.5 * (out["BOT_DEPTH"] + out["TOP_DEPTH"])
        features.extend(["INTERVAL_THICKNESS", "INTERVAL_MID_DEPTH"])
    features.extend([c for c in logs if c in out.columns])

    grouped = out.groupby("WELL", sort=False)
    for feature in [c for c in logs if c in out.columns]:
        mean = grouped[feature].transform("mean")
        std = grouped[feature].transform("std").replace(0, np.nan)
        out[f"{feature}_WELL_Z"] = (out[feature] - mean) / std
        out[f"{feature}_ROLL5"] = grouped[feature].transform(lambda s: s.rolling(5, min_periods=1, center=True).mean())
        out[f"{feature}_ROLL11"] = grouped[feature].transform(lambda s: s.rolling(11, min_periods=1, center=True).mean())
        out[f"{feature}_GRAD"] = grouped[feature].transform(lambda s: s.diff().fillna(0.0))
        features.extend([f"{feature}_WELL_Z", f"{feature}_ROLL5", f"{feature}_ROLL11", f"{feature}_GRAD"])

    if "GR" in out.columns:
        for denom in ["AC", "DEN", "PEF", "LLD", "LLS", "SP", "CALI"]:
            if denom in out.columns:
                col = f"GR_OVER_{denom}"
                out[col] = out["GR"] / (np.abs(out[denom]) + 1e-3)
                features.append(col)
    if {"LLD", "LLS"}.issubset(out.columns):
        out["LLD_LLS_LOG_RATIO"] = np.log1p(np.abs(out["LLD"])) - np.log1p(np.abs(out["LLS"]))
        features.append("LLD_LLS_LOG_RATIO")

    for feature in list(features):
        miss = f"{feature}_MISSING"
        out[miss] = out[feature].isna().astype(np.float32)
        features.append(miss)
    return out, features


def load_external(path: Path, well_col: str, depth_col: str, label_col: str, logs: list[str]):
    df = pd.read_csv(path)
    required = [well_col, depth_col, label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.rename(columns={well_col: "WELL", depth_col: "DEPTH_MD", label_col: "LABEL_RAW"}).copy()
    df = df[df["WELL"].notna() & df["DEPTH_MD"].notna() & df["LABEL_RAW"].notna()].copy()
    df["WELL"] = df["WELL"].astype(str)
    encoder = LabelEncoder()
    df["TARGET"] = encoder.fit_transform(df["LABEL_RAW"].astype(str))
    df, features = add_features(df, logs)
    return df.reset_index(drop=True), features, list(encoder.classes_)


def local_model(name: str, seed: int, n_classes: int, args):
    if name == "et_deep":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=args.n_estimators,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed,
                n_jobs=args.n_jobs,
            ),
        )
    if name == "rf_deep":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=args.n_estimators,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=args.n_jobs,
            ),
        )
    if name == "hgb_bal":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=args.learning_rate,
                max_iter=args.gbdt_max_iter,
                max_leaf_nodes=31,
                max_depth=args.max_depth,
                min_samples_leaf=12,
                l2_regularization=1e-4,
                class_weight="balanced",
                early_stopping=False,
                random_state=seed,
            ),
        )
    return make_model(name, seed, n_classes, args)


def fit_models(train: pd.DataFrame, features: list[str], n_classes: int, seed: int, args):
    classes = np.array(sorted(train["TARGET"].unique()))
    mapper = {cls: idx for idx, cls in enumerate(classes)}
    y = train["TARGET"].map(mapper)
    models = {}
    for name in args.models:
        model = local_model(name, seed, len(classes), args)
        model.fit(train[features], y)
        models[name] = model
    return models, classes


def smooth_by_well(frame: pd.DataFrame, proba: np.ndarray, window: int, blend: float) -> np.ndarray:
    if window <= 1:
        return normalized(proba)
    out = np.zeros_like(proba)
    work = frame.reset_index(drop=True)
    for _, group in work.groupby("WELL", sort=False):
        idx = group.index.to_numpy()
        rolled = pd.DataFrame(proba[idx]).rolling(window=window, min_periods=1, center=True).mean().to_numpy()
        out[idx] = float(blend) * proba[idx] + (1.0 - float(blend)) * rolled
    return normalized(out)


def score_bank(post: dict[str, np.ndarray], p: np.ndarray) -> dict[str, np.ndarray]:
    names = sorted(post)
    anchors = [post[n] for n in names]
    pool = posterior_pool(anchors)
    vote = anchor_vote_share(anchors, p.shape[1])
    agree = np.mean([(q.argmax(axis=1) == p.argmax(axis=1)).astype(float) for q in anchors], axis=0)
    div = np.mean([js_divergence(q, p) for q in anchors], axis=0)
    scores = {
        "margin": margin(p),
        "pool_margin": margin(pool),
        "consensus_margin": (
            0.42 * rank_percentile(margin(p))
            + 0.24 * rank_percentile(margin(pool))
            + 0.22 * rank_percentile(vote)
            + 0.16 * rank_percentile(agree)
            - 0.04 * rank_percentile(div)
        ),
        "strict_consensus": np.minimum.reduce([
            rank_percentile(margin(p)),
            rank_percentile(margin(pool)),
            rank_percentile(vote),
            rank_percentile(agree),
        ]),
    }
    return scores


def posterior_bank(post: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(post)
    names = sorted(post)
    if names:
        out["pool_all"] = posterior_pool([post[n] for n in names])
    tree_names = [n for n in ["cat", "xgb", "lgbm", "rf_deep", "et_deep", "hgb_bal"] if n in post]
    if tree_names:
        out["pool_tree"] = posterior_pool([post[n] for n in tree_names])
    if {"cat", "et_deep"}.issubset(post):
        for w in (0.35, 0.50):
            out[f"cat_et_w{int(w*100):02d}"] = normalized((1.0 - w) * post["cat"] + w * post["et_deep"])
    if {"cat", "hgb_bal"}.issubset(post):
        for w in (0.25, 0.35):
            out[f"cat_hgb_w{int(w*100):02d}"] = normalized((1.0 - w) * post["cat"] + w * post["hgb_bal"])
    if {"xgb", "lgbm"}.issubset(post):
        out["xgb_lgbm_pool"] = posterior_pool([post["xgb"], post["lgbm"]])
    return out


@dataclass(frozen=True)
class Policy:
    posterior: str
    score: str
    budget: str
    pool_multiplier: float
    min_class_share: float

    @property
    def name(self) -> str:
        return (
            f"{self.posterior}|{self.score}|{self.budget}|"
            f"x{self.pool_multiplier:.1f}|m{self.min_class_share:.2f}"
        )


def allowed_policy(policy: Policy, coverage: float) -> bool:
    if coverage <= 0.05 and policy.budget == "global":
        return False
    if coverage >= 0.20 and policy.posterior in {"hgb_bal", "cat_hgb_w25", "cat_hgb_w35"}:
        return True
    if coverage >= 0.20 and policy.budget == "global":
        return False
    return True


def source_prior(source: pd.DataFrame, n_classes: int, smoothing: float) -> np.ndarray:
    counts = source["TARGET"].value_counts().reindex(range(n_classes), fill_value=0).to_numpy(dtype=float)
    counts = counts + smoothing
    return counts / counts.sum()


def select_with_budget(
    proba: np.ndarray,
    score: np.ndarray,
    coverage: float,
    prior: np.ndarray,
    policy: Policy,
) -> np.ndarray:
    n = len(score)
    keep_n = max(1, int(round(n * coverage)))
    pred = proba.argmax(axis=1)
    order = np.argsort(score)[::-1]
    if policy.budget == "global":
        return order[:keep_n]

    if policy.budget == "uniform":
        share = np.ones(proba.shape[1], dtype=float) / proba.shape[1]
    elif policy.budget == "source":
        uniform = np.ones(proba.shape[1], dtype=float) / proba.shape[1]
        share = (1.0 - policy.min_class_share) * prior + policy.min_class_share * uniform
    elif policy.budget == "predicted":
        counts = np.bincount(pred, minlength=proba.shape[1]).astype(float) + 1.0
        pred_share = counts / counts.sum()
        uniform = np.ones(proba.shape[1], dtype=float) / proba.shape[1]
        share = 0.70 * pred_share + 0.30 * uniform
    else:
        raise ValueError(f"Unknown budget: {policy.budget}")
    share = share / share.sum()
    pool_n = min(n, max(keep_n, int(np.ceil(keep_n * policy.pool_multiplier))))
    pool = order[:pool_n]
    quotas = np.floor(share * keep_n).astype(int)

    selected: list[int] = []
    used = np.zeros(n, dtype=bool)
    for cls in range(proba.shape[1]):
        idx = pool[pred[pool] == cls]
        quota = int(quotas[cls])
        if quota <= 0 or len(idx) == 0:
            continue
        take = idx[np.argsort(score[idx])[::-1][: min(quota, len(idx))]]
        selected.extend(take.tolist())
        used[take] = True
    if len(selected) < keep_n:
        fill = order[~used[order]][: keep_n - len(selected)]
        selected.extend(fill.tolist())
    return np.array(selected[:keep_n], dtype=int)


def policy_rows(y: np.ndarray, proba: np.ndarray, score: np.ndarray, seed: int, method: str, coverages: list[float], prior: np.ndarray, policy: Policy):
    pred = proba.argmax(axis=1)
    rows = []
    for coverage in coverages:
        keep = select_with_budget(proba, score, coverage, prior, policy)
        row = metric_row(y[keep], pred[keep])
        row.update({"seed": seed, "split": "external", "method": method, "coverage": coverage, "kept_rows": len(keep)})
        rows.append(row)
    return rows


def candidate_rows(frame: pd.DataFrame, post: dict[str, np.ndarray], y: np.ndarray, coverages: list[float], prior: np.ndarray, policies: list[Policy], seed: int, split: str):
    rows = []
    banks = posterior_bank(post)
    for pname, p in banks.items():
        scores = score_bank(post, p)
        for policy in policies:
            if policy.posterior != pname or policy.score not in scores:
                continue
            method = f"rba_{policy.name}"
            pred = p.argmax(axis=1)
            for coverage in coverages:
                keep = select_with_budget(p, scores[policy.score], coverage, prior, policy)
                row = metric_row(y[keep], pred[keep])
                row.update({"seed": seed, "split": split, "method": method, "coverage": coverage, "kept_rows": len(keep)})
                rows.append(row)
    return rows


def make_policy_grid(post_names: list[str]) -> list[Policy]:
    policies: list[Policy] = []
    preferred = [
        "pool_all",
        "pool_tree",
        "cat_hgb_w25",
        "cat_hgb_w35",
        "cat_et_w35",
        "cat_et_w50",
        "xgb_lgbm_pool",
        "cat",
        "et_deep",
        "hgb_bal",
    ]
    active = [name for name in preferred if name in post_names]
    for posterior in active:
        for score in ["margin", "consensus_margin"]:
            policies.append(Policy(posterior, score, "global", 1.0, 0.0))
            for budget in ["source", "uniform"]:
                for pool_multiplier in [1.4, 2.2]:
                    for min_class_share in [0.08]:
                        policies.append(Policy(posterior, score, budget, pool_multiplier, min_class_share))
    return policies


def select_policies(source: pd.DataFrame, features: list[str], n_classes: int, outer_seed: int, args):
    wells = np.array(sorted(source["WELL"].unique()))
    rng = np.random.default_rng(outer_seed)
    rng.shuffle(wells)
    folds = np.array_split(wells, min(args.inner_folds, len(wells)))
    rows = []
    all_policies: list[Policy] | None = None
    for fold_id, held_wells in enumerate(folds):
        inner_train = source[~source["WELL"].isin(set(held_wells.tolist()))].copy()
        inner_val = source[source["WELL"].isin(set(held_wells.tolist()))].copy()
        if inner_train.empty or inner_val.empty or inner_train["TARGET"].nunique() < 2:
            continue
        models, classes = fit_models(inner_train, features, n_classes, outer_seed * 100 + fold_id, args)
        post = {
            name: smooth_by_well(inner_val, model_proba(model, inner_val[features], classes, n_classes), args.smooth_window, args.smooth_blend)
            for name, model in models.items()
        }
        banks = posterior_bank(post)
        if all_policies is None:
            all_policies = make_policy_grid(sorted(banks))
        prior = source_prior(inner_train, n_classes, args.prior_smoothing)
        y_val = inner_val["TARGET"].to_numpy(dtype=int)
        rows.extend(candidate_rows(inner_val, post, y_val, args.coverages, prior, all_policies, -1, "inner"))
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("No inner validation rows were generated")
    selected: dict[float, str] = {}
    for coverage, cov_raw in raw.groupby("coverage"):
        cov_raw = cov_raw[cov_raw["method"].map(lambda name: allowed_policy(parse_policy(str(name)), float(coverage)))]
        summary = cov_raw.groupby("method", as_index=False)[METRICS].mean()
        metric_weights = {
            "Accuracy": 1.0,
            "F1_weighted": 1.0,
            "Balanced Accuracy": 1.25 if coverage >= 0.08 else 1.0,
            "F1_macro": 1.25 if coverage >= 0.08 else 1.0,
            "MCC": 1.75 if coverage <= 0.05 else 1.15,
        }
        score = np.zeros(len(summary), dtype=float)
        for metric in METRICS:
            score += metric_weights[metric] * summary[metric].rank(ascending=False, method="average").to_numpy()
        summary["selection_rank"] = score
        selected[float(coverage)] = str(summary.sort_values(["selection_rank", "method"]).iloc[0]["method"])
    return selected, raw


def parse_policy(method: str) -> Policy:
    payload = method.removeprefix("rba_")
    posterior, score, budget, xpart, mpart = payload.split("|")
    return Policy(posterior, score, budget, float(xpart[1:]), float(mpart[1:]))


def run_seed(df: pd.DataFrame, features: list[str], n_classes: int, seed: int, args):
    train_wells, test_wells = split_wells(df, seed, args.test_wells)
    source = df[df["WELL"].isin(train_wells)].copy()
    target = df[df["WELL"].isin(test_wells)].copy()
    selected, inner_raw = select_policies(source, features, n_classes, seed, args)
    models, classes = fit_models(source, features, n_classes, seed, args)
    post = {
        name: smooth_by_well(target, model_proba(model, target[features], classes, n_classes), args.smooth_window, args.smooth_blend)
        for name, model in models.items()
    }
    banks = posterior_bank(post)
    prior = source_prior(source, n_classes, args.prior_smoothing)
    y = target["TARGET"].to_numpy(dtype=int)
    rows = []
    for name, p in post.items():
        rows.extend(selective_rows(y, p, seed, "external", name, args.coverages))
        full = metric_row(y, p.argmax(axis=1))
        full.update({"seed": seed, "split": "external", "method": name, "coverage": 1.0, "kept_rows": len(y)})
        rows.append(full)
    manifest_rows = []
    for coverage in args.coverages:
        selected_method = selected[float(coverage)]
        policy = parse_policy(selected_method)
        p = banks[policy.posterior]
        scores = score_bank(post, p)
        rows.extend(policy_rows(y, p, scores[policy.score], seed, "ReliabilityBudgetAnchor", [coverage], prior, policy))
        manifest_rows.append(
            {
                "seed": seed,
                "coverage": coverage,
                "train_wells": ",".join(sorted(train_wells)),
                "test_wells": ",".join(sorted(test_wells)),
                "selected_policy": selected_method,
                "inner_rows": len(inner_raw),
            }
        )
    p_full = banks["pool_tree"] if "pool_tree" in banks else banks["pool_all"]
    full = metric_row(y, p_full.argmax(axis=1))
    full.update({"seed": seed, "split": "external", "method": "ReliabilityBudgetAnchor", "coverage": 1.0, "kept_rows": len(y)})
    rows.append(full)
    return rows, manifest_rows, inner_raw


def main() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", message="A single label was found*")
    warnings.filterwarnings("ignore", message="X does not have valid feature names*")
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true*")
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--dataset-name", default="figshare_reliability_budget_anchor_11seed")
    parser.add_argument("--well-col", default="WELL")
    parser.add_argument("--depth-col", default="DEPTH_MD")
    parser.add_argument("--label-col", default="LITHOLOGY")
    parser.add_argument("--logs", nargs="+", default=["GR", "AC", "DEN", "PEF", "LLD", "LLS", "SP", "CALI"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(11)))
    parser.add_argument("--test-wells", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--coverages", nargs="+", type=float, default=DEFAULT_COVERAGES + [0.40])
    parser.add_argument("--models", nargs="+", default=["rf_deep", "et_deep", "hgb_bal", "cat", "xgb", "lgbm", "mlp"])
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--smooth-blend", type=float, default=0.82)
    parser.add_argument("--prior-smoothing", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=140)
    parser.add_argument("--gbdt-max-iter", type=int, default=140)
    parser.add_argument("--disable-gbdt-early-stopping", action="store_true")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-max-iter", type=int, default=4000)
    parser.add_argument("--knn-neighbors", type=int, default=15)
    parser.add_argument("--mlp-max-iter", type=int, default=160)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--inner-csv", type=Path, default=None)
    args = parser.parse_args()

    out_base = Path("results")
    args.out_csv = args.out_csv or out_base / f"{args.dataset_name}.csv"
    args.summary_csv = args.summary_csv or out_base / f"{args.dataset_name}_summary.csv"
    args.manifest_csv = args.manifest_csv or out_base / f"{args.dataset_name}_manifest.csv"
    args.inner_csv = args.inner_csv or out_base / f"{args.dataset_name}_inner.csv"

    df, features, class_names = load_external(args.csv, args.well_col, args.depth_col, args.label_col, args.logs)
    rows = []
    manifests = []
    inner_rows = []
    for seed in args.seeds:
        seed_rows, manifest_rows, inner_raw = run_seed(df, features, len(class_names), seed, args)
        rows.extend(seed_rows)
        manifests.extend(manifest_rows)
        tmp = inner_raw.copy()
        tmp["outer_seed"] = seed
        inner_rows.append(tmp)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        pd.DataFrame(manifests).to_csv(args.manifest_csv, index=False)
        pd.concat(inner_rows, ignore_index=True).to_csv(args.inner_csv, index=False)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    summary.to_csv(args.summary_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.manifest_csv}")
    print(f"Wrote {args.inner_csv}")


if __name__ == "__main__":
    main()
