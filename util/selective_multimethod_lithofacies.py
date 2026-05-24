import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

from model.spatial_lithofacies_selective_geoshift_seq import margin
from model.spatial_lithofacies_tree_stnet_posterior_fusion import family_posteriors, fit_family_views, normalized
from data.spatial_multimethod_group_benchmark import build_features, load_force, make_model, sample_by_well, split_wells_by_space

try:
    import torch
except Exception:
    torch = None


DEFAULT_COVERAGES = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01]


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "MCC": matthews_corrcoef(y_true, pred),
        "F1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
    }


def globalize(local: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((local.shape[0], n_classes), dtype=float)
    for j, cls in enumerate(classes):
        out[:, int(cls)] = local[:, j]
    out += 1e-12
    return out / out.sum(axis=1, keepdims=True)


def model_proba(model, x: pd.DataFrame, classes: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        local = model.predict_proba(x)
    else:
        local = None
    if local is None and hasattr(model, "decision_function"):
        local = softmax(model.decision_function(x), axis=1)
    if local is None:
        pred = np.asarray(model.predict(x), dtype=int).reshape(-1)
        local = np.zeros((len(pred), len(classes)), dtype=float)
        local[np.arange(len(pred)), pred] = 1.0
    if isinstance(local, list):
        local = np.column_stack([p[:, 1] for p in local])
    return globalize(np.asarray(local, dtype=float), classes, n_classes)


def selective_rows(y: np.ndarray, proba: np.ndarray, seed: int, split: str, method: str, coverages: list[float], score: np.ndarray | None = None) -> list[dict]:
    pred = proba.argmax(axis=1)
    if score is None:
        score = margin(proba)
    order = np.argsort(score)[::-1]
    rows = []
    for coverage in coverages:
        keep_n = max(1, int(round(len(y) * coverage)))
        keep = order[:keep_n]
        row = metric_row(y[keep], pred[keep])
        row.update(
            {
                "seed": seed,
                "split": split,
                "method": method,
                "coverage": coverage,
                "kept_rows": keep_n,
            }
        )
        rows.append(row)
    return rows


def posterior_pool(posteriors: list[np.ndarray]) -> np.ndarray:
    if not posteriors:
        raise ValueError("At least one posterior is required.")
    return normalized(np.stack(posteriors, axis=0).mean(axis=0))


def multi_anchor_score(geoshift: np.ndarray, anchor_pool: np.ndarray, anchors: list[np.ndarray]) -> np.ndarray:
    anchor_preds = np.stack([p.argmax(axis=1) for p in anchors], axis=0)
    agreement = np.array(
        [
            np.max(np.bincount(anchor_preds[:, i], minlength=geoshift.shape[1])) / anchor_preds.shape[0]
            for i in range(anchor_preds.shape[1])
        ],
        dtype=float,
    )
    pool_agree = (geoshift.argmax(axis=1) == anchor_pool.argmax(axis=1)).astype(float)
    anchor_margin = np.mean([margin(p) for p in anchors], axis=0)
    return 0.30 * margin(geoshift) + 0.30 * margin(anchor_pool) + 0.20 * anchor_margin + 0.20 * (0.5 * agreement + 0.5 * pool_agree)


def multi_anchor_conservative_score(geoshift: np.ndarray, anchor_pool: np.ndarray, anchors: list[np.ndarray], coupled: np.ndarray) -> np.ndarray:
    anchor_preds = np.stack([p.argmax(axis=1) for p in anchors], axis=0)
    vote_share = np.array(
        [
            np.max(np.bincount(anchor_preds[:, i], minlength=geoshift.shape[1])) / anchor_preds.shape[0]
            for i in range(anchor_preds.shape[1])
        ],
        dtype=float,
    )
    pool_agree = (geoshift.argmax(axis=1) == anchor_pool.argmax(axis=1)).astype(float)
    anchor_margin = np.mean([margin(p) for p in anchors], axis=0)
    min_margin = np.minimum.reduce([margin(geoshift), margin(anchor_pool), anchor_margin, margin(coupled)])
    return 1.2 * pool_agree + 0.8 * vote_share + 0.8 * min_margin + 0.3 * margin(coupled)


def agreement_gate_score(primary: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    agree = (primary.argmax(axis=1) == anchor.argmax(axis=1)).astype(float)
    return 2.0 * agree + 0.50 * np.minimum(margin(primary), margin(anchor)) + 0.25 * margin(primary) + 0.25 * margin(anchor)


def triple_agreement_score(p_tree: np.ndarray, p_seq: np.ndarray, p_anchor: np.ndarray, coupled: np.ndarray) -> np.ndarray:
    preds = np.stack([p_tree.argmax(axis=1), p_seq.argmax(axis=1), p_anchor.argmax(axis=1)], axis=0)
    agreement = np.array(
        [
            np.max(np.bincount(preds[:, i], minlength=coupled.shape[1])) / preds.shape[0]
            for i in range(preds.shape[1])
        ],
        dtype=float,
    )
    all_agree = ((preds[0] == preds[1]) & (preds[0] == preds[2])).astype(float)
    min_margin = np.minimum.reduce([margin(p_tree), margin(p_seq), margin(p_anchor), margin(coupled)])
    return 2.5 * all_agree + 1.0 * agreement + 0.6 * min_margin + 0.4 * margin(coupled)


def adaptive_reliability_rows(
    y: np.ndarray,
    geoshift: np.ndarray,
    anchor_single: np.ndarray,
    anchor_pool: np.ndarray,
    anchors: list[np.ndarray],
    seed: int,
    split: str,
    coverages: list[float],
    args,
) -> list[dict]:
    rows = []
    p_low = normalized(args.anchor_weight * geoshift + (1.0 - args.anchor_weight) * anchor_single)
    p_mid = normalized(args.multi_anchor_weight * geoshift + (1.0 - args.multi_anchor_weight) * anchor_pool)
    p_high = normalized(args.high_coverage_multi_anchor_weight * geoshift + (1.0 - args.high_coverage_multi_anchor_weight) * anchor_pool)
    score_low = agreement_gate_score(geoshift, anchor_single)
    score_mid = margin(p_mid)
    score_high = margin(p_high)
    for coverage in coverages:
        if coverage <= args.low_coverage_cutoff:
            proba, score = p_low, score_low
        elif coverage <= args.mid_coverage_cutoff:
            proba, score = p_mid, score_mid
        else:
            proba, score = p_high, score_high
        rows.extend(selective_rows(y, proba, seed, split, "adaptive_multi_anchor_geoshift_seq", [coverage], score=score))
    return rows


def interval_smooth(frame: pd.DataFrame, proba: np.ndarray, window: int, beta: float) -> np.ndarray:
    if window <= 1:
        return proba
    smooth = np.zeros_like(proba, dtype=float)
    for _, group in frame.reset_index(drop=True).groupby("WELL", sort=False):
        idx = group.index.to_numpy()
        block = pd.DataFrame(proba[idx])
        rolled = block.rolling(window=window, min_periods=1, center=True).mean().to_numpy()
        smooth[idx] = beta * proba[idx] + (1.0 - beta) * rolled
    return normalized(smooth)


def run_seed(seed: int, args) -> list[dict]:
    if torch is not None:
        torch.manual_seed(seed)
    np.random.seed(seed)
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, features = build_features(df, include_missing=True)
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    train_wells, interp_wells, extra_wells, _ = split_wells_by_space(df, args.train_fraction, args.interp_test_wells, seed)
    train_wells = set(train_wells)
    train = df[df["WELL"].isin(train_wells)].copy()
    n_classes = len(class_names)
    rows = []

    classes = np.array(sorted(train["TARGET"].unique()))
    class_to_local = {cls: idx for idx, cls in enumerate(classes)}
    y_train = train["TARGET"].map(class_to_local)
    fitted_models = {}
    for name in args.models:
        model = make_model(name, seed, len(classes), args)
        model.fit(train[features], y_train)
        fitted_models[name] = model
        for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
            test = df[df["WELL"].isin(wells)].copy()
            proba = model_proba(model, test[features], classes, n_classes)
            rows.extend(selective_rows(test["TARGET"].to_numpy(dtype=int), proba, seed, split, name, args.coverages))

    fitted = fit_family_views(df, train_wells, features, seed, n_classes, args)
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        test = df[df["WELL"].isin(wells)].copy()
        ordered, p_tree, p_seq = family_posteriors(fitted, test, n_classes, args)
        y = ordered["TARGET"].to_numpy(dtype=int)
        geoshift = normalized(args.tree_weight * p_tree + (1.0 - args.tree_weight) * p_seq)
        interval_geoshift = interval_smooth(ordered, geoshift, args.interval_window, args.interval_beta)
        rows.extend(selective_rows(y, p_tree, seed, split, "tree_geoshift_view", args.coverages))
        rows.extend(selective_rows(y, p_seq, seed, split, "sequence_geoshift_view", args.coverages))
        rows.extend(selective_rows(y, geoshift, seed, split, "geoshift_seq", args.coverages))
        rows.extend(selective_rows(y, interval_geoshift, seed, split, "interval_geoshift_seq", args.coverages))

        if "rf" in fitted_models:
            p_rf = model_proba(fitted_models["rf"], ordered[features], classes, n_classes)
            p_anchor = normalized(args.anchor_weight * geoshift + (1.0 - args.anchor_weight) * p_rf)
            p_interval_anchor = normalized(args.anchor_weight * interval_geoshift + (1.0 - args.anchor_weight) * p_rf)
            p_interval_anchor = interval_smooth(ordered, p_interval_anchor, args.interval_window, args.interval_beta)
            rows.extend(selective_rows(y, p_anchor, seed, split, "anchored_geoshift_seq", args.coverages))
            rows.extend(selective_rows(y, p_interval_anchor, seed, split, "interval_anchor_geoshift_seq", args.coverages))
            agree = (geoshift.argmax(axis=1) == p_rf.argmax(axis=1)).astype(float)
            score = 0.45 * margin(geoshift) + 0.35 * margin(p_rf) + 0.20 * agree
            rows.extend(selective_rows(y, geoshift, seed, split, "consensus_geoshift_seq", args.coverages, score=score))
            interval_agree = (interval_geoshift.argmax(axis=1) == p_rf.argmax(axis=1)).astype(float)
            interval_score = 0.45 * margin(interval_geoshift) + 0.35 * margin(p_rf) + 0.20 * interval_agree
            rows.extend(
                selective_rows(
                    y,
                    p_interval_anchor,
                    seed,
                    split,
                    "interval_consensus_geoshift_seq",
                    args.coverages,
                    score=interval_score,
                )
            )
            rows.extend(
                selective_rows(
                    y,
                    p_anchor,
                    seed,
                    split,
                    "agreement_gated_anchor_geoshift_seq",
                    args.coverages,
                    score=agreement_gate_score(geoshift, p_rf),
                )
            )
            rows.extend(
                selective_rows(
                    y,
                    geoshift,
                    seed,
                    split,
                    "triple_consensus_geoshift_seq",
                    args.coverages,
                    score=triple_agreement_score(p_tree, p_seq, p_rf, geoshift),
                )
            )

        anchor_posteriors = [
            model_proba(fitted_models[name], ordered[features], classes, n_classes)
            for name in args.anchor_models
            if name in fitted_models
        ]
        if anchor_posteriors:
            p_anchor_pool = posterior_pool(anchor_posteriors)
            if "rf" in fitted_models:
                rows.extend(
                    adaptive_reliability_rows(
                        y,
                        geoshift,
                        model_proba(fitted_models["rf"], ordered[features], classes, n_classes),
                        p_anchor_pool,
                        anchor_posteriors,
                        seed,
                        split,
                        args.coverages,
                        args,
                    )
                )
            for multi_weight in args.multi_anchor_weights:
                p_multi = normalized(float(multi_weight) * geoshift + (1.0 - float(multi_weight)) * p_anchor_pool)
                suffix = f"w{int(round(float(multi_weight) * 100)):03d}"
                rows.extend(selective_rows(y, p_multi, seed, split, f"multi_anchor_geoshift_seq_{suffix}", args.coverages))
                rows.extend(
                    selective_rows(
                        y,
                        p_multi,
                        seed,
                        split,
                        f"multi_anchor_consensus_geoshift_seq_{suffix}",
                        args.coverages,
                        score=multi_anchor_score(geoshift, p_anchor_pool, anchor_posteriors),
                    )
                )
                rows.extend(
                    selective_rows(
                        y,
                        p_multi,
                        seed,
                        split,
                        f"multi_anchor_conservative_geoshift_seq_{suffix}",
                        args.coverages,
                        score=multi_anchor_conservative_score(geoshift, p_anchor_pool, anchor_posteriors, p_multi),
                    )
                )
                if abs(float(multi_weight) - float(args.multi_anchor_weight)) < 1e-12:
                    rows.extend(selective_rows(y, p_multi, seed, split, "multi_anchor_geoshift_seq", args.coverages))
                    rows.extend(
                        selective_rows(
                            y,
                            p_multi,
                            seed,
                            split,
                            "multi_anchor_consensus_geoshift_seq",
                            args.coverages,
                            score=multi_anchor_score(geoshift, p_anchor_pool, anchor_posteriors),
                        )
                    )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted", "kept_rows"]
    summary = raw.groupby(["method", "split", "coverage"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/force2020"))
    parser.add_argument("--target", default="FORCE_2020_LITHOFACIES_LITHOLOGY")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42])
    parser.add_argument("--models", nargs="+", default=["svm", "knn", "mlp", "rf", "xgb", "lgbm", "cat"])
    parser.add_argument("--anchor-models", nargs="+", default=["mlp", "rf", "xgb", "lgbm", "cat"])
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--interp-test-wells", type=int, default=10)
    parser.add_argument("--max-rows-per-well", type=int, default=800)
    parser.add_argument("--tree-weight", type=float, default=0.75)
    parser.add_argument("--anchor-weight", type=float, default=0.70)
    parser.add_argument("--multi-anchor-weight", type=float, default=0.50)
    parser.add_argument("--multi-anchor-weights", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--high-coverage-multi-anchor-weight", type=float, default=0.70)
    parser.add_argument("--low-coverage-cutoff", type=float, default=0.02)
    parser.add_argument("--mid-coverage-cutoff", type=float, default=0.10)
    parser.add_argument("--interval-window", type=int, default=9)
    parser.add_argument("--interval-beta", type=float, default=0.65)
    parser.add_argument("--coverages", nargs="+", type=float, default=DEFAULT_COVERAGES)
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
    parser.add_argument("--out-csv", type=Path, default=Path("results/selective_multimethod_lithofacies_11seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/selective_multimethod_lithofacies_11seed_summary.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = []
    done = set()
    if args.resume and args.out_csv.exists():
        existing = pd.read_csv(args.out_csv)
        rows = existing.to_dict("records")
        done = set(existing["seed"].unique())
    for seed in args.seeds:
        if seed in done:
            continue
        rows.extend(run_seed(seed, args))
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    raw = pd.DataFrame(rows)
    summarize(raw).to_csv(args.summary_csv, index=False)
    print(summarize(raw).to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")


if __name__ == "__main__":
    main()
