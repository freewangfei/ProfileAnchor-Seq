import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

from util.spatial_lithofacies_feature_view_fusion import y_band_fit_calibration
from model.spatial_lithofacies_tree_stnet_posterior_fusion import family_posteriors, fit_family_views, normalized
from data.spatial_multimethod_group_benchmark import build_features, load_force, sample_by_well, split_wells_by_space

try:
    import torch
except Exception:
    torch = None


COVERAGES = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05]


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "MCC": matthews_corrcoef(y_true, pred),
        "F1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
    }


def entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])


def margin(proba: np.ndarray) -> np.ndarray:
    order = np.sort(proba, axis=1)
    return order[:, -1] - order[:, -2]


def nearest_train_distance(frame: pd.DataFrame, train_wells: set[str]) -> np.ndarray:
    well_xy = frame.groupby("WELL")[["X_LOC", "Y_LOC"]].mean().dropna()
    train_xy = well_xy.loc[[w for w in train_wells if w in well_xy.index]].to_numpy(dtype=float)
    if len(train_xy) == 0:
        return np.zeros(len(frame), dtype=float)
    dist_by_well = {}
    for well, xy in well_xy.iterrows():
        d = np.sqrt(((train_xy - xy.to_numpy(dtype=float)) ** 2).sum(axis=1))
        dist_by_well[well] = float(d.min())
    return frame["WELL"].map(dist_by_well).fillna(0.0).to_numpy(dtype=float)


def minmax(x: np.ndarray) -> np.ndarray:
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def reliability_scores(p_tree: np.ndarray, p_seq: np.ndarray, p_fused: np.ndarray, frame: pd.DataFrame, train_wells: set[str]) -> dict[str, np.ndarray]:
    pred_tree = p_tree.argmax(axis=1)
    pred_seq = p_seq.argmax(axis=1)
    pred_fused = p_fused.argmax(axis=1)
    conf = p_fused.max(axis=1)
    marg = margin(p_fused)
    ent = entropy(p_fused)
    agree = ((pred_tree == pred_seq) & (pred_tree == pred_fused)).astype(float)
    disagreement = 1.0 - np.abs(p_tree[np.arange(len(p_tree)), pred_fused] - p_seq[np.arange(len(p_seq)), pred_fused])
    dist = minmax(nearest_train_distance(frame, train_wells))
    return {
        "confidence": conf,
        "margin": marg,
        "confidence_margin": 0.5 * conf + 0.5 * marg,
        "agreement_margin": 0.45 * conf + 0.35 * marg + 0.20 * agree,
        "spatial_reliability": 0.45 * conf + 0.35 * marg + 0.15 * agree + 0.05 * disagreement - 0.15 * dist,
        "low_entropy": -ent,
    }


def selective_rows(y: np.ndarray, pred: np.ndarray, scores: dict[str, np.ndarray], split: str, seed: int, method_name: str) -> list[dict]:
    rows = []
    full = metric_row(y, pred)
    full_acc = full["Accuracy"]
    for score_name, score in scores.items():
        order = np.argsort(score)[::-1]
        for coverage in COVERAGES:
            keep_n = max(1, int(round(len(y) * coverage)))
            keep = order[:keep_n]
            row = metric_row(y[keep], pred[keep])
            row.update(
                {
                    "seed": seed,
                    "split": split,
                    "method": method_name,
                    "score": score_name,
                    "coverage": coverage,
                    "kept_rows": keep_n,
                    "full_accuracy": full_acc,
                    "accuracy_gain": row["Accuracy"] - full_acc,
                    "score_threshold": float(score[keep].min()),
                }
            )
            rows.append(row)
    return rows


def run_seed(seed: int, args) -> list[dict]:
    if torch is not None:
        torch.manual_seed(seed)
    np.random.seed(seed)
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, all_features = build_features(df, include_missing=True)
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    train_wells, interp_wells, extra_wells, y_cut = split_wells_by_space(df, args.train_fraction, args.interp_test_wells, seed)
    train_wells = set(train_wells)
    n_classes = len(class_names)
    fitted = fit_family_views(df, train_wells, all_features, seed, n_classes, args)
    rows = []
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        frame = df[df["WELL"].isin(wells)].copy()
        ordered, p_tree, p_seq = family_posteriors(fitted, frame, n_classes, args)
        p_fused = normalized(args.tree_weight * p_tree + (1.0 - args.tree_weight) * p_seq)
        y = ordered["TARGET"].to_numpy(dtype=int)
        pred = p_fused.argmax(axis=1)
        scores = reliability_scores(p_tree, p_seq, p_fused, ordered, train_wells)
        seed_rows = selective_rows(y, pred, scores, split, seed, f"geoshift_seq_w{int(args.tree_weight * 100):03d}")
        for row in seed_rows:
            row.update(
                {
                    "target": args.target,
                    "train_rows": int(df["WELL"].isin(train_wells).sum()),
                    "test_rows": len(ordered),
                    "train_wells": len(train_wells),
                    "test_wells": len(wells),
                    "y_cut": y_cut,
                }
            )
        rows.extend(seed_rows)
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted", "accuracy_gain", "kept_rows"]
    summary = raw.groupby(["target", "method", "split", "score", "coverage"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def paired_stats(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ["interpolation", "extrapolation"]:
        subset = raw[raw["split"] == split]
        base = subset[(subset["score"] == "confidence") & (subset["coverage"] == 1.0)].set_index("seed")
        for score in sorted(subset["score"].unique()):
            for coverage in sorted(subset["coverage"].unique()):
                cand = subset[(subset["score"] == score) & (subset["coverage"] == coverage)].set_index("seed").reindex(base.index)
                for metric in ["Accuracy", "F1_weighted"]:
                    b = base[metric].dropna()
                    c = cand[metric].reindex(b.index).dropna()
                    b = b.reindex(c.index)
                    diff = c.to_numpy() - b.to_numpy()
                    rows.append(
                        {
                            "split": split,
                            "score": score,
                            "coverage": coverage,
                            "metric": metric,
                            "n": len(diff),
                            "baseline_mean": float(b.mean()) if len(diff) else np.nan,
                            "method_mean": float(c.mean()) if len(diff) else np.nan,
                            "delta_mean": float(diff.mean()) if len(diff) else np.nan,
                            "wins": int((diff > 0).sum()),
                            "ties": int((diff == 0).sum()),
                            "losses": int((diff < 0).sum()),
                            "paired_t_p": float(ttest_rel(c, b).pvalue) if len(diff) > 1 else np.nan,
                            "wilcoxon_p": float(wilcoxon(diff).pvalue) if len(diff) > 1 and np.any(diff != 0) else np.nan,
                            "deltas": ";".join(f"{d:.6f}" for d in diff),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/force2020"))
    parser.add_argument("--target", default="FORCE_2020_LITHOFACIES_LITHOLOGY")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 42])
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--interp-test-wells", type=int, default=10)
    parser.add_argument("--max-rows-per-well", type=int, default=800)
    parser.add_argument("--tree-weight", type=float, default=0.75)
    parser.add_argument("--view-alpha", type=float, default=0.75)
    parser.add_argument("--tree-model", default="rf")
    parser.add_argument("--model", default="rf")
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
    parser.add_argument("--out-csv", type=Path, default=Path("results/spatial_lithofacies_selective_geoshift_seq_5seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/spatial_lithofacies_selective_geoshift_seq_5seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/spatial_lithofacies_selective_geoshift_seq_5seed_paired.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = []
    done = set()
    if args.resume and args.out_csv.exists():
        raw = pd.read_csv(args.out_csv)
        rows = raw.to_dict("records")
        done = set(raw["seed"].unique())
    for seed in args.seeds:
        if seed in done:
            continue
        rows.extend(run_seed(seed, args))
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    paired = paired_stats(raw)
    summary.to_csv(args.summary_csv, index=False)
    paired.to_csv(args.paired_csv, index=False)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.paired_csv}")


if __name__ == "__main__":
    main()
