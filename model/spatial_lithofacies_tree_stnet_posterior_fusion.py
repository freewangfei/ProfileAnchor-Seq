import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel, wilcoxon

from util.spatial_lithofacies_feature_ablation_smote import select_features
from util.spatial_lithofacies_feature_view_fusion import (
    fit_view as fit_tree_view,
    metric_row,
    proba_view as proba_tree_view,
    y_band_fit_calibration,
)
from model.spatial_lithofacies_stnet_view_fusion import (
    fit_view as fit_stnet_view,
    proba_view as proba_stnet_view,
)
from data.spatial_multimethod_group_benchmark import build_features, load_force, sample_by_well, split_wells_by_space


def ordered_proba_tree(fitted: dict, frame: pd.DataFrame, n_classes: int) -> np.ndarray:
    ordered = frame.sort_values(["WELL", "DEPTH_MD"])
    return proba_tree_view(fitted, ordered, n_classes), ordered


def ordered_proba_stnet(fitted: dict, frame: pd.DataFrame, args) -> tuple[np.ndarray, pd.DataFrame]:
    ordered = frame.sort_values(["WELL", "DEPTH_MD"])
    proba = proba_stnet_view(fitted, ordered, args).loc[ordered.index.to_numpy()].to_numpy()
    return proba, ordered


def normalized(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def fit_family_views(df: pd.DataFrame, train_wells: set[str], all_features: list[str], seed: int, n_classes: int, args) -> dict:
    train = df[df["WELL"].isin(train_wells)].copy()
    base_features = select_features(all_features, "base")
    noxyz_features = select_features(all_features, "no_xyz")
    args.model = args.tree_model
    tree_base = fit_tree_view(train, base_features, seed, args)
    tree_noxyz = fit_tree_view(train, noxyz_features, seed, args)
    stnet_base = fit_stnet_view(df, train_wells, base_features, seed, n_classes, args)
    stnet_noxyz = fit_stnet_view(df, train_wells, noxyz_features, seed, n_classes, args)
    return {
        "tree_base": tree_base,
        "tree_noxyz": tree_noxyz,
        "stnet_base": stnet_base,
        "stnet_noxyz": stnet_noxyz,
    }


def family_posteriors(fitted: dict, frame: pd.DataFrame, n_classes: int, args) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    p_tree_base, ordered = ordered_proba_tree(fitted["tree_base"], frame, n_classes)
    p_tree_noxyz, _ = ordered_proba_tree(fitted["tree_noxyz"], frame, n_classes)
    p_stnet_base, ordered_stnet = ordered_proba_stnet(fitted["stnet_base"], frame, args)
    p_stnet_noxyz, _ = ordered_proba_stnet(fitted["stnet_noxyz"], frame, args)
    if not np.array_equal(ordered.index.to_numpy(), ordered_stnet.index.to_numpy()):
        raise RuntimeError("Tree and STNet posterior row orders differ.")
    p_tree = normalized(args.view_alpha * p_tree_base + (1.0 - args.view_alpha) * p_tree_noxyz)
    p_stnet = normalized(args.view_alpha * p_stnet_base + (1.0 - args.view_alpha) * p_stnet_noxyz)
    return ordered, p_tree, p_stnet


def tune_tree_weight(df: pd.DataFrame, train_wells: set[str], all_features: list[str], seed: int, n_classes: int, args):
    train = df[df["WELL"].isin(train_wells)].copy()
    fit, cal = y_band_fit_calibration(train, args.cal_fraction)
    fitted = fit_family_views(df, set(fit["WELL"].unique()), all_features, seed, n_classes, args)
    ordered, p_tree, p_stnet = family_posteriors(fitted, cal, n_classes, args)
    y = ordered["TARGET"].to_numpy(dtype=np.int64)
    best_weight = 1.0
    best_score = -np.inf
    trace = []
    for w in args.tree_weight_grid:
        p = normalized(float(w) * p_tree + (1.0 - float(w)) * p_stnet)
        row = metric_row(y, p.argmax(axis=1))
        score = row["F1_weighted"] + args.macro_weight * row["F1_macro"] + args.ba_weight * row["Balanced Accuracy"]
        item = {"seed": seed, "tree_weight": float(w), "selection_score": score, **row}
        trace.append(item)
        if score > best_score + 1e-12:
            best_score = score
            best_weight = float(w)
    return best_weight, trace


def run_seed(seed: int, args) -> tuple[list[dict], list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, all_features = build_features(df, include_missing=True)
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    train_wells, interp_wells, extra_wells, y_cut = split_wells_by_space(df, args.train_fraction, args.interp_test_wells, seed)
    train_wells = set(train_wells)
    n_classes = len(class_names)
    if args.skip_inner:
        selected_weight = float(args.default_tree_weight)
        trace = []
    else:
        selected_weight, trace = tune_tree_weight(df, train_wells, all_features, seed, n_classes, args)
    fitted = fit_family_views(df, train_wells, all_features, seed, n_classes, args)
    variants = {
        "tree_geoshift_view": 1.0,
        "stnet_geoshift_view": 0.0,
        "tree_stnet_inner_selected_fusion": selected_weight,
    }
    for w in args.report_tree_weights:
        variants[f"tree_stnet_fixed_w{int(round(w * 100)):03d}"] = float(w)

    rows = []
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        frame = df[df["WELL"].isin(wells)].copy()
        ordered, p_tree, p_stnet = family_posteriors(fitted, frame, n_classes, args)
        y = ordered["TARGET"].to_numpy(dtype=np.int64)
        for method, w in variants.items():
            p = normalized(float(w) * p_tree + (1.0 - float(w)) * p_stnet)
            row = metric_row(y, p.argmax(axis=1))
            row.update(
                {
                    "seed": seed,
                    "target": args.target,
                    "method": method,
                    "split": split,
                    "tree_weight": float(w),
                    "selected_tree_weight": float(selected_weight),
                    "view_alpha": args.view_alpha,
                    "tree_model": args.tree_model,
                    "window": args.window,
                    "epochs": args.epochs,
                    "hidden": args.hidden,
                    "train_rows": int(df["WELL"].isin(train_wells).sum()),
                    "test_rows": len(ordered),
                    "train_wells": len(train_wells),
                    "test_wells": len(wells),
                    "y_cut": y_cut,
                }
            )
            rows.append(row)
    return rows, trace


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted"]
    summary = raw.groupby(["target", "method", "split"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def paired_stats(raw: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows = []
    for split in ["interpolation", "extrapolation"]:
        subset = raw[raw["split"] == split]
        base = subset[subset["method"] == baseline].set_index("seed")
        for method in sorted(m for m in subset["method"].unique() if m != baseline):
            cand = subset[subset["method"] == method].set_index("seed").reindex(base.index)
            for metric in ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted"]:
                b = base[metric].dropna()
                c = cand[metric].reindex(b.index).dropna()
                b = b.reindex(c.index)
                diff = c.to_numpy() - b.to_numpy()
                rows.append(
                    {
                        "split": split,
                        "baseline": baseline,
                        "method": method,
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
    parser.add_argument("--cal-fraction", type=float, default=0.25)
    parser.add_argument("--view-alpha", type=float, default=0.75)
    parser.add_argument("--tree-weight-grid", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    parser.add_argument("--report-tree-weights", nargs="+", type=float, default=[0.75, 0.9])
    parser.add_argument("--skip-inner", action="store_true")
    parser.add_argument("--default-tree-weight", type=float, default=0.75)
    parser.add_argument("--macro-weight", type=float, default=0.20)
    parser.add_argument("--ba-weight", type=float, default=0.10)
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
    parser.add_argument("--out-csv", type=Path, default=Path("results/spatial_lithofacies_tree_stnet_fusion_5seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/spatial_lithofacies_tree_stnet_fusion_5seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/spatial_lithofacies_tree_stnet_fusion_5seed_paired.csv"))
    parser.add_argument("--trace-csv", type=Path, default=Path("results/spatial_lithofacies_tree_stnet_fusion_5seed_trace.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = []
    trace_rows = []
    done = set()
    if args.resume and args.out_csv.exists():
        existing = pd.read_csv(args.out_csv)
        rows = existing.to_dict("records")
        done = set(existing["seed"].unique())
        if args.trace_csv.exists():
            trace_rows = pd.read_csv(args.trace_csv).to_dict("records")
    for seed in args.seeds:
        if seed in done:
            continue
        seed_rows, seed_trace = run_seed(seed, args)
        rows.extend(seed_rows)
        trace_rows.extend(seed_trace)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        pd.DataFrame(trace_rows).to_csv(args.trace_csv, index=False)
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    paired = paired_stats(raw, baseline="tree_geoshift_view")
    summary.to_csv(args.summary_csv, index=False)
    paired.to_csv(args.paired_csv, index=False)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.paired_csv}")
    print(f"Wrote {args.trace_csv}")


if __name__ == "__main__":
    main()
