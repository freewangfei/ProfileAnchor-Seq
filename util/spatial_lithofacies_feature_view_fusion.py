import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

from util.spatial_lithofacies_feature_ablation_smote import select_features
from data.spatial_multimethod_group_benchmark import (
    build_features,
    load_force,
    make_model,
    sample_by_well,
    split_wells_by_space,
)
from model.spatial_tree_smote_aligned_lithofacies import smote_augment


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "MCC": matthews_corrcoef(y_true, pred),
        "F1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
    }


def fit_view(train: pd.DataFrame, features: list[str], seed: int, args) -> dict:
    classes = np.array(sorted(train["TARGET"].unique()))
    class_to_local = {cls: idx for idx, cls in enumerate(classes)}
    y_train = train["TARGET"].map(class_to_local).to_numpy(dtype=int)
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_aug, y_aug = smote_augment(x_train, y_train, seed, args)
    model = make_model(args.model, seed, len(classes), args)
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    start = time.time()
    estimator.fit(x_aug, y_aug)
    return {
        "classes": classes,
        "imputer": imputer,
        "estimator": estimator,
        "features": features,
        "synthetic_rows": len(x_aug) - len(x_train),
        "Training Time": time.time() - start,
    }


def proba_view(fitted: dict, frame: pd.DataFrame, n_classes_total: int) -> np.ndarray:
    x = fitted["imputer"].transform(frame[fitted["features"]])
    local = fitted["estimator"].predict_proba(x)
    out = np.full((len(frame), n_classes_total), 1e-12, dtype=float)
    out[:, fitted["classes"]] = local
    out /= out.sum(axis=1, keepdims=True)
    return out


def y_band_fit_calibration(train: pd.DataFrame, cal_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    well_y = train.groupby("WELL")["Y_LOC"].mean().dropna().sort_values()
    n_cal = max(2, int(round(len(well_y) * cal_fraction)))
    n_cal = min(n_cal, max(1, len(well_y) - 2))
    cal_wells = set(well_y.head(n_cal).index)
    fit = train[~train["WELL"].isin(cal_wells)].copy()
    cal = train[train["WELL"].isin(cal_wells)].copy()
    if fit["TARGET"].nunique() < 2 or cal["TARGET"].nunique() < 2:
        raise ValueError("Degenerate inner fit/calibration split.")
    return fit, cal


def tune_alpha(fit: pd.DataFrame, cal: pd.DataFrame, all_features: list[str], seed: int, n_total: int, args) -> tuple[float, list[dict]]:
    base_features = select_features(all_features, "base")
    noxyz_features = select_features(all_features, "no_xyz")
    base = fit_view(fit, base_features, seed, args)
    noxyz = fit_view(fit, noxyz_features, seed, args)
    p_base = proba_view(base, cal, n_total)
    p_noxyz = proba_view(noxyz, cal, n_total)
    y = cal["TARGET"].to_numpy()
    trace = []
    best_alpha = 1.0
    best_score = -np.inf
    for alpha in args.alpha_grid:
        proba = alpha * p_base + (1.0 - alpha) * p_noxyz
        pred = proba.argmax(axis=1)
        row = metric_row(y, pred)
        score = row["F1_weighted"] + args.macro_weight * row["F1_macro"] + args.ba_weight * row["Balanced Accuracy"]
        item = {"seed": seed, "alpha": alpha, "selection_score": score, **row}
        trace.append(item)
        if score > best_score + 1e-12:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, trace


def run_seed(seed: int, args) -> tuple[list[dict], list[dict]]:
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, all_features = build_features(df, include_missing=True)
    train_wells, interp_wells, extra_wells, y_cut = split_wells_by_space(
        df, args.train_fraction, args.interp_test_wells, seed
    )
    train = df[df["WELL"].isin(train_wells)].copy()
    fit, cal = y_band_fit_calibration(train, args.cal_fraction)
    selected_alpha, trace = tune_alpha(fit, cal, all_features, seed, len(class_names), args)

    base = fit_view(train, select_features(all_features, "base"), seed, args)
    noxyz = fit_view(train, select_features(all_features, "no_xyz"), seed, args)
    rows = []
    variants = {
        "base_smote": 1.0,
        "noxyz_smote": 0.0,
        "fixed_view_fusion_a0.75": 0.75,
        "fixed_view_fusion_a0.90": 0.90,
        "inner_selected_view_fusion": selected_alpha,
    }
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        test = df[df["WELL"].isin(wells)].copy()
        y = test["TARGET"].to_numpy()
        p_base = proba_view(base, test, len(class_names))
        p_noxyz = proba_view(noxyz, test, len(class_names))
        for method, alpha in variants.items():
            proba = alpha * p_base + (1.0 - alpha) * p_noxyz
            pred = proba.argmax(axis=1)
            row = metric_row(y, pred)
            row.update(
                {
                    "seed": seed,
                    "target": args.target,
                    "model": args.model,
                    "method": method,
                    "split": split,
                    "feature_set": "base_plus_noxyz_view_fusion",
                    "alpha": alpha,
                    "selected_alpha": selected_alpha,
                    "n_classes_total": len(class_names),
                    "n_classes_train": train["TARGET"].nunique(),
                    "train_rows": len(train),
                    "fit_rows": len(fit),
                    "cal_rows": len(cal),
                    "test_rows": len(test),
                    "train_wells": len(train_wells),
                    "fit_wells": fit["WELL"].nunique(),
                    "cal_wells": cal["WELL"].nunique(),
                    "test_wells": len(wells),
                    "Training Time": base["Training Time"] + noxyz["Training Time"],
                    "synthetic_rows": base["synthetic_rows"] + noxyz["synthetic_rows"],
                    "y_cut": y_cut,
                }
            )
            rows.append(row)
    return rows, trace


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted", "Training Time", "synthetic_rows"]
    summary = raw.groupby(["target", "model", "method", "feature_set", "split"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def paired_stats(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted"]
    for split in ["interpolation", "extrapolation"]:
        subset = raw[raw["split"] == split]
        base = subset[subset["method"] == "base_smote"].set_index("seed")
        for method in sorted(m for m in subset["method"].unique() if m != "base_smote"):
            cand = subset[subset["method"] == method].set_index("seed").reindex(base.index)
            for metric in metric_cols:
                b = base[metric].dropna()
                c = cand[metric].reindex(b.index).dropna()
                b = b.reindex(c.index)
                diff = c.to_numpy() - b.to_numpy()
                p_t = np.nan
                p_w = np.nan
                if len(diff) > 1:
                    p_t = float(ttest_rel(c, b).pvalue)
                    try:
                        p_w = float(wilcoxon(diff).pvalue)
                    except ValueError:
                        p_w = np.nan
                rows.append(
                    {
                        "split": split,
                        "baseline": "base_smote",
                        "method": method,
                        "metric": metric,
                        "n": len(diff),
                        "baseline_mean": float(b.mean()),
                        "method_mean": float(c.mean()),
                        "delta_mean": float(diff.mean()),
                        "wins": int((diff > 0).sum()),
                        "ties": int((diff == 0).sum()),
                        "losses": int((diff < 0).sum()),
                        "paired_t_p": p_t,
                        "wilcoxon_p": p_w,
                        "deltas": ";".join(f"{d:.6f}" for d in diff),
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/force2020"))
    parser.add_argument("--target", default="FORCE_2020_LITHOFACIES_LITHOLOGY")
    parser.add_argument("--model", default="rf")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 42])
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--interp-test-wells", type=int, default=10)
    parser.add_argument("--max-rows-per-well", type=int, default=800)
    parser.add_argument("--cal-fraction", type=float, default=0.25)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.0, 0.5, 0.75, 0.9, 1.0])
    parser.add_argument("--macro-weight", type=float, default=0.20)
    parser.add_argument("--ba-weight", type=float, default=0.10)
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
    parser.add_argument("--out-csv", type=Path, default=Path("results/spatial_lithofacies_feature_view_fusion_5seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/spatial_lithofacies_feature_view_fusion_5seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/spatial_lithofacies_feature_view_fusion_5seed_paired_stats.csv"))
    parser.add_argument("--trace-csv", type=Path, default=Path("results/spatial_lithofacies_feature_view_fusion_5seed_trace.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.models = [args.model]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.out_csv.exists():
        raw = pd.read_csv(args.out_csv)
        trace = pd.read_csv(args.trace_csv) if args.trace_csv.exists() else pd.DataFrame()
        done = set(raw["seed"].unique())
    else:
        raw = pd.DataFrame()
        trace = pd.DataFrame()
        done = set()
    for seed in args.seeds:
        if seed in done:
            continue
        rows, trace_rows = run_seed(seed, args)
        raw = pd.concat([raw, pd.DataFrame(rows)], ignore_index=True)
        trace = pd.concat([trace, pd.DataFrame(trace_rows)], ignore_index=True)
        raw.to_csv(args.out_csv, index=False)
        trace.to_csv(args.trace_csv, index=False)
    summary = summarize(raw)
    paired = paired_stats(raw)
    summary.to_csv(args.summary_csv, index=False)
    paired.to_csv(args.paired_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.paired_csv}")
    print(f"Wrote {args.trace_csv}")


if __name__ == "__main__":
    main()
