import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from data.spatial_multimethod_group_benchmark import (
    build_features,
    evaluate_model,
    load_force,
    sample_by_well,
    split_wells_by_space,
)
from model.spatial_tree_smote_aligned_lithofacies import evaluate_smote_model, summarize


def select_features(features: list[str], mode: str) -> list[str]:
    horizontal = {"X_LOC", "Y_LOC", "X_LOC_MISSING", "Y_LOC_MISSING"}
    spatial = horizontal | {"Z_LOC", "Z_LOC_MISSING"}
    depth = {"DEPTH_MD", "DEPTH_MD_MISSING"}
    logs = {"GR", "RHOB", "NPHI", "DTC"}
    wellz = {f"{name}_WELL_Z" for name in ["GR", "RHOB", "NPHI", "DTC"]}
    log_missing = {f"{name}_MISSING" for name in logs | wellz}
    log_family = logs | wellz | log_missing
    if mode == "base":
        return features
    if mode == "no_xy":
        return [f for f in features if f not in horizontal]
    if mode == "no_xyz":
        return [f for f in features if f not in spatial]
    if mode == "logs_only":
        return [f for f in features if f in log_family]
    if mode == "logs_depth":
        return [f for f in features if f in (log_family | depth)]
    raise ValueError(f"Unknown feature mode: {mode}")


def run_seed(seed: int, args) -> list[dict]:
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, all_features = build_features(df, include_missing=True)
    train_wells, interp_wells, extra_wells, y_cut = split_wells_by_space(
        df, args.train_fraction, args.interp_test_wells, seed
    )
    train = df[df["WELL"].isin(train_wells)].copy()
    rows = []
    for mode in args.feature_modes:
        features = select_features(all_features, mode)
        for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
            test = df[df["WELL"].isin(wells)].copy()
            none_row = evaluate_model(train, test, features, args.model, seed, args)
            smote_row = evaluate_smote_model(train, test, features, args.model, seed, args)
            for augmentation, row in [("none", none_row), ("smote", smote_row)]:
                row.update(
                    {
                        "seed": seed,
                        "target": args.target,
                        "model": args.model,
                        "augmentation": augmentation,
                        "split": split,
                        "feature_set": mode,
                        "n_classes_total": len(class_names),
                        "n_classes_train": train["TARGET"].nunique(),
                        "n_features": len(features),
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "train_wells": len(train_wells),
                        "test_wells": len(wells),
                        "y_cut": y_cut,
                    }
                )
                row.setdefault("synthetic_rows", 0)
                rows.append(row)
    return rows


def paired_against_base_smote(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted"]
    for split in ["interpolation", "extrapolation"]:
        base = raw[(raw["split"] == split) & (raw["feature_set"] == "base") & (raw["augmentation"] == "smote")].set_index("seed")
        for mode in sorted(raw["feature_set"].unique()):
            for augmentation in sorted(raw["augmentation"].unique()):
                if mode == "base" and augmentation == "smote":
                    continue
                cand = raw[(raw["split"] == split) & (raw["feature_set"] == mode) & (raw["augmentation"] == augmentation)].set_index("seed")
                cand = cand.reindex(base.index)
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
                            "method": f"{mode}_{augmentation}",
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
    parser.add_argument("--feature-modes", nargs="+", default=["base", "no_xy", "no_xyz", "logs_depth", "logs_only"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 42])
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--interp-test-wells", type=int, default=10)
    parser.add_argument("--max-rows-per-well", type=int, default=800)
    parser.add_argument("--target-quantile", type=float, default=0.75)
    parser.add_argument("--max-augmented-multiplier", type=float, default=1.5)
    parser.add_argument("--smote-k", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--gbdt-max-iter", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-max-iter", type=int, default=4000)
    parser.add_argument("--knn-neighbors", type=int, default=15)
    parser.add_argument("--mlp-max-iter", type=int, default=160)
    parser.add_argument("--out-csv", type=Path, default=Path("results/spatial_lithofacies_feature_ablation_smote_5seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/spatial_lithofacies_feature_ablation_smote_5seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/spatial_lithofacies_feature_ablation_smote_5seed_paired_stats.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.models = [args.model]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.out_csv.exists():
        raw = pd.read_csv(args.out_csv)
        done = set(raw["seed"].unique())
    else:
        raw = pd.DataFrame()
        done = set()
    for seed in args.seeds:
        if seed in done:
            continue
        raw = pd.concat([raw, pd.DataFrame(run_seed(seed, args))], ignore_index=True)
        raw.to_csv(args.out_csv, index=False)

    summary = summarize(raw)
    paired = paired_against_base_smote(raw)
    summary.to_csv(args.summary_csv, index=False)
    paired.to_csv(args.paired_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.paired_csv}")


if __name__ == "__main__":
    main()
