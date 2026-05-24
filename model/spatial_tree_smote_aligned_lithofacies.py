import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.neighbors import NearestNeighbors

from data.spatial_multimethod_group_benchmark import (
    build_features,
    evaluate_model,
    load_force,
    make_model,
    sample_by_well,
    split_wells_by_space,
)


def smote_plan(y: np.ndarray, target_quantile: float, min_class_samples: int, max_multiplier: float) -> dict[int, int]:
    labels, counts = np.unique(y, return_counts=True)
    target = int(np.ceil(np.quantile(counts, target_quantile)))
    raw_needs = {
        int(label): max(0, target - int(count))
        for label, count in zip(labels, counts)
        if count >= min_class_samples and count < target
    }
    max_new = max(0, int(len(y) * max_multiplier) - len(y))
    total_new = sum(raw_needs.values())
    if total_new == 0 or max_new == 0:
        return {}
    if total_new <= max_new:
        return raw_needs
    scale = max_new / total_new
    return {label: int(np.floor(need * scale)) for label, need in raw_needs.items() if int(np.floor(need * scale)) > 0}


def smote_augment(x: np.ndarray, y: np.ndarray, seed: int, args) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    plan = smote_plan(y, args.target_quantile, args.smote_k + 1, args.max_augmented_multiplier)
    if not plan:
        return x, y
    xs = [x]
    ys = [y]
    for label, n_new in plan.items():
        idx = np.flatnonzero(y == label)
        if n_new <= 0 or len(idx) <= args.smote_k:
            continue
        x_cls = x[idx]
        nn = NearestNeighbors(n_neighbors=args.smote_k + 1).fit(x_cls)
        neigh = nn.kneighbors(x_cls, return_distance=False)[:, 1:]
        anchors = rng.integers(0, len(idx), size=n_new)
        neigh_pos = rng.integers(0, args.smote_k, size=n_new)
        x0 = x_cls[anchors]
        x1 = x_cls[neigh[anchors, neigh_pos]]
        lam = rng.random((n_new, 1))
        xs.append(x0 + lam * (x1 - x0))
        ys.append(np.full(n_new, label, dtype=y.dtype))
    return np.vstack(xs), np.concatenate(ys)


def evaluate_smote_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str], model_name: str, seed: int, args):
    classes = np.array(sorted(train["TARGET"].unique()))
    class_to_local = {cls: idx for idx, cls in enumerate(classes)}
    y_train = train["TARGET"].map(class_to_local).to_numpy(dtype=int)
    y_test = test["TARGET"].to_numpy()

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_test = imputer.transform(test[features])
    x_aug, y_aug = smote_augment(x_train, y_train, seed, args)

    model = make_model(model_name, seed, len(classes), args)
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    else:
        estimator = model

    start = time.time()
    estimator.fit(x_aug, y_aug)
    train_time = time.time() - start
    pred_local = estimator.predict(x_test).astype(int).reshape(-1)
    pred = classes[pred_local]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
        "F1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        "Training Time": train_time,
        "synthetic_rows": len(x_aug) - len(x_train),
    }


def run_seed(seed: int, args) -> list[dict]:
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, features = build_features(df, include_missing=True)
    train_wells, interp_wells, extra_wells, y_cut = split_wells_by_space(
        df, args.train_fraction, args.interp_test_wells, seed
    )
    train = df[df["WELL"].isin(train_wells)].copy()
    rows = []
    for model_name in args.models:
        for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
            test = df[df["WELL"].isin(wells)].copy()
            none_row = evaluate_model(train, test, features, model_name, seed, args)
            smote_row = evaluate_smote_model(train, test, features, model_name, seed, args)
            for strategy, row in [("none", none_row), ("smote", smote_row)]:
                row.update(
                    {
                        "seed": seed,
                        "target": args.target,
                        "model": model_name,
                        "augmentation": strategy,
                        "split": split,
                        "feature_set": "base_well_z_missing_aligned",
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


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted", "Training Time", "synthetic_rows"]
    summary = raw.groupby(["target", "model", "augmentation", "feature_set", "split"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def paired_stats(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted"]
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except Exception:
        ttest_rel = wilcoxon = None
    for model in sorted(raw["model"].unique()):
        for split in ["interpolation", "extrapolation"]:
            subset = raw[(raw["model"] == model) & (raw["split"] == split)]
            pivot = subset.pivot(index="seed", columns="augmentation", values=metrics)
            for metric in metrics:
                if ("none" not in pivot[metric].columns) or ("smote" not in pivot[metric].columns):
                    continue
                none = pivot[metric]["none"].dropna()
                smote = pivot[metric]["smote"].reindex(none.index).dropna()
                none = none.reindex(smote.index)
                deltas = smote.to_numpy() - none.to_numpy()
                p_t = np.nan
                p_w = np.nan
                if ttest_rel is not None and len(deltas) > 1:
                    p_t = float(ttest_rel(smote, none).pvalue)
                    try:
                        p_w = float(wilcoxon(deltas).pvalue)
                    except ValueError:
                        p_w = np.nan
                rows.append(
                    {
                        "model": model,
                        "split": split,
                        "metric": metric,
                        "n": len(deltas),
                        "none_mean": float(none.mean()),
                        "smote_mean": float(smote.mean()),
                        "delta_mean": float(deltas.mean()),
                        "wins": int((deltas > 0).sum()),
                        "ties": int((deltas == 0).sum()),
                        "losses": int((deltas < 0).sum()),
                        "paired_t_p": p_t,
                        "wilcoxon_p": p_w,
                        "deltas": ";".join(f"{d:.6f}" for d in deltas),
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/force2020"))
    parser.add_argument("--target", default="FORCE_2020_LITHOFACIES_LITHOLOGY")
    parser.add_argument("--models", nargs="+", default=["rf", "gbdt", "xgb", "lgbm", "cat"])
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
    parser.add_argument("--out-csv", type=Path, default=Path("results/spatial_tree_smote_aligned_lithofacies_5seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/spatial_tree_smote_aligned_lithofacies_5seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/spatial_tree_smote_aligned_lithofacies_5seed_paired_stats.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.out_csv.exists():
        raw = pd.read_csv(args.out_csv)
        done = set(zip(raw["seed"], raw["model"]))
    else:
        raw = pd.DataFrame()
        done = set()

    for seed in args.seeds:
        seed_rows = []
        for model in args.models:
            if (seed, model) in done:
                continue
            sub_args = argparse.Namespace(**vars(args))
            sub_args.models = [model]
            seed_rows.extend(run_seed(seed, sub_args))
            updated = pd.concat([raw, pd.DataFrame(seed_rows)], ignore_index=True)
            updated.to_csv(args.out_csv, index=False)
        if seed_rows:
            raw = pd.concat([raw, pd.DataFrame(seed_rows)], ignore_index=True)

    summary = summarize(raw)
    paired = paired_stats(raw)
    summary.to_csv(args.summary_csv, index=False)
    paired.to_csv(args.paired_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.paired_csv}")


if __name__ == "__main__":
    main()
