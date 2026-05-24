import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


def _optional_classifier(module_name: str, class_name: str):
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception as exc:
        raise ImportError(
            f"{class_name} requires optional dependency {module_name!r}, "
            "which is not importable in this environment."
        ) from exc


BASE_FEATURES = ["DEPTH_MD", "X_LOC", "Y_LOC", "Z_LOC", "GR", "RHOB", "NPHI", "DTC"]
WELL_Z_FEATURES = [f"{name}_WELL_Z" for name in ["GR", "RHOB", "NPHI", "DTC"]]


def load_force(data_dir: Path, target: str) -> tuple[pd.DataFrame, list[str]]:
    try:
        df = pd.read_csv(data_dir / "train.csv", sep=";")
    except (UnicodeDecodeError, pd.errors.ParserError):
        df = pd.read_csv(data_dir / "train.csv", sep=";", encoding="latin1", on_bad_lines="skip")
    df = df[df[target].notna()].copy()
    encoder = LabelEncoder()
    df["TARGET"] = encoder.fit_transform(df[target].astype(str))
    return df, list(encoder.classes_)


def sample_by_well(df: pd.DataFrame, max_rows_per_well: int | None, seed: int) -> pd.DataFrame:
    if max_rows_per_well is None:
        return df
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in df.groupby("WELL", sort=False):
        if len(group) <= max_rows_per_well:
            parts.append(group)
        else:
            idx = rng.choice(group.index.to_numpy(), size=max_rows_per_well, replace=False)
            parts.append(group.loc[idx].sort_values("DEPTH_MD"))
    return pd.concat(parts, axis=0, ignore_index=True)


def split_wells_by_space(df: pd.DataFrame, train_fraction: float, interp_test_wells: int, seed: int):
    rng = np.random.default_rng(seed)
    well_xy = df.groupby("WELL")[["X_LOC", "Y_LOC"]].mean().dropna()
    y_cut = well_xy["Y_LOC"].quantile(1.0 - train_fraction)
    north_wells = well_xy[well_xy["Y_LOC"] >= y_cut].index.to_numpy()
    south_wells = well_xy[well_xy["Y_LOC"] < y_cut].index.to_numpy()
    if len(north_wells) <= interp_test_wells:
        raise ValueError("Not enough north wells for interpolation split.")
    interp_test = rng.choice(north_wells, size=interp_test_wells, replace=False)
    train = np.array([well for well in north_wells if well not in set(interp_test)])
    return train, interp_test, south_wells, y_cut


def add_well_zscore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("WELL", sort=False)
    for feature in ["GR", "RHOB", "NPHI", "DTC"]:
        mean = grouped[feature].transform("mean")
        std = grouped[feature].transform("std").replace(0, np.nan)
        out[f"{feature}_WELL_Z"] = (out[feature] - mean) / std
    return out


def build_features(df: pd.DataFrame, include_missing: bool) -> tuple[pd.DataFrame, list[str]]:
    out = add_well_zscore(df)
    features = [col for col in BASE_FEATURES + WELL_Z_FEATURES if col in out.columns]
    if include_missing:
        for feature in list(features):
            col = f"{feature}_MISSING"
            out[col] = out[feature].isna().astype(np.float32)
            features.append(col)
    return out, features


def make_model(name: str, seed: int, n_classes: int, args):
    if name == "svm":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearSVC(
                C=args.svm_c,
                class_weight="balanced",
                dual="auto",
                max_iter=args.svm_max_iter,
                random_state=seed,
            ),
        )
    if name == "nb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            GaussianNB(),
        )
    if name == "knn":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=args.knn_neighbors,
                weights="distance",
                n_jobs=args.n_jobs,
            ),
        )
    if name == "mlp":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=args.mlp_max_iter,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=seed,
            ),
        )
    if name == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=args.n_estimators,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=args.n_jobs,
                random_state=seed,
            ),
        )
    if name == "gbdt":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=args.learning_rate,
                max_iter=args.gbdt_max_iter,
                max_leaf_nodes=31,
                max_depth=args.max_depth,
                min_samples_leaf=20,
                l2_regularization=1e-3,
                early_stopping=not getattr(args, "disable_gbdt_early_stopping", False),
                class_weight="balanced",
                random_state=seed,
            ),
        )
    if name == "xgb":
        XGBClassifier = _optional_classifier("xgboost", "XGBClassifier")
        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                subsample=0.9,
                colsample_bytree=0.85,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=seed,
                n_jobs=args.n_jobs,
            ),
        )
    if name == "lgbm":
        LGBMClassifier = _optional_classifier("lightgbm", "LGBMClassifier")
        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                num_leaves=min(31, 2 ** args.max_depth),
                subsample=0.9,
                colsample_bytree=0.85,
                objective="multiclass",
                num_class=n_classes,
                random_state=seed,
                n_jobs=args.n_jobs,
                verbose=-1,
            ),
        )
    if name == "cat":
        CatBoostClassifier = _optional_classifier("catboost", "CatBoostClassifier")
        return make_pipeline(
            SimpleImputer(strategy="median"),
            CatBoostClassifier(
                iterations=args.n_estimators,
                depth=args.max_depth,
                learning_rate=args.learning_rate,
                loss_function="MultiClass",
                random_seed=seed,
                thread_count=args.n_jobs,
                verbose=False,
                allow_writing_files=False,
            ),
        )
    raise ValueError(f"Unknown model: {name}")


def evaluate_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str], model_name: str, seed: int, args):
    classes = np.array(sorted(train["TARGET"].unique()))
    class_to_local = {cls: idx for idx, cls in enumerate(classes)}
    y_train = train["TARGET"].map(class_to_local)
    model = make_model(model_name, seed, len(classes), args)
    start = time.time()
    model.fit(train[features], y_train)
    train_time = time.time() - start
    pred_local = model.predict(test[features]).astype(int).reshape(-1)
    pred = classes[pred_local]
    y = test["TARGET"].to_numpy()
    return {
        "Accuracy": accuracy_score(y, pred),
        "Balanced Accuracy": balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
        "F1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        "Training Time": train_time,
    }


def run_seed(seed: int, args) -> list[dict]:
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, features = build_features(df, include_missing=not args.no_missing_indicators)
    train_wells, interp_wells, extra_wells, y_cut = split_wells_by_space(
        df, args.train_fraction, args.interp_test_wells, seed
    )
    train = df[df["WELL"].isin(train_wells)].copy()
    rows = []
    for model_name in args.models:
        for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
            test = df[df["WELL"].isin(wells)].copy()
            row = evaluate_model(train, test, features, model_name, seed, args)
            row.update(
                {
                    "seed": seed,
                    "model": model_name,
                    "split": split,
                    "target": args.target,
                    "feature_set": "base_well_z_missing",
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
            rows.append(row)
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted", "Training Time"]
    summary = raw.groupby(["target", "model", "feature_set", "split"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/force2020"))
    parser.add_argument("--target", default="GROUP", choices=["GROUP", "FORMATION", "FORCE_2020_LITHOFACIES_LITHOLOGY"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 42])
    parser.add_argument("--models", nargs="+", default=["svm", "nb", "knn", "mlp", "rf", "gbdt", "xgb", "lgbm", "cat"])
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--interp-test-wells", type=int, default=10)
    parser.add_argument("--max-rows-per-well", type=int, default=800)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--mlp-max-iter", type=int, default=80)
    parser.add_argument("--svm-max-iter", type=int, default=4000)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--knn-neighbors", type=int, default=11)
    parser.add_argument("--gbdt-max-iter", type=int, default=120)
    parser.add_argument("--no-missing-indicators", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=Path("results/spatial_multimethod_group_5seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/spatial_multimethod_group_5seed_summary.csv"))
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, args))
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")


if __name__ == "__main__":
    main()
