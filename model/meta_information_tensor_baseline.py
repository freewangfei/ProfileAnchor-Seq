import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import QuantileTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.spatial_multimethod_group_benchmark import build_features, load_force, sample_by_well, split_wells_by_space


DEFAULT_COVERAGES = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50]
LOG_CURVES = ["GR", "RHOB", "NPHI", "DTC"]


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "MCC": matthews_corrcoef(y_true, pred),
        "F1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
    }


def normalized_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(proba, 1e-12, None)
    return proba / proba.sum(axis=1, keepdims=True)


def selective_rows(y: np.ndarray, proba: np.ndarray, seed: int, split: str, method: str, coverages: list[float]) -> list[dict]:
    pred = proba.argmax(axis=1)
    ordered = np.sort(proba, axis=1)
    score = ordered[:, -1] - ordered[:, -2]
    rank = np.argsort(-score)
    rows = []
    for coverage in coverages:
        keep = max(1, int(round(len(y) * coverage)))
        idx = rank[:keep]
        row = metrics(y[idx], pred[idx])
        row.update({"seed": seed, "split": split, "method": method, "coverage": coverage, "kept_rows": keep})
        rows.append(row)
    return rows


def add_quality_features(df: pd.DataFrame, train_mask: np.ndarray, args) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    features = []
    curves = [curve for curve in LOG_CURVES if curve in out.columns]
    for curve in curves:
        values = out[curve].to_numpy(dtype=np.float64)
        missing = np.isnan(values).astype(np.float64)
        train_values = out.loc[train_mask, curve].dropna().to_numpy(dtype=np.float64)
        if len(train_values) < 10:
            center = np.nanmedian(values)
            scale = np.nanstd(values)
            low = np.nanpercentile(values, args.outlier_low)
            high = np.nanpercentile(values, args.outlier_high)
        else:
            center = np.median(train_values)
            scale = np.std(train_values)
            low = np.percentile(train_values, args.outlier_low)
            high = np.percentile(train_values, args.outlier_high)
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        z = (values - center) / scale
        outlier = ((values < low) | (values > high)).astype(np.float64)
        outlier[np.isnan(values)] = 1.0
        well_mean = out.groupby("WELL")[curve].transform("mean").to_numpy(dtype=np.float64)
        well_std = out.groupby("WELL")[curve].transform("std").replace(0, np.nan).to_numpy(dtype=np.float64)
        well_z = (values - well_mean) / well_std
        depth_grad = out.groupby("WELL")[curve].diff().to_numpy(dtype=np.float64)
        depth_curv = out.groupby("WELL")[curve].diff().diff().to_numpy(dtype=np.float64)
        local_med = out.groupby("WELL")[curve].transform(lambda s: s.rolling(args.rolling_window, center=True, min_periods=1).median())
        local_std = out.groupby("WELL")[curve].transform(lambda s: s.rolling(args.rolling_window, center=True, min_periods=2).std())
        local_dev = values - local_med.to_numpy(dtype=np.float64)
        quality = 1.0 - missing
        quality *= np.exp(-np.minimum(np.abs(np.nan_to_num(z, nan=0.0)), args.z_clip) / args.z_clip)
        quality *= 1.0 - args.outlier_penalty * outlier
        quality = np.clip(quality, 0.0, 1.0)
        names_values = {
            f"{curve}_QRAW": values,
            f"{curve}_QZ": z,
            f"{curve}_QWELLZ": well_z,
            f"{curve}_QGRAD": depth_grad,
            f"{curve}_QCURV": depth_curv,
            f"{curve}_QLOCAL_DEV": local_dev,
            f"{curve}_QLOCAL_STD": local_std.to_numpy(dtype=np.float64),
            f"{curve}_QMISSING": missing,
            f"{curve}_QOUTLIER": outlier,
            f"{curve}_QWEIGHT": quality,
            f"{curve}_QWEIGHTED": np.nan_to_num(values, nan=center) * quality,
            f"{curve}_QABSZ": np.abs(z),
        }
        for name, col in names_values.items():
            out[name] = col
            features.append(name)
    if {"GR", "RHOB"}.issubset(out.columns):
        out["GR_RHOB_QINTERACT"] = out["GR_QWEIGHTED"] * out["RHOB_QWEIGHT"]
        features.append("GR_RHOB_QINTERACT")
    if {"NPHI", "RHOB"}.issubset(out.columns):
        out["NPHI_RHOB_QCONTRAST"] = out["NPHI_QWELLZ"] - out["RHOB_QWELLZ"]
        features.append("NPHI_RHOB_QCONTRAST")
    if {"GR", "DTC"}.issubset(out.columns):
        out["GR_DTC_QCONTRAST"] = out["GR_QWELLZ"] - out["DTC_QWELLZ"]
        features.append("GR_DTC_QCONTRAST")
    out["QUALITY_MEAN"] = out[[f"{curve}_QWEIGHT" for curve in curves]].mean(axis=1)
    out["QUALITY_MIN"] = out[[f"{curve}_QWEIGHT" for curve in curves]].min(axis=1)
    features.extend(["QUALITY_MEAN", "QUALITY_MIN"])
    return out, features


class QualityAwareForest:
    def __init__(self, seed: int, args):
        self.seed = seed
        self.args = args
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.quantile = QuantileTransformer(
            n_quantiles=args.n_quantiles,
            output_distribution="normal",
            random_state=seed,
            subsample=200000,
        )
        self.rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=args.n_jobs,
        )
        self.et = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            random_state=seed + 101,
            n_jobs=args.n_jobs,
        )
        self.hgb = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=args.learning_rate,
            max_iter=args.gbdt_max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.gbdt_min_samples_leaf,
            l2_regularization=args.l2_regularization,
            class_weight="balanced",
            random_state=seed + 202,
        ) if args.hgb_weight > 0 else None

    def _transform(self, x: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            base = self.imputer.fit_transform(x)
            scaled = self.scaler.fit_transform(base)
            quant = self.quantile.fit_transform(base)
        else:
            base = self.imputer.transform(x)
            scaled = self.scaler.transform(base)
            quant = self.quantile.transform(base)
        return np.hstack([base, scaled, quant])

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.classes_ = np.array(sorted(np.unique(y)))
        x_t = self._transform(x, fit=True)
        self.rf.fit(x_t, y)
        self.et.fit(x_t, y)
        if self.hgb is not None:
            self.hgb.fit(x_t, y)
        return self

    def _aligned(self, estimator, x: np.ndarray) -> np.ndarray:
        raw = estimator.predict_proba(x)
        out = np.zeros((x.shape[0], len(self.classes_)), dtype=np.float64)
        for j, label in enumerate(estimator.classes_):
            out[:, int(np.where(self.classes_ == label)[0][0])] = raw[:, j]
        return out

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_t = self._transform(x, fit=False)
        proba = (
            self.args.rf_weight * self._aligned(self.rf, x_t)
            + self.args.et_weight * self._aligned(self.et, x_t)
        )
        if self.hgb is not None:
            proba += self.args.hgb_weight * self._aligned(self.hgb, x_t)
        return normalized_proba(proba)


def global_proba(model: QualityAwareForest, x: np.ndarray, n_classes: int) -> np.ndarray:
    local = model.predict_proba(x)
    out = np.zeros((len(x), n_classes), dtype=np.float64)
    for j, label in enumerate(model.classes_):
        out[:, int(label)] = local[:, j]
    missing = out.sum(axis=1) == 0
    if np.any(missing):
        out[missing] = 1.0 / n_classes
    return normalized_proba(out)


def run_seed(seed: int, args) -> list[dict]:
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, base_features = build_features(df, include_missing=True)
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    train_wells, interp_wells, extra_wells, _ = split_wells_by_space(df, args.train_fraction, args.interp_test_wells, seed)
    train_mask = df["WELL"].isin(train_wells).to_numpy()
    df, quality_features = add_quality_features(df, train_mask, args)
    features = base_features + quality_features
    train = df[df["WELL"].isin(train_wells)].copy()
    x_train = train[features].to_numpy(dtype=np.float64)
    y_train = train["TARGET"].to_numpy(dtype=np.int64)
    start = time.time()
    model = QualityAwareForest(seed, args).fit(x_train, y_train)
    train_time = time.time() - start
    rows = []
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        test = df[df["WELL"].isin(wells)].copy()
        ordered = test.sort_values(["WELL", "DEPTH_MD"])
        proba = global_proba(model, ordered[features].to_numpy(dtype=np.float64), len(class_names))
        y = ordered["TARGET"].to_numpy(dtype=np.int64)
        rows.extend(selective_rows(y, proba, seed, split, "meta_information_tensor_margin", args.coverages))
        full = metrics(y, proba.argmax(axis=1))
        full.update({"seed": seed, "split": split, "method": "meta_information_tensor_full", "coverage": 1.0, "kept_rows": len(y)})
        rows.append(full)
    for row in rows:
        row.update({"n_features": len(features), "train_time": train_time})
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted", "kept_rows"]
    summary = raw.groupby(["method", "split", "coverage"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def paired_stats(raw: pd.DataFrame, baseline_csv: Path) -> pd.DataFrame:
    if not baseline_csv.exists():
        return pd.DataFrame()
    base = pd.read_csv(baseline_csv)
    metric_cols = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted"]
    cand_all = raw[(raw["split"] == "extrapolation") & (raw["method"] == "meta_information_tensor_margin")]
    rows = []
    for coverage in sorted(cand_all["coverage"].unique()):
        c = cand_all[cand_all["coverage"] == coverage].set_index("seed")
        for baseline in sorted(base["method"].unique()):
            b = base[(base["method"] == baseline) & (base["split"] == "extrapolation") & (base["coverage"] == coverage)].set_index("seed")
            common = sorted(set(c.index).intersection(b.index))
            if len(common) < 2:
                continue
            for metric in metric_cols:
                cv = c.loc[common, metric].to_numpy(dtype=np.float64)
                bv = b.loc[common, metric].to_numpy(dtype=np.float64)
                diff = cv - bv
                rows.append(
                    {
                        "method": "meta_information_tensor_margin",
                        "baseline": baseline,
                        "coverage": coverage,
                        "metric": metric,
                        "n": len(common),
                        "method_mean": float(cv.mean()),
                        "baseline_mean": float(bv.mean()),
                        "delta_mean": float(diff.mean()),
                        "wins": int((diff > 0).sum()),
                        "losses": int((diff < 0).sum()),
                        "paired_t_p": float(ttest_rel(cv, bv).pvalue) if len(common) > 1 else np.nan,
                        "wilcoxon_p": float(wilcoxon(diff).pvalue) if len(common) > 1 and (diff != 0).any() else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def self_check(args):
    rng = np.random.default_rng(0)
    n = 90
    wells = np.repeat(["A", "B", "C"], n // 3)
    frame = pd.DataFrame(
        {
            "WELL": wells,
            "DEPTH_MD": np.tile(np.arange(n // 3), 3).astype(float),
            "X_LOC": np.repeat([0.0, 1.0, 2.0], n // 3),
            "Y_LOC": np.repeat([2.0, 1.0, 0.0], n // 3),
            "Z_LOC": -np.tile(np.arange(n // 3), 3).astype(float),
            "GR": rng.normal(80, 15, n),
            "RHOB": rng.normal(2.4, 0.1, n),
            "NPHI": rng.normal(0.25, 0.04, n),
            "DTC": rng.normal(90, 8, n),
            "TARGET": np.array([0] * 30 + [1] * 30 + [2] * 30),
        }
    )
    frame.loc[[3, 11, 55], "GR"] = np.nan
    train_mask = frame["WELL"].isin(["A", "B"]).to_numpy()
    frame, quality_features = add_quality_features(frame, train_mask, args)
    model = QualityAwareForest(0, args).fit(frame.loc[train_mask, quality_features].to_numpy(), frame.loc[train_mask, "TARGET"].to_numpy())
    proba = global_proba(model, frame.loc[~train_mask, quality_features].to_numpy(), 3)
    rows = selective_rows(frame.loc[~train_mask, "TARGET"].to_numpy(), proba, 0, "self_check", "meta_information_tensor_margin", [0.2])
    if proba.shape[1] != 3 or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Meta-information probabilities are invalid.")
    if rows[0]["kept_rows"] != 6 or len(quality_features) < 20:
        raise RuntimeError("Meta-information selective self-check failed.")
    print("meta_information_tensor_baseline self-check passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/force2020"))
    parser.add_argument("--target", default="FORCE_2020_LITHOFACIES_LITHOLOGY")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42])
    parser.add_argument("--coverages", nargs="+", type=float, default=DEFAULT_COVERAGES)
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--interp-test-wells", type=int, default=10)
    parser.add_argument("--max-rows-per-well", type=int, default=None)
    parser.add_argument("--rolling-window", type=int, default=9)
    parser.add_argument("--outlier-low", type=float, default=1.0)
    parser.add_argument("--outlier-high", type=float, default=99.0)
    parser.add_argument("--z-clip", type=float, default=6.0)
    parser.add_argument("--outlier-penalty", type=float, default=0.35)
    parser.add_argument("--n-quantiles", type=int, default=200)
    parser.add_argument("--n-estimators", type=int, default=45)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--gbdt-max-iter", type=int, default=90)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--gbdt-min-samples-leaf", type=int, default=20)
    parser.add_argument("--l2-regularization", type=float, default=1e-3)
    parser.add_argument("--rf-weight", type=float, default=0.55)
    parser.add_argument("--et-weight", type=float, default=0.45)
    parser.add_argument("--hgb-weight", type=float, default=0.0)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--out-csv", type=Path, default=Path("results/recent_meta_information_tensor_force_11seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/recent_meta_information_tensor_force_11seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/recent_meta_information_tensor_force_11seed_paired.csv"))
    parser.add_argument("--reference-summary", type=Path, default=Path("results/force_release_seed_level_reference.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    weight_sum = args.rf_weight + args.et_weight + args.hgb_weight
    args.rf_weight /= weight_sum
    args.et_weight /= weight_sum
    args.hgb_weight /= weight_sum
    if args.self_check:
        self_check(args)
        return
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
    summary = summarize(raw)
    paired = paired_stats(raw, args.reference_summary)
    summary.to_csv(args.summary_csv, index=False)
    paired.to_csv(args.paired_csv, index=False)
    print(summary.to_string(index=False))
    if not paired.empty:
        print(paired.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.paired_csv}")


if __name__ == "__main__":
    main()
