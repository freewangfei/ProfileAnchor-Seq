import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.spatial_multimethod_group_benchmark import build_features, load_force, sample_by_well, split_wells_by_space


DEFAULT_COVERAGES = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50]
CURVES = ["GR", "RHOB", "NPHI", "DTC"]


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


def safe_divide(a: pd.Series, b: pd.Series) -> np.ndarray:
    denom = b.to_numpy(dtype=np.float64)
    denom = np.where(np.abs(denom) < 1e-8, np.nan, denom)
    return a.to_numpy(dtype=np.float64) / denom


def add_integrated_features(df: pd.DataFrame, args) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    features = []
    curves = [curve for curve in CURVES if curve in out.columns]
    grouped = out.groupby("WELL", sort=False)
    for curve in curves:
        values = out[curve].to_numpy(dtype=np.float64)
        roll = grouped[curve].transform(lambda s: s.rolling(args.rolling_window, center=True, min_periods=1).mean())
        roll_std = grouped[curve].transform(lambda s: s.rolling(args.rolling_window, center=True, min_periods=2).std())
        roll_med = grouped[curve].transform(lambda s: s.rolling(args.rolling_window, center=True, min_periods=1).median())
        grad = grouped[curve].diff().to_numpy(dtype=np.float64)
        curv = grouped[curve].diff().diff().to_numpy(dtype=np.float64)
        names = {
            f"{curve}_ILF_LOCAL_MEAN": roll.to_numpy(dtype=np.float64),
            f"{curve}_ILF_LOCAL_STD": roll_std.to_numpy(dtype=np.float64),
            f"{curve}_ILF_LOCAL_MED": roll_med.to_numpy(dtype=np.float64),
            f"{curve}_ILF_LOCAL_DEV": values - roll.to_numpy(dtype=np.float64),
            f"{curve}_ILF_MED_DEV": values - roll_med.to_numpy(dtype=np.float64),
            f"{curve}_ILF_GRAD": grad,
            f"{curve}_ILF_ABS_GRAD": np.abs(grad),
            f"{curve}_ILF_CURV": curv,
            f"{curve}_ILF_ABS_CURV": np.abs(curv),
        }
        for name, col in names.items():
            out[name] = col
            features.append(name)
    pairs = [("GR", "RHOB"), ("GR", "NPHI"), ("GR", "DTC"), ("NPHI", "RHOB"), ("DTC", "RHOB"), ("DTC", "NPHI")]
    for a, b in pairs:
        if {a, b}.issubset(out.columns):
            out[f"{a}_{b}_ILF_RATIO"] = safe_divide(out[a], out[b])
            out[f"{a}_{b}_ILF_DIFF"] = out[a].to_numpy(dtype=np.float64) - out[b].to_numpy(dtype=np.float64)
            features.extend([f"{a}_{b}_ILF_RATIO", f"{a}_{b}_ILF_DIFF"])
    if {"GR", "NPHI", "RHOB"}.issubset(out.columns):
        out["GR_NPHI_RHOB_ILF_PRODUCT"] = out["GR"].to_numpy(dtype=np.float64) * out["NPHI"].to_numpy(dtype=np.float64) * out["RHOB"].to_numpy(dtype=np.float64)
        features.append("GR_NPHI_RHOB_ILF_PRODUCT")
    if {"X_LOC", "Y_LOC", "Z_LOC", "DEPTH_MD"}.issubset(out.columns):
        out["ILF_RELATIVE_TVD"] = out["Z_LOC"].to_numpy(dtype=np.float64) - out["DEPTH_MD"].to_numpy(dtype=np.float64)
        out["ILF_XY_RADIUS"] = np.sqrt(out["X_LOC"].to_numpy(dtype=np.float64) ** 2 + out["Y_LOC"].to_numpy(dtype=np.float64) ** 2)
        features.extend(["ILF_RELATIVE_TVD", "ILF_XY_RADIUS"])
    return out, features


class IntegratedFeatureEnsemble:
    def __init__(self, seed: int, args):
        self.seed = seed
        self.args = args
        self.base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.quantile = make_pipeline(
            SimpleImputer(strategy="median"),
            QuantileTransformer(n_quantiles=args.n_quantiles, output_distribution="normal", random_state=seed, subsample=200000),
        )
        self.rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_depth=args.max_depth,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=args.n_jobs,
        )
        self.et = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_depth=args.max_depth,
            class_weight="balanced",
            random_state=seed + 23,
            n_jobs=args.n_jobs,
        )
        self.hgb = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=args.learning_rate,
            max_iter=args.gbdt_max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.gbdt_min_samples_leaf,
            class_weight="balanced",
            l2_regularization=args.l2_regularization,
            early_stopping=False,
            random_state=seed + 47,
        )

    def _features(self, x: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            scaled = self.base.fit_transform(x)
            quant = self.quantile.fit_transform(x)
        else:
            scaled = self.base.transform(x)
            quant = self.quantile.transform(x)
        return np.hstack([scaled, quant])

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.classes_ = np.array(sorted(np.unique(y)))
        xt = self._features(x, fit=True)
        self.rf.fit(xt, y)
        self.et.fit(xt, y)
        self.hgb.fit(xt, y)
        return self

    def _aligned(self, estimator, x: np.ndarray) -> np.ndarray:
        raw = estimator.predict_proba(x)
        out = np.zeros((x.shape[0], len(self.classes_)), dtype=np.float64)
        for j, label in enumerate(estimator.classes_):
            out[:, int(np.where(self.classes_ == label)[0][0])] = raw[:, j]
        return out

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        xt = self._features(x, fit=False)
        proba = (
            self.args.rf_weight * self._aligned(self.rf, xt)
            + self.args.et_weight * self._aligned(self.et, xt)
            + self.args.hgb_weight * self._aligned(self.hgb, xt)
        )
        return normalized_proba(proba)


def global_proba(model: IntegratedFeatureEnsemble, x: np.ndarray, n_classes: int) -> np.ndarray:
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
    df, integrated_features = add_integrated_features(df, args)
    features = base_features + integrated_features
    train_mask = df["WELL"].isin(train_wells).to_numpy()
    x_train = df.loc[train_mask, features].to_numpy(dtype=np.float64)
    y_train = df.loc[train_mask, "TARGET"].to_numpy(dtype=np.int64)
    start = time.time()
    model = IntegratedFeatureEnsemble(seed, args).fit(x_train, y_train)
    train_time = time.time() - start
    rows = []
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        test = df[df["WELL"].isin(wells)].copy().sort_values(["WELL", "DEPTH_MD"])
        x_test = test[features].to_numpy(dtype=np.float64)
        proba = global_proba(model, x_test, len(class_names))
        y = test["TARGET"].to_numpy(dtype=np.int64)
        rows.extend(selective_rows(y, proba, seed, split, "integrated_logging_features_margin", args.coverages))
        full = metrics(y, proba.argmax(axis=1))
        full.update({"seed": seed, "split": split, "method": "integrated_logging_features_full", "coverage": 1.0, "kept_rows": len(y)})
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
    if "seed" not in base.columns:
        return pd.DataFrame()
    cand_all = raw[(raw["split"] == "extrapolation") & (raw["method"] == "integrated_logging_features_margin")]
    rows = []
    for coverage in sorted(cand_all["coverage"].unique()):
        c = cand_all[cand_all["coverage"] == coverage].set_index("seed")
        for baseline in ["ProfileAnchor-Seq", "Random forest"]:
            b = base[(base["method"] == baseline) & (base["split"] == "extrapolation") & (base["coverage"] == coverage)].set_index("seed")
            common = sorted(set(c.index) & set(b.index))
            if len(common) < 2:
                continue
            for metric in ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted"]:
                cv = c.loc[common, metric].to_numpy(dtype=np.float64)
                bv = b.loc[common, metric].to_numpy(dtype=np.float64)
                diff = cv - bv
                rows.append(
                    {
                        "method": "integrated_logging_features_margin",
                        "baseline": baseline,
                        "coverage": coverage,
                        "metric": metric,
                        "n": len(common),
                        "method_mean": float(cv.mean()),
                        "baseline_mean": float(bv.mean()),
                        "delta_mean": float(diff.mean()),
                        "wins": int((diff > 0).sum()),
                        "losses": int((diff < 0).sum()),
                        "paired_t_p": float(ttest_rel(cv, bv).pvalue),
                        "wilcoxon_p": float(wilcoxon(diff).pvalue) if (diff != 0).any() else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def self_check(args):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "WELL": np.repeat(["A", "B"], 30),
            "DEPTH_MD": np.tile(np.arange(30), 2),
            "X_LOC": rng.normal(size=60),
            "Y_LOC": rng.normal(size=60),
            "Z_LOC": rng.normal(size=60),
            "GR": rng.normal(size=60),
            "RHOB": rng.normal(size=60) + 2,
            "NPHI": rng.normal(size=60),
            "DTC": rng.normal(size=60) + 80,
            "TARGET": np.array([0, 1, 2] * 20),
        }
    )
    df, base = build_features(df, include_missing=True)
    df, integ = add_integrated_features(df, args)
    x = df[base + integ].to_numpy(dtype=np.float64)
    y = df["TARGET"].to_numpy(dtype=np.int64)
    model = IntegratedFeatureEnsemble(0, args).fit(x[:45], y[:45])
    proba = global_proba(model, x[45:], 3)
    rows = selective_rows(y[45:], proba, 0, "self_check", "integrated_logging_features_margin", [0.4])
    if proba.shape != (15, 3) or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Integrated logging feature probabilities are invalid.")
    if rows[0]["kept_rows"] != 6:
        raise RuntimeError("Integrated logging feature selective check failed.")
    print("integrated_logging_features_baseline self-check passed")


def main():
    parser = argparse.ArgumentParser(description="Integrated-logging-feature lithofacies baseline under FORCE complete-well release protocol.")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/force2020"))
    parser.add_argument("--target", default="FORCE_2020_LITHOFACIES_LITHOLOGY")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42])
    parser.add_argument("--coverages", nargs="+", type=float, default=DEFAULT_COVERAGES)
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--interp-test-wells", type=int, default=10)
    parser.add_argument("--max-rows-per-well", type=int, default=None)
    parser.add_argument("--rolling-window", type=int, default=15)
    parser.add_argument("--n-estimators", type=int, default=90)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--gbdt-max-iter", type=int, default=90)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--gbdt-min-samples-leaf", type=int, default=20)
    parser.add_argument("--l2-regularization", type=float, default=1e-3)
    parser.add_argument("--n-quantiles", type=int, default=256)
    parser.add_argument("--rf-weight", type=float, default=0.35)
    parser.add_argument("--et-weight", type=float, default=0.40)
    parser.add_argument("--hgb-weight", type=float, default=0.25)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out-csv", type=Path, default=Path("results/recent_integrated_logging_features_force_11seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/recent_integrated_logging_features_force_11seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/recent_integrated_logging_features_force_11seed_paired.csv"))
    parser.add_argument("--reference-summary", type=Path, default=Path("results/profile_anchor_seq_force_11seed_summary.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
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
