import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.spatial_multimethod_group_benchmark import build_features, load_force, sample_by_well, split_wells_by_space


DEFAULT_COVERAGES = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50]


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "MCC": matthews_corrcoef(y_true, pred),
        "F1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted", "kept_rows"]
    summary = raw.groupby(["method", "split", "coverage"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-12, None)
    return x / x.sum(axis=1, keepdims=True)


def selective_rows(y: np.ndarray, proba: np.ndarray, seed: int, split: str, method: str, coverages: list[float]) -> list[dict]:
    pred = proba.argmax(axis=1)
    ordered = np.sort(proba, axis=1)
    score = ordered[:, -1] - ordered[:, -2]
    rank = np.argsort(-score)
    rows = []
    for coverage in coverages:
        keep = max(1, int(round(len(y) * coverage)))
        idx = rank[:keep]
        row = metric_row(y[idx], pred[idx])
        row.update({"seed": seed, "split": split, "method": method, "coverage": coverage, "kept_rows": keep})
        rows.append(row)
    return rows


class PDSMVKNN:
    def __init__(self, n_neighbors: int, power: float, distance_floor: float, view_weights: list[float] | None = None):
        self.n_neighbors = int(n_neighbors)
        self.power = float(power)
        self.distance_floor = float(distance_floor)
        self.view_weights = view_weights

    def fit(self, x: np.ndarray, y: np.ndarray, views: list[list[int]]):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.imputer.fit_transform(x))
        self.y_train = np.asarray(y, dtype=np.int64)
        self.classes_ = np.array(sorted(np.unique(self.y_train)))
        self.views = [np.array(v, dtype=np.int64) for v in views if len(v) > 0]
        if self.view_weights is None:
            self.view_weights_ = np.ones(len(self.views), dtype=np.float64) / max(1, len(self.views))
        else:
            weights = np.asarray(self.view_weights[: len(self.views)], dtype=np.float64)
            weights = np.clip(weights, 0.0, None)
            if weights.sum() <= 0:
                weights = np.ones(len(self.views), dtype=np.float64)
            self.view_weights_ = weights / weights.sum()
        return self

    def predict_proba(self, x: np.ndarray, n_classes: int) -> np.ndarray:
        x_test = self.scaler.transform(self.imputer.transform(x))
        out = np.zeros((x_test.shape[0], n_classes), dtype=np.float64)
        batch = 512
        for start in range(0, x_test.shape[0], batch):
            stop = min(x_test.shape[0], start + batch)
            proba = np.zeros((stop - start, len(self.classes_)), dtype=np.float64)
            for view_weight, idx in zip(self.view_weights_, self.views):
                xt = x_test[start:stop, :][:, idx]
                xr = self.x_train[:, idx]
                dist = np.sqrt(((xt[:, None, :] - xr[None, :, :]) ** 2).mean(axis=2))
                k = min(self.n_neighbors, dist.shape[1])
                nn = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
                nn_dist = np.take_along_axis(dist, nn, axis=1)
                weights = 1.0 / np.power(nn_dist + self.distance_floor, self.power)
                labels = self.y_train[nn]
                view_proba = np.zeros_like(proba)
                for local, cls in enumerate(self.classes_):
                    view_proba[:, local] = (weights * (labels == cls)).sum(axis=1)
                proba += view_weight * normalize_rows(view_proba)
            proba = normalize_rows(proba)
            for local, cls in enumerate(self.classes_):
                out[start:stop, int(cls)] = proba[:, local]
        return normalize_rows(out)


def make_views(features: list[str]) -> list[list[int]]:
    groups = [
        ["GR"],
        ["RHOB", "NPHI"],
        ["DTC"],
        ["DEPTH_MD", "Z_LOC"],
        ["X_LOC", "Y_LOC"],
    ]
    views = []
    for group in groups:
        idx = [i for i, name in enumerate(features) if any(token in name for token in group)]
        if idx:
            views.append(sorted(set(idx)))
    used = sorted({i for view in views for i in view})
    rest = [i for i in range(len(features)) if i not in used]
    if rest:
        views.append(rest)
    return views


def run_seed(seed: int, args) -> list[dict]:
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, features = build_features(df, include_missing=True)
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    train_wells, interp_wells, extra_wells, _ = split_wells_by_space(df, args.train_fraction, args.interp_test_wells, seed)
    train = df[df["WELL"].isin(train_wells)].copy()
    x_train = train[features].to_numpy(dtype=np.float64)
    y_train = train["TARGET"].to_numpy(dtype=np.int64)
    views = make_views(features)
    model = PDSMVKNN(args.n_neighbors, args.power, args.distance_floor).fit(x_train, y_train, views)
    rows = []
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        test = df[df["WELL"].isin(wells)].copy().sort_values(["WELL", "DEPTH_MD"])
        x_test = test[features].to_numpy(dtype=np.float64)
        y = test["TARGET"].to_numpy(dtype=np.int64)
        proba = model.predict_proba(x_test, len(class_names))
        rows.extend(selective_rows(y, proba, seed, split, "pdsmvknn_margin", args.coverages))
        full = metric_row(y, proba.argmax(axis=1))
        full.update({"seed": seed, "split": split, "method": "pdsmvknn_full", "coverage": 1.0, "kept_rows": len(y)})
        rows.append(full)
    return rows


def self_check(args):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 10))
    y = np.array([0, 1, 2, 3] * 20)
    model = PDSMVKNN(5, 2.0, 1e-3).fit(x[:60], y[:60], [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]])
    proba = model.predict_proba(x[60:], 4)
    rows = selective_rows(y[60:], proba, 0, "self_check", "pdsmvknn_margin", [0.25])
    if proba.shape != (20, 4) or not np.allclose(proba.sum(axis=1), 1.0):
        raise RuntimeError("PDS-MVKNN probability check failed")
    if rows[0]["kept_rows"] != 5:
        raise RuntimeError("PDS-MVKNN selective release check failed")
    print("recent_pdsmvknn_baseline self-check passed")


def main():
    parser = argparse.ArgumentParser(description="PDS-MVKNN-style lithology baseline under FORCE complete-well release protocol.")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data/force2020"))
    parser.add_argument("--target", default="FORCE_2020_LITHOFACIES_LITHOLOGY")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42])
    parser.add_argument("--coverages", nargs="+", type=float, default=DEFAULT_COVERAGES)
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--interp-test-wells", type=int, default=10)
    parser.add_argument("--max-rows-per-well", type=int, default=800)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--distance-floor", type=float, default=1e-3)
    parser.add_argument("--out-csv", type=Path, default=Path("profile_anchor_code/results/recent_pdsmvknn_force_11seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("profile_anchor_code/results/recent_pdsmvknn_force_11seed_summary.csv"))
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
    summary.to_csv(args.summary_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")


if __name__ == "__main__":
    main()
