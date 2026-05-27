import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.spatial_multimethod_group_benchmark import build_features, load_force, sample_by_well, split_wells_by_space


DEFAULT_COVERAGES = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50]


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


def kmeans_smote(x: np.ndarray, y: np.ndarray, seed: int, args) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    target = int(np.ceil(np.quantile(counts, args.target_quantile)))
    max_new = max(0, int(len(y) * args.max_augmented_multiplier) - len(y))
    if target <= 0 or max_new <= 0:
        return x, y, 0
    needs = {int(label): max(0, target - int(count)) for label, count in zip(labels, counts)}
    needs = {label: need for label, need in needs.items() if need > 0 and np.sum(y == label) >= args.smote_k + 1}
    total_need = sum(needs.values())
    if total_need == 0:
        return x, y, 0
    if total_need > max_new:
        scale = max_new / total_need
        needs = {label: int(np.floor(need * scale)) for label, need in needs.items()}
        needs = {label: need for label, need in needs.items() if need > 0}
    if not needs:
        return x, y, 0

    n_clusters = min(args.kmeans_clusters, max(2, len(x) // args.min_cluster_size))
    clusters = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(x)
    xs = [x]
    ys = [y]
    for label, need in needs.items():
        class_idx = np.flatnonzero(y == label)
        class_clusters = clusters[class_idx]
        usable = []
        for cluster in np.unique(class_clusters):
            idx = class_idx[class_clusters == cluster]
            if len(idx) >= args.smote_k + 1:
                usable.append(idx)
        if not usable:
            continue
        sizes = np.array([len(idx) for idx in usable], dtype=np.float64)
        alloc = np.floor(need * sizes / sizes.sum()).astype(int)
        remainder = need - int(alloc.sum())
        if remainder > 0:
            order = np.argsort(-sizes)
            alloc[order[:remainder]] += 1
        for idx, n_new in zip(usable, alloc):
            if n_new <= 0:
                continue
            anchors = rng.choice(idx, size=n_new, replace=True)
            neigh = []
            for anchor in anchors:
                candidates = idx[idx != anchor]
                neigh.append(rng.choice(candidates))
            neigh = np.asarray(neigh, dtype=int)
            lam = rng.random((n_new, 1))
            xs.append(x[anchors] + lam * (x[neigh] - x[anchors]))
            ys.append(np.full(n_new, label, dtype=y.dtype))
    x_aug = np.vstack(xs)
    y_aug = np.concatenate(ys)
    return x_aug, y_aug, len(y_aug) - len(y)


class CascadeForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_layers=3, n_estimators=160, max_depth=None, min_samples_leaf=2, random_state=0, n_jobs=1):
        self.n_layers = n_layers
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, x, y):
        self.classes_ = np.array(sorted(np.unique(y)))
        features = x
        self.layers_ = []
        for layer in range(self.n_layers):
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced_subsample",
                random_state=self.random_state + 17 * layer,
                n_jobs=self.n_jobs,
            )
            et = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced",
                random_state=self.random_state + 17 * layer + 7,
                n_jobs=self.n_jobs,
            )
            rf.fit(features, y)
            et.fit(features, y)
            self.layers_.append((rf, et))
            features = np.hstack([x, self._layer_proba((rf, et), features)])
        return self

    def _aligned(self, estimator, features):
        raw = estimator.predict_proba(features)
        out = np.zeros((features.shape[0], len(self.classes_)), dtype=np.float64)
        for j, label in enumerate(estimator.classes_):
            pos = int(np.where(self.classes_ == label)[0][0])
            out[:, pos] = raw[:, j]
        return out

    def _layer_proba(self, layer, features):
        return normalized_proba(0.5 * self._aligned(layer[0], features) + 0.5 * self._aligned(layer[1], features))

    def predict_proba(self, x):
        features = x
        proba = None
        for layer in self.layers_:
            proba = self._layer_proba(layer, features)
            features = np.hstack([x, proba])
        return normalized_proba(proba)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


def fit_deepforest(x_train: np.ndarray, y_train: np.ndarray, seed: int, args):
    x_aug, y_aug, synthetic_rows = kmeans_smote(x_train, y_train, seed, args)
    model = CascadeForestClassifier(
        n_layers=args.cascade_layers,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=seed,
        n_jobs=args.n_jobs,
    )
    model.fit(x_aug, y_aug)
    return model, synthetic_rows


def local_to_global_proba(model: CascadeForestClassifier, x: np.ndarray, n_classes: int) -> np.ndarray:
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
    df, features = build_features(df, include_missing=True)
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    train_wells, interp_wells, extra_wells, _ = split_wells_by_space(df, args.train_fraction, args.interp_test_wells, seed)
    train = df[df["WELL"].isin(train_wells)].copy()
    n_classes = len(class_names)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[features]))
    y_train = train["TARGET"].to_numpy(dtype=np.int64)
    start = time.time()
    model, synthetic_rows = fit_deepforest(x_train, y_train, seed, args)
    train_time = time.time() - start
    rows = []
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        test = df[df["WELL"].isin(wells)].copy()
        ordered = test.sort_values(["WELL", "DEPTH_MD"])
        x_test = scaler.transform(imputer.transform(ordered[features]))
        proba = local_to_global_proba(model, x_test, n_classes)
        y = ordered["TARGET"].to_numpy(dtype=np.int64)
        rows.extend(selective_rows(y, proba, seed, split, "deepforest_kmeans_smote_margin", args.coverages))
        full = metrics(y, proba.argmax(axis=1))
        full.update({"seed": seed, "split": split, "method": "deepforest_kmeans_smote_full", "coverage": 1.0, "kept_rows": len(y)})
        rows.append(full)
    for row in rows:
        row.update(
            {
                "synthetic_rows": synthetic_rows,
                "cascade_layers": args.cascade_layers,
                "n_estimators": args.n_estimators,
                "train_time": train_time,
            }
        )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted", "kept_rows", "synthetic_rows"]
    summary = raw.groupby(["method", "split", "coverage"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def paired_stats(raw: pd.DataFrame, baseline_csv: Path) -> pd.DataFrame:
    if not baseline_csv.exists():
        return pd.DataFrame()
    base = pd.read_csv(baseline_csv)
    if "seed" not in base.columns:
        return pd.DataFrame()
    rows = []
    cand_all = raw[(raw["split"] == "extrapolation") & (raw["method"] == "deepforest_kmeans_smote_margin")]
    for coverage in sorted(set(cand_all["coverage"]) - {1.0}):
        cand = cand_all[cand_all["coverage"] == coverage].set_index("seed")
        for baseline in ["ProfileAnchor-Seq", "Random forest"]:
            b = base[(base["method"] == baseline) & (base["split"] == "extrapolation") & (base["coverage"] == coverage)].set_index("seed")
            common = sorted(set(cand.index) & set(b.index))
            if not common:
                continue
            for metric in ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted"]:
                cv = cand.loc[common, metric]
                bv = b.loc[common, metric]
                diff = cv - bv
                rows.append(
                    {
                        "method": "deepforest_kmeans_smote_margin",
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
    x = rng.normal(size=(80, 6))
    y = np.array([0] * 45 + [1] * 25 + [2] * 10)
    model, synthetic_rows = fit_deepforest(x, y, 0, args)
    proba = local_to_global_proba(model, x[:12], 3)
    rows = selective_rows(y[:12], proba, 0, "self_check", "deepforest_kmeans_smote_margin", [0.5])
    if proba.shape != (12, 3) or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("DeepForest probabilities are invalid.")
    if synthetic_rows <= 0 or rows[0]["kept_rows"] != 6:
        raise RuntimeError("DeepForest selective self-check failed.")
    print("deepforest_kmeans_smote_baseline self-check passed")


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
    parser.add_argument("--target-quantile", type=float, default=0.75)
    parser.add_argument("--max-augmented-multiplier", type=float, default=1.5)
    parser.add_argument("--kmeans-clusters", type=int, default=12)
    parser.add_argument("--min-cluster-size", type=int, default=80)
    parser.add_argument("--smote-k", type=int, default=5)
    parser.add_argument("--cascade-layers", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--out-csv", type=Path, default=Path("results/recent_deepforest_kmeans_smote_force_11seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/recent_deepforest_kmeans_smote_force_11seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/recent_deepforest_kmeans_smote_force_11seed_paired.csv"))
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
