import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.neighbors import NearestNeighbors
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


def class_prototypes(x: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    proto = np.zeros((n_classes, x.shape[1]), dtype=np.float64)
    global_center = np.nanmean(x, axis=0)
    for cls in range(n_classes):
        idx = y == cls
        proto[cls] = np.nanmean(x[idx], axis=0) if np.any(idx) else global_center
    return proto


def graph_features(x: np.ndarray, source_x: np.ndarray, source_y: np.ndarray, prototypes: np.ndarray, args) -> np.ndarray:
    n_classes = prototypes.shape[0]
    k = min(args.knn, len(source_x))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=args.n_jobs)
    nn.fit(source_x)
    dist, ind = nn.kneighbors(x, return_distance=True)
    sigma = np.median(dist[:, -1])
    if not np.isfinite(sigma) or sigma <= 1e-8:
        sigma = 1.0
    weights = np.exp(-dist / sigma)
    weights = weights / weights.sum(axis=1, keepdims=True).clip(min=1e-12)
    neigh_y = source_y[ind]
    class_affinity = np.zeros((len(x), n_classes), dtype=np.float64)
    for cls in range(n_classes):
        class_affinity[:, cls] = (weights * (neigh_y == cls)).sum(axis=1)
    neigh_mean = np.einsum("ij,ijd->id", weights, source_x[ind])
    smooth_residual = x - neigh_mean
    proto_dist = np.linalg.norm(x[:, None, :] - prototypes[None, :, :], axis=2)
    proto_affinity = np.exp(-proto_dist / np.median(proto_dist).clip(min=1e-8))
    proto_affinity = proto_affinity / proto_affinity.sum(axis=1, keepdims=True).clip(min=1e-12)
    entropy = -(class_affinity * np.log(class_affinity.clip(min=1e-12))).sum(axis=1, keepdims=True) / np.log(n_classes)
    degree_support = 1.0 / (1.0 + dist.mean(axis=1, keepdims=True))
    return np.hstack(
        [
            class_affinity,
            proto_affinity,
            proto_dist.min(axis=1, keepdims=True),
            proto_dist.mean(axis=1, keepdims=True),
            dist.mean(axis=1, keepdims=True),
            dist.std(axis=1, keepdims=True),
            entropy,
            degree_support,
            smooth_residual[:, : min(args.residual_dims, smooth_residual.shape[1])],
        ]
    )


class GraphFeatureForest:
    def __init__(self, seed: int, args):
        self.seed = seed
        self.args = args
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
            random_state=seed + 17,
            n_jobs=args.n_jobs,
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.classes_ = np.array(sorted(np.unique(y)))
        self.rf.fit(x, y)
        self.et.fit(x, y)
        return self

    def _aligned(self, estimator, x: np.ndarray) -> np.ndarray:
        raw = estimator.predict_proba(x)
        out = np.zeros((x.shape[0], len(self.classes_)), dtype=np.float64)
        for j, label in enumerate(estimator.classes_):
            out[:, int(np.where(self.classes_ == label)[0][0])] = raw[:, j]
        return out

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return normalized_proba(0.5 * self._aligned(self.rf, x) + 0.5 * self._aligned(self.et, x))


def global_proba(model: GraphFeatureForest, x: np.ndarray, n_classes: int) -> np.ndarray:
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
    train_mask = df["WELL"].isin(train_wells).to_numpy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_all = scaler.fit_transform(imputer.fit_transform(df[features]))
    y_all = df["TARGET"].to_numpy(dtype=np.int64)
    source_x = x_all[train_mask]
    source_y = y_all[train_mask]
    n_classes = len(class_names)
    prototypes = class_prototypes(source_x, source_y, n_classes)
    g_all = graph_features(x_all, source_x, source_y, prototypes, args)
    x_aug = np.hstack([x_all, g_all])
    start = time.time()
    model = GraphFeatureForest(seed, args).fit(x_aug[train_mask], source_y)
    train_time = time.time() - start
    rows = []
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        test = df[df["WELL"].isin(wells)].copy()
        ordered = test.sort_values(["WELL", "DEPTH_MD"])
        pos = ordered.index.to_numpy(dtype=np.int64)
        proba = global_proba(model, x_aug[pos], n_classes)
        y = ordered["TARGET"].to_numpy(dtype=np.int64)
        rows.extend(selective_rows(y, proba, seed, split, "graph_feature_margin", args.coverages))
        full = metrics(y, proba.argmax(axis=1))
        full.update({"seed": seed, "split": split, "method": "graph_feature_full", "coverage": 1.0, "kept_rows": len(y)})
        rows.append(full)
    for row in rows:
        row.update({"n_features": x_aug.shape[1], "knn": args.knn, "train_time": train_time})
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
    cand_all = raw[(raw["split"] == "extrapolation") & (raw["method"] == "graph_feature_margin")]
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
                        "method": "graph_feature_margin",
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
    x = rng.normal(size=(50, 6))
    y = np.array([0] * 20 + [1] * 18 + [2] * 12)
    proto = class_prototypes(x[:35], y[:35], 3)
    g = graph_features(x, x[:35], y[:35], proto, args)
    model = GraphFeatureForest(0, args).fit(np.hstack([x[:35], g[:35]]), y[:35])
    proba = global_proba(model, np.hstack([x[35:], g[35:]]), 3)
    rows = selective_rows(y[35:], proba, 0, "self_check", "graph_feature_margin", [0.4])
    if g.shape[0] != 50 or proba.shape != (15, 3) or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Graph-feature probabilities are invalid.")
    if rows[0]["kept_rows"] != 6:
        raise RuntimeError("Graph-feature selective self-check failed.")
    print("graph_feature_extraction_baseline self-check passed")


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
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--residual-dims", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=45)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--out-csv", type=Path, default=Path("results/recent_graph_feature_force_11seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/recent_graph_feature_force_11seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/recent_graph_feature_force_11seed_paired.csv"))
    parser.add_argument("--reference-summary", type=Path, default=Path("results/force_release_seed_level_reference.csv"))
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
