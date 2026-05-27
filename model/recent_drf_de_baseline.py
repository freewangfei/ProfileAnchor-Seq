import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
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


def source_feature_selection(x: np.ndarray, y: np.ndarray, seed: int, args) -> np.ndarray:
    if args.mi_top_k <= 0 or args.mi_top_k >= x.shape[1]:
        return np.arange(x.shape[1], dtype=int)
    scores = mutual_info_classif(x, y, discrete_features=False, random_state=seed)
    order = np.argsort(-np.nan_to_num(scores, nan=0.0))
    return np.sort(order[: args.mi_top_k])


def adaptive_interpolation(x: np.ndarray, y: np.ndarray, seed: int, args) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    target = int(np.ceil(np.quantile(counts, args.balance_quantile)))
    max_new = max(0, int(len(y) * args.max_augmented_multiplier) - len(y))
    xs = [x]
    ys = [y]
    made = 0
    for label, count in zip(labels, counts):
        need = min(max(0, target - int(count)), max_new - made)
        idx = np.flatnonzero(y == label)
        if need <= 0 or len(idx) < 2:
            continue
        anchors = rng.choice(idx, size=need, replace=True)
        neighbours = []
        for anchor in anchors:
            candidates = idx[idx != anchor]
            neighbours.append(rng.choice(candidates))
        neighbours = np.asarray(neighbours, dtype=int)
        lam = rng.random((need, 1))
        xs.append(x[anchors] + lam * (x[neighbours] - x[anchors]))
        ys.append(np.full(need, label, dtype=y.dtype))
        made += need
        if made >= max_new:
            break
    return np.vstack(xs), np.concatenate(ys), made


def tree_purity(estimator, x: np.ndarray) -> float:
    leaves = estimator.apply(x)
    tree = estimator.tree_
    unique, counts = np.unique(leaves, return_counts=True)
    impurity = tree.impurity[unique]
    weights = counts / counts.sum()
    return float(1.0 - np.sum(weights * impurity))


class PrunedRandomForest:
    def __init__(self, seed: int, args, params: dict):
        self.seed = seed
        self.args = args
        self.params = params

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.classes_ = np.array(sorted(np.unique(y)))
        self.model_ = RandomForestClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            min_samples_leaf=self.params["min_samples_leaf"],
            max_features=self.params["max_features"],
            class_weight="balanced_subsample",
            bootstrap=True,
            random_state=self.seed,
            n_jobs=self.args.n_jobs,
        )
        self.model_.fit(x, y)
        purities = np.array([tree_purity(est, x) for est in self.model_.estimators_], dtype=np.float64)
        keep = np.flatnonzero(purities >= self.params["purity_threshold"])
        if len(keep) < max(3, int(0.15 * len(purities))):
            keep = np.argsort(-purities)[: max(3, int(0.35 * len(purities)))]
        self.estimators_ = [self.model_.estimators_[int(i)] for i in keep]
        self.retained_trees_ = len(self.estimators_)
        self.mean_tree_purity_ = float(purities[keep].mean())
        return self

    def _aligned(self, estimator, x: np.ndarray) -> np.ndarray:
        raw = estimator.predict_proba(x)
        out = np.zeros((x.shape[0], len(self.classes_)), dtype=np.float64)
        for j, label in enumerate(estimator.classes_):
            matches = np.where(self.classes_ == label)[0]
            if len(matches):
                pos = int(matches[0])
            else:
                local = int(label)
                if local < 0 or local >= len(self.model_.classes_):
                    continue
                actual = self.model_.classes_[local]
                matches = np.where(self.classes_ == actual)[0]
                if not len(matches):
                    continue
                pos = int(matches[0])
            out[:, pos] = raw[:, j]
        return out

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        proba = np.zeros((x.shape[0], len(self.classes_)), dtype=np.float64)
        for estimator in self.estimators_:
            proba += self._aligned(estimator, x)
        return normalized_proba(proba / max(1, len(self.estimators_)))


def decode_params(vector: np.ndarray) -> dict:
    return {
        "n_estimators": int(round(vector[0])),
        "max_depth": int(round(vector[1])),
        "min_samples_leaf": int(round(vector[2])),
        "max_features": float(vector[3]),
        "purity_threshold": float(vector[4]),
    }


def fit_drf_de(x_source: np.ndarray, y_source: np.ndarray, well_source: np.ndarray, seed: int, args):
    rng = np.random.default_rng(seed)
    wells = np.array(sorted(np.unique(well_source)))
    n_val = min(max(1, args.inner_validation_wells), max(1, len(wells) - 1))
    val_wells = set(rng.choice(wells, size=n_val, replace=False))
    val_mask = np.array([well in val_wells for well in well_source])
    if val_mask.all() or not val_mask.any():
        val_mask = np.zeros(len(y_source), dtype=bool)
        val_mask[rng.choice(np.arange(len(y_source)), size=max(1, len(y_source) // 5), replace=False)] = True
    x_inner, y_inner = x_source[~val_mask], y_source[~val_mask]
    x_val, y_val = x_source[val_mask], y_source[val_mask]

    def objective(vector):
        params = decode_params(vector)
        try:
            model = PrunedRandomForest(seed, args, params).fit(x_inner, y_inner)
            proba = model.predict_proba(x_val)
            pred = model.classes_[proba.argmax(axis=1)]
            score = 0.5 * balanced_accuracy_score(y_val, pred) + 0.5 * f1_score(y_val, pred, average="macro", zero_division=0)
            return -float(score)
        except Exception:
            return 1.0

    bounds = [
        (args.min_estimators, args.max_estimators),
        (args.min_depth, args.max_depth),
        (args.min_leaf, args.max_leaf),
        (args.min_features, args.max_features),
        (args.min_purity, args.max_purity),
    ]
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=args.de_max_iter,
        popsize=args.de_popsize,
        polish=False,
        workers=1,
        updating="immediate",
        tol=args.de_tol,
    )
    params = decode_params(result.x)
    model = PrunedRandomForest(seed, args, params).fit(x_source, y_source)
    return model, params, float(-result.fun)


def global_proba(model: PrunedRandomForest, x: np.ndarray, n_classes: int) -> np.ndarray:
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
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_all = scaler.fit_transform(imputer.fit_transform(train[features]))
    y_train = train["TARGET"].to_numpy(dtype=np.int64)
    selected = source_feature_selection(x_train_all, y_train, seed, args)
    x_train = x_train_all[:, selected]
    x_aug, y_aug, synthetic_rows = adaptive_interpolation(x_train, y_train, seed, args)
    source_wells = train["WELL"].to_numpy()
    if len(x_aug) > len(x_train):
        source_wells = np.concatenate([source_wells, np.repeat(source_wells[0], len(x_aug) - len(x_train))])
    start = time.time()
    model, params, inner_score = fit_drf_de(x_aug, y_aug, source_wells, seed, args)
    train_time = time.time() - start
    rows = []
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        test = df[df["WELL"].isin(wells)].copy()
        ordered = test.sort_values(["WELL", "DEPTH_MD"])
        x_test = scaler.transform(imputer.transform(ordered[features]))[:, selected]
        proba = global_proba(model, x_test, len(class_names))
        y = ordered["TARGET"].to_numpy(dtype=np.int64)
        rows.extend(selective_rows(y, proba, seed, split, "drf_de_margin", args.coverages))
        full = metrics(y, proba.argmax(axis=1))
        full.update({"seed": seed, "split": split, "method": "drf_de_full", "coverage": 1.0, "kept_rows": len(y)})
        rows.append(full)
    for row in rows:
        row.update(
            {
                "selected_features": len(selected),
                "synthetic_rows": synthetic_rows,
                "retained_trees": model.retained_trees_,
                "mean_tree_purity": model.mean_tree_purity_,
                "inner_score": inner_score,
                "train_time": train_time,
                **params,
            }
        )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted", "kept_rows", "retained_trees", "synthetic_rows", "train_time"]
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
    cand_all = raw[(raw["split"] == "extrapolation") & (raw["method"] == "drf_de_margin")]
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
                        "method": "drf_de_margin",
                        "baseline": baseline,
                        "coverage": coverage,
                        "metric": metric,
                        "n": len(common),
                        "mean_diff": float(diff.mean()),
                        "method_mean": float(cv.mean()),
                        "baseline_mean": float(bv.mean()),
                        "paired_t_p": float(ttest_rel(cv, bv).pvalue) if len(common) > 1 else np.nan,
                        "wilcoxon_p": float(wilcoxon(diff).pvalue) if len(common) > 1 and np.any(diff != 0) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def self_check():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 10))
    y = rng.integers(0, 4, size=80)
    wells = np.repeat(np.array(["A", "B", "C", "D"]), 20)
    args = argparse.Namespace(
        n_jobs=1,
        inner_validation_wells=1,
        min_estimators=12,
        max_estimators=18,
        min_depth=3,
        max_depth=6,
        min_leaf=1,
        max_leaf=3,
        min_features=0.4,
        max_features=0.9,
        min_purity=0.2,
        max_purity=0.9,
        de_max_iter=1,
        de_popsize=3,
        de_tol=0.05,
    )
    model, _, _ = fit_drf_de(x, y, wells, 0, args)
    proba = global_proba(model, x[:10], 4)
    rows = selective_rows(y[:10], proba, 0, "self_check", "drf_de_margin", [0.5])
    assert proba.shape == (10, 4)
    assert rows and np.isfinite(rows[0]["Accuracy"])
    print("recent_drf_de_baseline self-check passed")


def parse_args():
    parser = argparse.ArgumentParser(description="DRF-DE-style lithology baseline under FORCE complete-well release protocol.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/force2020"))
    parser.add_argument("--target", default="FORCE_2020_LITHOFACIES_LITHOLOGY")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42])
    parser.add_argument("--coverages", nargs="+", type=float, default=DEFAULT_COVERAGES)
    parser.add_argument("--max-rows-per-well", type=int, default=800)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--interp-test-wells", type=int, default=10)
    parser.add_argument("--mi-top-k", type=int, default=18)
    parser.add_argument("--balance-quantile", type=float, default=0.70)
    parser.add_argument("--max-augmented-multiplier", type=float, default=1.35)
    parser.add_argument("--inner-validation-wells", type=int, default=3)
    parser.add_argument("--min-estimators", type=int, default=80)
    parser.add_argument("--max-estimators", type=int, default=180)
    parser.add_argument("--min-depth", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-leaf", type=int, default=1)
    parser.add_argument("--max-leaf", type=int, default=8)
    parser.add_argument("--min-features", type=float, default=0.35)
    parser.add_argument("--max-features", type=float, default=0.95)
    parser.add_argument("--min-purity", type=float, default=0.20)
    parser.add_argument("--max-purity", type=float, default=0.90)
    parser.add_argument("--de-max-iter", type=int, default=3)
    parser.add_argument("--de-popsize", type=int, default=4)
    parser.add_argument("--de-tol", type=float, default=0.01)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--baseline-csv", type=Path, default=Path("results/force_release_seed_level_reference.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("results/recent_drf_de_force_11seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/recent_drf_de_force_11seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/recent_drf_de_force_11seed_paired.csv"))
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_check:
        self_check()
        return
    all_rows = []
    for seed in args.seeds:
        all_rows.extend(run_seed(seed, args))
    raw = pd.DataFrame(all_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_csv, index=False)
    summarize(raw).to_csv(args.summary_csv, index=False)
    paired = paired_stats(raw, args.baseline_csv)
    if not paired.empty:
        paired.to_csv(args.paired_csv, index=False)
    print(raw.groupby(["method", "split", "coverage"])[["Accuracy", "F1_weighted", "Balanced Accuracy", "F1_macro", "MCC"]].mean())


if __name__ == "__main__":
    main()
