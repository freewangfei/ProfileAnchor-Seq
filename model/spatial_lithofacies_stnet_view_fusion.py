import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from util.spatial_lithofacies_feature_ablation_smote import select_features
from util.spatial_lithofacies_feature_view_fusion import y_band_fit_calibration
from model.spatial_lithofacies_stnet_like import STNetLike, WeightedFocalLoss, WindowDataset, balanced_sampler, class_weights
from data.spatial_multimethod_group_benchmark import build_features, load_force, sample_by_well, split_wells_by_space


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "MCC": matthews_corrcoef(y_true, pred),
        "F1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
    }


def sequence_features(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in [
            "DEPTH_MD",
            "GR",
            "RHOB",
            "NPHI",
            "DTC",
            "GR_WELL_Z",
            "RHOB_WELL_Z",
            "NPHI_WELL_Z",
            "DTC_WELL_Z",
            "GR_MISSING",
            "RHOB_MISSING",
            "NPHI_MISSING",
            "DTC_MISSING",
        ]
        if col in df.columns
    ]


def make_matrices(df: pd.DataFrame, train_mask: np.ndarray, seq_features: list[str], point_features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    seq_imputer = SimpleImputer(strategy="median")
    point_imputer = SimpleImputer(strategy="median")
    seq_scaler = StandardScaler()
    point_scaler = StandardScaler()
    seq_scaler.fit(seq_imputer.fit_transform(df.loc[train_mask, seq_features]))
    point_scaler.fit(point_imputer.fit_transform(df.loc[train_mask, point_features]))
    seq_all = seq_scaler.transform(seq_imputer.transform(df[seq_features]))
    point_all = point_scaler.transform(point_imputer.transform(df[point_features]))
    return seq_all, point_all


def fit_view(
    df: pd.DataFrame,
    train_wells: set[str],
    point_features: list[str],
    seed: int,
    n_classes: int,
    args,
) -> dict:
    train_mask = df["WELL"].isin(train_wells).to_numpy()
    seq_features = sequence_features(df)
    seq_all, point_all = make_matrices(df, train_mask, seq_features, point_features)
    labels = df["TARGET"].to_numpy(dtype=np.int64)
    train_frame = df[df["WELL"].isin(train_wells)].copy()
    train_dataset = WindowDataset(train_frame, seq_all, point_all, labels, args.window)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    shuffle = True
    if getattr(args, "stnet_balanced_sampler", False):
        sampler = balanced_sampler(
            train_dataset,
            labels,
            n_classes,
            getattr(args, "stnet_sampler_power", getattr(args, "stnet_class_weight_power", 0.5)),
            seed,
        )
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        generator=generator if sampler is None else None,
    )
    device = torch.device(args.device)
    model = STNetLike(len(seq_features), len(point_features), n_classes, args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weight = class_weights(labels[train_mask], n_classes, getattr(args, "stnet_class_weight_power", 0.5)).to(device)
    if getattr(args, "stnet_focal_gamma", 0.0) > 0:
        criterion = WeightedFocalLoss(weight, getattr(args, "stnet_focal_gamma", 0.0))
    else:
        criterion = nn.CrossEntropyLoss(weight=weight)
    start = time.time()
    model.train()
    for _ in range(args.epochs):
        for seq, point, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(seq.to(device), point.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
    return {
        "model": model,
        "seq_all": seq_all,
        "point_all": point_all,
        "point_features": point_features,
        "Training Time": time.time() - start,
    }


def proba_view(fitted: dict, frame: pd.DataFrame, args) -> pd.Series:
    model = fitted["model"]
    model.eval()
    dataset = WindowDataset(frame, fitted["seq_all"], fitted["point_all"], None, args.window)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    device = torch.device(args.device)
    with torch.no_grad():
        for seq, point, row_index in loader:
            logits = model(seq.to(device), point.to(device))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            block = pd.DataFrame(prob, index=row_index.numpy())
            rows.append(block)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows).sort_index()


def tune_alpha(df: pd.DataFrame, train_wells: set[str], all_features: list[str], seed: int, n_classes: int, args) -> tuple[float, list[dict]]:
    train = df[df["WELL"].isin(train_wells)].copy()
    fit, cal = y_band_fit_calibration(train, args.cal_fraction)
    fit_wells = set(fit["WELL"].unique())
    base = fit_view(df, fit_wells, select_features(all_features, "base"), seed, n_classes, args)
    noxyz = fit_view(df, fit_wells, select_features(all_features, "no_xyz"), seed, n_classes, args)
    ordered = cal.sort_values(["WELL", "DEPTH_MD"])
    p_base = proba_view(base, ordered, args).loc[ordered.index.to_numpy()].to_numpy()
    p_noxyz = proba_view(noxyz, ordered, args).loc[ordered.index.to_numpy()].to_numpy()
    y = ordered["TARGET"].to_numpy(dtype=np.int64)
    trace = []
    best_alpha = 1.0
    best_score = -np.inf
    for alpha in args.alpha_grid:
        pred = (float(alpha) * p_base + (1.0 - float(alpha)) * p_noxyz).argmax(axis=1)
        row = metric_row(y, pred)
        score = row["F1_weighted"] + args.macro_weight * row["F1_macro"] + args.ba_weight * row["Balanced Accuracy"]
        trace.append({"seed": seed, "alpha": float(alpha), "selection_score": score, **row})
        if score > best_score + 1e-12:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, trace


def run_seed(seed: int, args) -> tuple[list[dict], list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, all_features = build_features(df, include_missing=True)
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    train_wells, interp_wells, extra_wells, y_cut = split_wells_by_space(df, args.train_fraction, args.interp_test_wells, seed)
    train_wells = set(train_wells)
    n_classes = len(class_names)
    selected_alpha, trace = tune_alpha(df, train_wells, all_features, seed, n_classes, args) if not args.skip_inner else (args.fixed_alpha, [])
    base = fit_view(df, train_wells, select_features(all_features, "base"), seed, n_classes, args)
    noxyz = fit_view(df, train_wells, select_features(all_features, "no_xyz"), seed, n_classes, args)
    variants = {
        "stnet_base": 1.0,
        "stnet_noxyz": 0.0,
        f"stnet_view_fusion_a{int(args.fixed_alpha * 100):03d}": args.fixed_alpha,
        "stnet_inner_selected_view_fusion": selected_alpha,
    }
    rows = []
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        ordered = df[df["WELL"].isin(wells)].copy().sort_values(["WELL", "DEPTH_MD"])
        y = ordered["TARGET"].to_numpy(dtype=np.int64)
        p_base = proba_view(base, ordered, args).loc[ordered.index.to_numpy()].to_numpy()
        p_noxyz = proba_view(noxyz, ordered, args).loc[ordered.index.to_numpy()].to_numpy()
        for method, alpha in variants.items():
            pred = (float(alpha) * p_base + (1.0 - float(alpha)) * p_noxyz).argmax(axis=1)
            row = metric_row(y, pred)
            row.update(
                {
                    "seed": seed,
                    "target": args.target,
                    "model": "stnet_like",
                    "method": method,
                    "split": split,
                    "alpha": float(alpha),
                    "selected_alpha": float(selected_alpha),
                    "window": args.window,
                    "epochs": args.epochs,
                    "hidden": args.hidden,
                    "train_rows": int(df["WELL"].isin(train_wells).sum()),
                    "test_rows": len(ordered),
                    "train_wells": len(train_wells),
                    "test_wells": len(wells),
                    "Training Time": base["Training Time"] + noxyz["Training Time"],
                    "y_cut": y_cut,
                }
            )
            rows.append(row)
    return rows, trace


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted", "Training Time"]
    summary = raw.groupby(["target", "model", "method", "split"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def paired_stats(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ["interpolation", "extrapolation"]:
        subset = raw[raw["split"] == split]
        base = subset[subset["method"] == "stnet_base"].set_index("seed")
        for method in sorted(m for m in subset["method"].unique() if m != "stnet_base"):
            cand = subset[subset["method"] == method].set_index("seed").reindex(base.index)
            for metric in ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted"]:
                b = base[metric].dropna()
                c = cand[metric].reindex(b.index).dropna()
                b = b.reindex(c.index)
                diff = c.to_numpy() - b.to_numpy()
                rows.append(
                    {
                        "split": split,
                        "baseline": "stnet_base",
                        "method": method,
                        "metric": metric,
                        "n": len(diff),
                        "baseline_mean": float(b.mean()),
                        "method_mean": float(c.mean()),
                        "delta_mean": float(diff.mean()),
                        "wins": int((diff > 0).sum()),
                        "ties": int((diff == 0).sum()),
                        "losses": int((diff < 0).sum()),
                        "paired_t_p": float(ttest_rel(c, b).pvalue) if len(diff) > 1 else np.nan,
                        "wilcoxon_p": float(wilcoxon(diff).pvalue) if len(diff) > 1 and (diff != 0).any() else np.nan,
                        "deltas": ";".join(f"{d:.6f}" for d in diff),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/force2020"))
    parser.add_argument("--target", default="FORCE_2020_LITHOFACIES_LITHOLOGY")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--interp-test-wells", type=int, default=10)
    parser.add_argument("--max-rows-per-well", type=int, default=800)
    parser.add_argument("--cal-fraction", type=float, default=0.25)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.75, 0.90, 1.0])
    parser.add_argument("--fixed-alpha", type=float, default=0.75)
    parser.add_argument("--macro-weight", type=float, default=0.20)
    parser.add_argument("--ba-weight", type=float, default=0.10)
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stnet-class-weight-power", type=float, default=0.5)
    parser.add_argument("--stnet-focal-gamma", type=float, default=0.0)
    parser.add_argument("--stnet-balanced-sampler", action="store_true")
    parser.add_argument("--stnet-sampler-power", type=float, default=0.5)
    parser.add_argument("--skip-inner", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=Path("results/spatial_lithofacies_stnet_view_fusion_smoke.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/spatial_lithofacies_stnet_view_fusion_smoke_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/spatial_lithofacies_stnet_view_fusion_smoke_paired.csv"))
    parser.add_argument("--trace-csv", type=Path, default=Path("results/spatial_lithofacies_stnet_view_fusion_smoke_trace.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = []
    trace_rows = []
    done = set()
    if args.resume and args.out_csv.exists():
        existing = pd.read_csv(args.out_csv)
        rows = existing.to_dict("records")
        done = set(existing["seed"].unique())
        if args.trace_csv.exists():
            trace_rows = pd.read_csv(args.trace_csv).to_dict("records")
    for seed in args.seeds:
        if seed in done:
            continue
        seed_rows, seed_trace = run_seed(seed, args)
        rows.extend(seed_rows)
        trace_rows.extend(seed_trace)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        pd.DataFrame(trace_rows).to_csv(args.trace_csv, index=False)
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    paired = paired_stats(raw)
    summary.to_csv(args.summary_csv, index=False)
    paired.to_csv(args.paired_csv, index=False)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.paired_csv}")
    print(f"Wrote {args.trace_csv}")


if __name__ == "__main__":
    main()
