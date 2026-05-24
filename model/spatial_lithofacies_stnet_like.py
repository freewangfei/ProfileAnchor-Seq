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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data.spatial_multimethod_group_benchmark import build_features, load_force, sample_by_well, split_wells_by_space


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "MCC": matthews_corrcoef(y_true, pred),
        "F1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
    }


class WindowDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        seq_matrix: np.ndarray,
        point_matrix: np.ndarray,
        labels: np.ndarray | None,
        window: int,
    ) -> None:
        self.seq_matrix = seq_matrix.astype(np.float32, copy=False)
        self.point_matrix = point_matrix.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False) if labels is not None else None
        self.window = window
        self.half = window // 2
        self.samples: list[tuple[np.ndarray, int]] = []
        for _, group in frame.groupby("WELL", sort=False):
            ordered = group.sort_values("DEPTH_MD")
            positions = ordered.index.to_numpy(dtype=np.int64)
            for local_pos, center in enumerate(positions):
                self.samples.append((positions, local_pos))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        positions, local_pos = self.samples[idx]
        center = positions[local_pos]
        start = max(0, local_pos - self.half)
        end = min(len(positions), local_pos + self.half + 1)
        take = positions[start:end]
        seq = np.zeros((self.window, self.seq_matrix.shape[1]), dtype=np.float32)
        insert_at = self.half - (local_pos - start)
        seq[insert_at : insert_at + len(take)] = self.seq_matrix[take]
        point = self.point_matrix[center]
        if self.labels is None:
            return torch.from_numpy(seq), torch.from_numpy(point), center
        return torch.from_numpy(seq), torch.from_numpy(point), torch.tensor(self.labels[center], dtype=torch.long)


class STNetLike(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.spatial = nn.Sequential(
            nn.Linear(point_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, seq: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        x = seq.transpose(1, 2)
        x = self.temporal(x).transpose(1, 2)
        x, _ = self.gru(x)
        temporal = x.mean(dim=1)
        spatial = self.spatial(point)
        return self.head(torch.cat([temporal, spatial], dim=1))


def class_weights(labels: np.ndarray, n_classes: int, power: float = 0.5) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (counts.sum() / (n_classes * counts)) ** float(power)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class WeightedFocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor, gamma: float) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        row = torch.arange(target.shape[0], device=target.device)
        pt = p[row, target].clamp_min(1e-8)
        loss = -self.weight[target] * ((1.0 - pt) ** self.gamma) * logp[row, target]
        return loss.mean()


def balanced_sampler(dataset: WindowDataset, labels: np.ndarray, n_classes: int, power: float, seed: int) -> WeightedRandomSampler:
    weights = class_weights(labels, n_classes, power=float(power)).numpy()
    sample_weights = np.array([weights[int(labels[center])] for _, center in dataset.samples], dtype=np.float64)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.Series:
    model.eval()
    rows = []
    with torch.no_grad():
        for seq, point, row_index in loader:
            logits = model(seq.to(device), point.to(device))
            batch_pred = logits.argmax(dim=1).cpu().numpy()
            rows.append(pd.Series(batch_pred, index=row_index.numpy()))
    if not rows:
        return pd.Series(dtype=np.int64)
    return pd.concat(rows).astype(np.int64)


def run_seed(seed: int, args) -> list[dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, all_features = build_features(df, include_missing=True)
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    train_wells, interp_wells, extra_wells, y_cut = split_wells_by_space(
        df, args.train_fraction, args.interp_test_wells, seed
    )
    train_mask = df["WELL"].isin(train_wells).to_numpy()
    n_classes = len(class_names)

    seq_features = [
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
    point_features = all_features
    seq_imputer = SimpleImputer(strategy="median")
    point_imputer = SimpleImputer(strategy="median")
    seq_scaler = StandardScaler()
    point_scaler = StandardScaler()
    seq_train = seq_scaler.fit_transform(seq_imputer.fit_transform(df.loc[train_mask, seq_features]))
    point_train = point_scaler.fit_transform(point_imputer.fit_transform(df.loc[train_mask, point_features]))
    seq_all = seq_scaler.transform(seq_imputer.transform(df[seq_features]))
    point_all = point_scaler.transform(point_imputer.transform(df[point_features]))
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
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
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
    train_time = time.time() - start

    rows = []
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        test_frame = df[df["WELL"].isin(wells)].copy()
        test_dataset = WindowDataset(test_frame, seq_all, point_all, None, args.window)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        pred_series = predict(model, test_loader, device)
        ordered = test_frame.sort_values(["WELL", "DEPTH_MD"])
        pred = pred_series.loc[ordered.index.to_numpy()].to_numpy(dtype=np.int64)
        y = ordered["TARGET"].to_numpy(dtype=np.int64)
        row = metric_row(y, pred)
        row.update(
            {
                "seed": seed,
                "target": args.target,
                "model": "stnet_like",
                "split": split,
                "window": args.window,
                "epochs": args.epochs,
                "hidden": args.hidden,
                "train_rows": int(train_mask.sum()),
                "test_rows": len(test_frame),
                "train_wells": len(train_wells),
                "test_wells": len(wells),
                "Training Time": train_time,
                "y_cut": y_cut,
            }
        )
        rows.append(row)
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted", "Training Time"]
    summary = raw.groupby(["target", "model", "split"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def paired_against_rf(raw: pd.DataFrame, rf_csv: Path) -> pd.DataFrame:
    if not rf_csv.exists():
        return pd.DataFrame()
    rf = pd.read_csv(rf_csv)
    rf = rf[(rf["method"] == "base_smote") & (rf["split"].isin(["interpolation", "extrapolation"]))]
    rows = []
    for split in ["interpolation", "extrapolation"]:
        base = rf[rf["split"] == split].set_index("seed")
        cand = raw[raw["split"] == split].set_index("seed")
        common = sorted(set(base.index) & set(cand.index))
        for metric in ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted"]:
            b = base.loc[common, metric]
            c = cand.loc[common, metric]
            diff = c - b
            rows.append(
                {
                    "split": split,
                    "baseline": "rf_smote_spatial_view",
                    "method": "stnet_like",
                    "metric": metric,
                    "n": len(common),
                    "baseline_mean": float(b.mean()) if common else np.nan,
                    "method_mean": float(c.mean()) if common else np.nan,
                    "delta_mean": float(diff.mean()) if common else np.nan,
                    "wins": int((diff > 0).sum()),
                    "ties": int((diff == 0).sum()),
                    "losses": int((diff < 0).sum()),
                    "paired_t_p": float(ttest_rel(c, b).pvalue) if len(common) > 1 else np.nan,
                    "wilcoxon_p": float(wilcoxon(diff).pvalue) if len(common) > 1 and (diff != 0).any() else np.nan,
                    "deltas": ";".join(f"{v:.6f}" for v in diff),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/force2020"))
    parser.add_argument("--target", default="FORCE_2020_LITHOFACIES_LITHOLOGY")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 42])
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--interp-test-wells", type=int, default=10)
    parser.add_argument("--max-rows-per-well", type=int, default=800)
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stnet-class-weight-power", type=float, default=0.5)
    parser.add_argument("--stnet-focal-gamma", type=float, default=0.0)
    parser.add_argument("--stnet-balanced-sampler", action="store_true")
    parser.add_argument("--stnet-sampler-power", type=float, default=0.5)
    parser.add_argument("--out-csv", type=Path, default=Path("results/spatial_lithofacies_stnet_like_5seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/spatial_lithofacies_stnet_like_5seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/spatial_lithofacies_stnet_like_5seed_vs_rf_smote_paired.csv"))
    parser.add_argument("--rf-csv", type=Path, default=Path("results/spatial_lithofacies_feature_view_fusion_restricted_select_11seed.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

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
    paired = paired_against_rf(raw, args.rf_csv)
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
