
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "MCC": matthews_corrcoef(y_true, pred),
        "F1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
    }


def margin(proba: np.ndarray) -> np.ndarray:
    if proba.shape[1] == 1:
        return np.ones(proba.shape[0], dtype=float)
    part = np.partition(proba, -2, axis=1)
    return part[:, -1] - part[:, -2]


def selective_rows(y: np.ndarray, proba: np.ndarray, seed: int, method: str, coverages: list[float]) -> list[dict]:
    pred = proba.argmax(axis=1)
    order = np.argsort(margin(proba))[::-1]
    rows = []
    for coverage in coverages:
        keep_n = max(1, int(round(len(y) * coverage)))
        keep = order[:keep_n]
        row = metric_row(y[keep], pred[keep])
        row.update({"seed": seed, "split": "external", "method": method, "coverage": coverage, "kept_rows": keep_n})
        rows.append(row)
    full = metric_row(y, pred)
    full.update({"seed": seed, "split": "external", "method": method, "coverage": 1.0, "kept_rows": len(y)})
    rows.append(full)
    return rows


def add_well_zscore(df: pd.DataFrame, logs: list[str], well_col: str) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby(well_col, sort=False)
    for feature in logs:
        mean = grouped[feature].transform("mean")
        std = grouped[feature].transform("std").replace(0, np.nan)
        out[f"{feature}_WELL_Z"] = (out[feature] - mean) / std
    return out


def load_data(args) -> tuple[pd.DataFrame, list[str], list[str], int]:
    df = pd.read_csv(args.csv)
    df = add_well_zscore(df, list(args.logs), args.well_col)
    encoder = LabelEncoder()
    df["TARGET"] = encoder.fit_transform(df[args.label_col].astype(str))
    seq_features = [args.depth_col] + list(args.logs) + [f"{c}_WELL_Z" for c in args.logs]
    point_features = list(seq_features)
    for feature in list(point_features):
        missing = f"{feature}_missing"
        df[missing] = df[feature].isna().astype(np.float32)
        point_features.append(missing)
    for feature in list(seq_features):
        missing = f"{feature}_missing"
        if missing in df.columns:
            seq_features.append(missing)
    df = df.sort_values([args.well_col, args.depth_col]).reset_index(drop=True)
    return df, seq_features, point_features, int(df["TARGET"].max() + 1)


def class_weights(labels: np.ndarray, n_classes: int, power: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (counts.sum() / (n_classes * counts)) ** float(power)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class WindowDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, seq_matrix: np.ndarray, point_matrix: np.ndarray, labels: np.ndarray | None, window: int, well_col: str, depth_col: str):
        self.seq_matrix = seq_matrix.astype(np.float32, copy=False)
        self.point_matrix = point_matrix.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False) if labels is not None else None
        self.window = window
        self.half = window // 2
        self.samples: list[tuple[np.ndarray, int]] = []
        for _, group in frame.groupby(well_col, sort=False):
            ordered = group.sort_values(depth_col)
            positions = ordered.index.to_numpy(dtype=np.int64)
            for local_pos, _ in enumerate(positions):
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
        self.spatial = nn.Sequential(nn.Linear(point_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, seq: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        x = self.temporal(seq.transpose(1, 2)).transpose(1, 2)
        x, _ = self.gru(x)
        temporal = x.mean(dim=1)
        spatial = self.spatial(point)
        return self.head(torch.cat([temporal, spatial], dim=1))


class GCNLike(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden)
        self.g1 = nn.Linear(hidden, hidden)
        self.g2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, row: torch.Tensor, col: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        h0 = torch.relu(self.input(x))
        h = graph_propagate(h0, row, col, deg)
        h = self.dropout(torch.relu(self.g1(h)) * 0.5 + h0 * 0.5)
        h2 = graph_propagate(h, row, col, deg)
        h = self.dropout(torch.relu(self.g2(h2)) * 0.5 + h * 0.5)
        return self.head(h)


def make_edges(frame: pd.DataFrame, well_col: str, depth_col: str) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[int] = []
    cols: list[int] = []
    for _, group in frame.groupby(well_col, sort=False):
        local = np.arange(len(group.sort_values(depth_col)), dtype=np.int64)
        offset = len(rows)
    rows = []
    cols = []
    node_offset = 0
    for _, group in frame.groupby(well_col, sort=False):
        n = len(group)
        idx = np.arange(node_offset, node_offset + n, dtype=np.int64)
        rows.extend(idx.tolist())
        cols.extend(idx.tolist())
        if n > 1:
            rows.extend(idx[:-1].tolist())
            cols.extend(idx[1:].tolist())
            rows.extend(idx[1:].tolist())
            cols.extend(idx[:-1].tolist())
        node_offset += n
    return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)


def normalized_degrees(row: torch.Tensor, n_nodes: int) -> torch.Tensor:
    deg = torch.bincount(row, minlength=n_nodes).float().clamp_min(1.0)
    return deg.pow(-0.5)


def graph_propagate(x: torch.Tensor, row: torch.Tensor, col: torch.Tensor, deg_inv_sqrt: torch.Tensor) -> torch.Tensor:
    weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    out = torch.zeros_like(x)
    out.index_add_(0, row, x[col] * weight.unsqueeze(1))
    return out


def fit_scalers(df: pd.DataFrame, train_mask: np.ndarray, seq_features: list[str], point_features: list[str]):
    seq_imputer = SimpleImputer(strategy="median")
    point_imputer = SimpleImputer(strategy="median")
    seq_scaler = StandardScaler()
    point_scaler = StandardScaler()
    seq_train = seq_imputer.fit_transform(df.loc[train_mask, seq_features])
    point_train = point_imputer.fit_transform(df.loc[train_mask, point_features])
    seq_scaler.fit(seq_train)
    point_scaler.fit(point_train)
    seq_all = seq_scaler.transform(seq_imputer.transform(df[seq_features]))
    point_all = point_scaler.transform(point_imputer.transform(df[point_features]))
    return seq_all, point_all


def predict_stnet(model: nn.Module, loader: DataLoader, n_classes: int, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for seq, point, row_index in loader:
            p = torch.softmax(model(seq.to(device), point.to(device)), dim=1).cpu().numpy()
            block = pd.DataFrame(p, index=row_index.numpy(), columns=[f"p{j}" for j in range(n_classes)])
            rows.append(block)
    return pd.concat(rows) if rows else pd.DataFrame(columns=[f"p{j}" for j in range(n_classes)])


def train_stnet(df: pd.DataFrame, seq_all: np.ndarray, point_all: np.ndarray, train_mask: np.ndarray, train_wells: set[str], n_classes: int, seed: int, args, labels: np.ndarray):
    train_frame = df[df[args.well_col].isin(train_wells)].copy()
    ds = WindowDataset(train_frame, seq_all, point_all, labels, args.window, args.well_col, args.depth_col)
    weights = class_weights(labels[train_mask], n_classes, args.class_weight_power).numpy()
    sample_weights = np.array([weights[int(labels[center])] for _, center in ds.samples], dtype=np.float64)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    device = torch.device(args.device)
    model = STNetLike(seq_all.shape[1], point_all.shape[1], n_classes, args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights(labels[train_mask], n_classes, args.class_weight_power).to(device))
    model.train()
    for _ in range(args.epochs):
        for seq, point, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(seq.to(device), point.to(device)), y.to(device))
            loss.backward()
            opt.step()
    return model


def train_gcn(point_all: np.ndarray, train_frame: pd.DataFrame, labels: np.ndarray, n_classes: int, seed: int, args):
    torch.manual_seed(seed)
    device = torch.device(args.device)
    pos = train_frame.index.to_numpy(dtype=np.int64)
    x = torch.tensor(point_all[pos], dtype=torch.float32).to(device)
    y = torch.tensor(labels[pos], dtype=torch.long).to(device)
    row, col = make_edges(train_frame, args.well_col, args.depth_col)
    deg = normalized_degrees(row, len(train_frame))
    row, col, deg = row.to(device), col.to(device), deg.to(device)
    model = GCNLike(point_all.shape[1], args.hidden, n_classes, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights(labels[pos], n_classes, args.class_weight_power).to(device))
    model.train()
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(x, row, col, deg), y)
        loss.backward()
        opt.step()
    return model


def gcn_predict(model: nn.Module, point_all: np.ndarray, test_frame: pd.DataFrame, args) -> np.ndarray:
    device = torch.device(args.device)
    pos = test_frame.index.to_numpy(dtype=np.int64)
    x = torch.tensor(point_all[pos], dtype=torch.float32).to(device)
    row, col = make_edges(test_frame, args.well_col, args.depth_col)
    deg = normalized_degrees(row, len(test_frame))
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x, row.to(device), col.to(device), deg.to(device)), dim=1).cpu().numpy()


def run_seed(df: pd.DataFrame, seq_features: list[str], point_features: list[str], n_classes: int, split: pd.Series, args) -> list[dict]:
    seed = int(split["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_wells = set(str(split["train_wells"]).split(","))
    test_wells = set(str(split["test_wells"]).split(","))
    train_mask = df[args.well_col].isin(train_wells).to_numpy()
    labels = df["TARGET"].to_numpy(dtype=np.int64)
    seq_all, point_all = fit_scalers(df, train_mask, seq_features, point_features)
    train_frame = df[df[args.well_col].isin(train_wells)].copy()
    test_frame = df[df[args.well_col].isin(test_wells)].copy()
    y = test_frame["TARGET"].to_numpy(dtype=np.int64)
    rows = []
    if "stnet" in args.models:
        model = train_stnet(df, seq_all, point_all, train_mask, train_wells, n_classes, seed, args, labels)
        ds = WindowDataset(test_frame, seq_all, point_all, None, args.window, args.well_col, args.depth_col)
        pred_df = predict_stnet(model, DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0), n_classes, torch.device(args.device))
        proba = pred_df.loc[test_frame.index.to_numpy()].to_numpy(dtype=float)
        rows.extend(selective_rows(y, proba, seed, "stnet_like_margin", args.coverages))
    if "gcn" in args.models:
        model = train_gcn(point_all, train_frame, labels, n_classes, seed, args)
        proba = gcn_predict(model, point_all, test_frame, args)
        rows.extend(selective_rows(y, proba, seed, "gcn_like_margin", args.coverages))
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted", "kept_rows"]
    grouped = raw.groupby(["method", "split", "coverage"], sort=True)[metrics]
    out = grouped.agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(c).rstrip("_") for c in out.columns.to_flat_index()]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--well-col", default="WELL")
    parser.add_argument("--depth-col", default="DEPTH_MD")
    parser.add_argument("--label-col", default="LITHOLOGY")
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["stnet", "gcn"])
    parser.add_argument("--coverages", nargs="+", type=float, default=[0.01, 0.03])
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-csv", type=Path, default=Path("figshare_structural_external_baselines_11seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("figshare_structural_external_baselines_11seed_summary.csv"))
    args = parser.parse_args()

    df, seq_features, point_features, n_classes = load_data(args)
    manifest = pd.read_csv(args.manifest)
    rows = []
    for _, split in manifest.iterrows():
        rows.extend(run_seed(df, seq_features, point_features, n_classes, split, args))
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    raw = pd.DataFrame(rows)
    summarize(raw).to_csv(args.summary_csv, index=False)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")


if __name__ == "__main__":
    main()
