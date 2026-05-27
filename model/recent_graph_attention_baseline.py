import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn

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


def class_weights(labels: np.ndarray, n_classes: int, power: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (counts.sum() / (n_classes * counts)) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_edges(frame: pd.DataFrame, x: np.ndarray, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[int] = []
    cols: list[int] = []
    offset = 0
    for _, group in frame.groupby("WELL", sort=False):
        ordered = group.sort_values("DEPTH_MD")
        idx = ordered.index.to_numpy(dtype=np.int64)
        n = len(idx)
        local = np.arange(offset, offset + n, dtype=np.int64)
        rows.extend(local.tolist())
        cols.extend(local.tolist())
        if n > 1:
            rows.extend(local[:-1].tolist())
            cols.extend(local[1:].tolist())
            rows.extend(local[1:].tolist())
            cols.extend(local[:-1].tolist())
        if n > k + 1:
            neigh = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean")
            neigh.fit(x[idx])
            nn_idx = neigh.kneighbors(return_distance=False)
            for src_local, nbrs in enumerate(nn_idx):
                src = offset + src_local
                for nbr in nbrs[1:]:
                    rows.append(src)
                    cols.append(offset + int(nbr))
        offset += n
    return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)


def edge_softmax(score: torch.Tensor, row: torch.Tensor, n_nodes: int) -> torch.Tensor:
    max_per_row = torch.full((n_nodes,), -torch.inf, device=score.device, dtype=score.dtype)
    max_per_row.scatter_reduce_(0, row, score, reduce="amax", include_self=True)
    exp = torch.exp(score - max_per_row[row])
    denom = torch.zeros(n_nodes, device=score.device, dtype=score.dtype)
    denom.scatter_add_(0, row, exp)
    return exp / denom[row].clamp_min(1e-12)


class ResidualGraphAttention(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_classes: int, heads: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden)
        self.q = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(heads)])
        self.k = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(heads)])
        self.v = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(heads)])
        self.mix = nn.Linear(hidden * heads, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden ** -0.5

    def forward(self, x: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        h0 = torch.relu(self.input(x))
        chunks = []
        for q_layer, k_layer, v_layer in zip(self.q, self.k, self.v):
            q = q_layer(h0)
            k = k_layer(h0)
            v = v_layer(h0)
            score = (q[row] * k[col]).sum(dim=1) * self.scale
            weight = edge_softmax(score, row, h0.shape[0])
            out = torch.zeros_like(h0)
            out.index_add_(0, row, v[col] * weight.unsqueeze(1))
            chunks.append(out)
        h = self.norm1(h0 + self.dropout(self.mix(torch.cat(chunks, dim=1))))
        h = self.norm2(h + self.ffn(h))
        return self.head(h)


def fit_model(x_train: np.ndarray, y_train: np.ndarray, train_frame: pd.DataFrame, n_classes: int, seed: int, args):
    torch.manual_seed(seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    row, col = build_edges(train_frame, x_train, args.knn)
    x = torch.tensor(x_train[train_frame.index.to_numpy(dtype=np.int64)], dtype=torch.float32, device=device)
    y = torch.tensor(y_train[train_frame.index.to_numpy(dtype=np.int64)], dtype=torch.long, device=device)
    row = row.to(device)
    col = col.to(device)
    model = ResidualGraphAttention(x.shape[1], args.hidden, n_classes, args.heads, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y.cpu().numpy(), n_classes, args.class_weight_power).to(device))
    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x, row, col), y)
        loss.backward()
        optimizer.step()
    return model


def predict_model(model: nn.Module, x_all: np.ndarray, frame: pd.DataFrame, n_classes: int, args) -> np.ndarray:
    device = next(model.parameters()).device
    ordered = frame.sort_values(["WELL", "DEPTH_MD"])
    pos = ordered.index.to_numpy(dtype=np.int64)
    compact = ordered.reset_index(drop=True)
    x = torch.tensor(x_all[pos], dtype=torch.float32, device=device)
    row, col = build_edges(compact, x_all[pos], args.knn)
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(x, row.to(device), col.to(device)), dim=1).cpu().numpy()
    if proba.shape[1] != n_classes:
        out = np.zeros((len(proba), n_classes), dtype=np.float64)
        out[:, : proba.shape[1]] = proba
        proba = out
    return proba


def run_seed(seed: int, args) -> list[dict]:
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, features = build_features(df, include_missing=True)
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    train_wells, interp_wells, extra_wells, _ = split_wells_by_space(df, args.train_fraction, args.interp_test_wells, seed)
    train_mask = df["WELL"].isin(train_wells).to_numpy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(df.loc[train_mask, features]))
    x_all = scaler.transform(imputer.transform(df[features]))
    x_global = np.zeros_like(x_all)
    x_global[train_mask] = x_train
    x_global[~train_mask] = x_all[~train_mask]
    labels = df["TARGET"].to_numpy(dtype=np.int64)
    train_frame = df[df["WELL"].isin(train_wells)].copy()
    n_classes = len(class_names)
    start = time.time()
    model = fit_model(x_global, labels, train_frame, n_classes, seed, args)
    train_time = time.time() - start
    rows = []
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        frame = df[df["WELL"].isin(wells)].copy()
        ordered = frame.sort_values(["WELL", "DEPTH_MD"])
        proba = predict_model(model, x_global, ordered, n_classes, args)
        y = ordered["TARGET"].to_numpy(dtype=np.int64)
        rows.extend(selective_rows(y, proba, seed, split, "resgat_margin", args.coverages))
        full = metrics(y, proba.argmax(axis=1))
        full.update({"seed": seed, "split": split, "method": "resgat_full", "coverage": 1.0, "kept_rows": len(y)})
        rows.append(full)
    for row in rows:
        row.update({"epochs": args.epochs, "hidden": args.hidden, "heads": args.heads, "knn": args.knn, "train_time": train_time})
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
    rows = []
    cand_all = raw[(raw["split"] == "extrapolation") & (raw["method"] == "resgat_margin")]
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
                        "method": "resgat_margin",
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
    frame = pd.DataFrame({"WELL": ["A"] * 8 + ["B"] * 8, "DEPTH_MD": list(range(8)) * 2})
    x = np.random.default_rng(0).normal(size=(16, 5))
    row, col = build_edges(frame, x, 2)
    model = ResidualGraphAttention(5, 12, 3, 2, 0.1)
    out = torch.softmax(model(torch.tensor(x, dtype=torch.float32), row, col), dim=1).detach().numpy()
    rows = selective_rows(np.array([0, 1, 2, 0, 1, 2, 0, 1] * 2), out, 0, "self_check", "resgat_margin", [0.25])
    if out.shape != (16, 3) or not np.allclose(out.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("ResGAT probabilities are invalid.")
    if rows[0]["kept_rows"] != 4:
        raise RuntimeError("ResGAT selective self-check failed.")
    print("recent_graph_attention_baseline self-check passed")


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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--knn", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-csv", type=Path, default=Path("results/recent_resgat_force_11seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/recent_resgat_force_11seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/recent_resgat_force_11seed_paired.csv"))
    parser.add_argument("--reference-csv", type=Path, default=Path("results/force_release_seed_level_reference.csv"))
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
    paired = paired_stats(raw, args.reference_csv)
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
