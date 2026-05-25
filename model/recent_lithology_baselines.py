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
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
    score = np.sort(proba, axis=1)[:, -1] - np.sort(proba, axis=1)[:, -2]
    order = np.argsort(-score)
    rows = []
    for coverage in coverages:
        keep = max(1, int(round(len(y) * coverage)))
        idx = order[:keep]
        row = metrics(y[idx], pred[idx])
        row.update(
            {
                "seed": seed,
                "split": split,
                "method": method,
                "coverage": coverage,
                "kept_rows": keep,
            }
        )
        rows.append(row)
    return rows


class WindowDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, seq_matrix: np.ndarray, point_matrix: np.ndarray, labels: np.ndarray | None, window: int):
        self.seq_matrix = seq_matrix.astype(np.float32, copy=False)
        self.point_matrix = point_matrix.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False) if labels is not None else None
        self.window = window
        self.half = window // 2
        self.samples = []
        for _, group in frame.groupby("WELL", sort=False):
            ordered = group.sort_values("DEPTH_MD")
            positions = ordered.index.to_numpy(dtype=np.int64)
            for local_pos, _ in enumerate(positions):
                self.samples.append((positions, local_pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        positions, local_pos = self.samples[idx]
        center = positions[local_pos]
        start = max(0, local_pos - self.half)
        end = min(len(positions), local_pos + self.half + 1)
        take = positions[start:end]
        seq = np.zeros((self.window, self.seq_matrix.shape[1]), dtype=np.float32)
        insert = self.half - (local_pos - start)
        seq[insert : insert + len(take)] = self.seq_matrix[take]
        point = self.point_matrix[center]
        if self.labels is None:
            return torch.from_numpy(seq), torch.from_numpy(point), center
        return torch.from_numpy(seq), torch.from_numpy(point), torch.tensor(self.labels[center], dtype=torch.long)


class AttentionCNN(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.att = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.point = nn.Sequential(nn.Linear(point_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, seq, point):
        x = self.conv(seq.transpose(1, 2)).transpose(1, 2)
        w = torch.softmax(self.att(x).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (x * w).sum(dim=1)
        return self.head(torch.cat([pooled, self.point(point)], dim=1))


class RecurrentTransformer(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float, heads: int, layers: int):
        super().__init__()
        self.inp = nn.Linear(seq_dim, hidden)
        self.gru = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.point = nn.Sequential(nn.Linear(point_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, seq, point):
        x = self.inp(seq)
        x, _ = self.gru(x)
        x = self.encoder(x)
        return self.head(torch.cat([x.mean(dim=1), self.point(point)], dim=1))


class MultiscaleCNN(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        branch_hidden = max(8, hidden // 3)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(seq_dim, branch_hidden, kernel_size=k, padding=k // 2), nn.BatchNorm1d(branch_hidden), nn.ReLU())
                for k in (3, 5, 9)
            ]
        )
        merged = branch_hidden * len(self.branches)
        self.mix = nn.Sequential(nn.Conv1d(merged, hidden, kernel_size=1), nn.BatchNorm1d(hidden), nn.ReLU())
        self.point = nn.Sequential(nn.Linear(point_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, seq, point):
        x = seq.transpose(1, 2)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.mix(x).mean(dim=2)
        return self.head(torch.cat([x, self.point(point)], dim=1))


class DiffusionDenoiser(nn.Module):
    def __init__(self, dim: int, hidden: int, noise_levels: int):
        super().__init__()
        self.embed = nn.Embedding(noise_levels, hidden)
        self.net = nn.Sequential(nn.Linear(dim + hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, level):
        return self.net(torch.cat([x, self.embed(level)], dim=1))


def class_weights(labels: np.ndarray, n_classes: int, power: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (counts.sum() / (n_classes * counts)) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def sampler(dataset: WindowDataset, labels: np.ndarray, n_classes: int, power: float, seed: int) -> WeightedRandomSampler:
    weights = class_weights(labels, n_classes, power).numpy()
    sample_weights = np.array([weights[int(labels[center])] for _, center in dataset.samples], dtype=np.float64)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)


def make_model(name: str, seq_dim: int, point_dim: int, n_classes: int, args):
    if name == "att_cnn":
        return AttentionCNN(seq_dim, point_dim, n_classes, args.hidden, args.dropout)
    if name == "recurrent_transformer":
        return RecurrentTransformer(seq_dim, point_dim, n_classes, args.hidden, args.dropout, args.heads, args.layers)
    if name == "ddpm_mscnn":
        return MultiscaleCNN(seq_dim, point_dim, n_classes, args.hidden, args.dropout)
    raise ValueError(f"Unknown recent baseline: {name}")


def train_denoiser(seq_train: np.ndarray, args, device: torch.device) -> DiffusionDenoiser:
    dim = seq_train.shape[1]
    denoiser = DiffusionDenoiser(dim, args.hidden, args.noise_levels).to(device)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data = torch.tensor(seq_train, dtype=torch.float32)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed_for_loader))
    betas = torch.linspace(args.noise_min, args.noise_max, args.noise_levels, device=device)
    denoiser.train()
    for _ in range(args.diffusion_epochs):
        for x in loader:
            x = x.to(device)
            level = torch.randint(0, args.noise_levels, (x.shape[0],), device=device)
            noise = torch.randn_like(x) * betas[level].unsqueeze(1)
            pred = denoiser(x + noise, level)
            loss = torch.mean((pred - x) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return denoiser


def augment_minority(seq_train: np.ndarray, point_train: np.ndarray, labels: np.ndarray, args, device: torch.device):
    if args.diffusion_ratio <= 0:
        return seq_train, point_train, labels
    counts = np.bincount(labels)
    target = int(np.quantile(counts[counts > 0], args.diffusion_target_quantile))
    denoiser = train_denoiser(seq_train, args, device)
    rows_seq, rows_point, rows_y = [seq_train], [point_train], [labels]
    rng = np.random.default_rng(args.seed_for_loader)
    denoiser.eval()
    with torch.no_grad():
        for cls, count in enumerate(counts):
            need = max(0, min(target - count, int(count * args.diffusion_ratio)))
            if need <= 0:
                continue
            src = np.where(labels == cls)[0]
            pick = rng.choice(src, size=need, replace=True)
            x = torch.tensor(seq_train[pick], dtype=torch.float32, device=device)
            level = torch.full((need,), args.noise_levels - 1, dtype=torch.long, device=device)
            noisy = x + torch.randn_like(x) * args.noise_max
            seq_new = denoiser(noisy, level).cpu().numpy()
            point_new = point_train[pick].copy()
            rows_seq.append(seq_new)
            rows_point.append(point_new)
            rows_y.append(np.full(need, cls, dtype=np.int64))
    return np.vstack(rows_seq), np.vstack(rows_point), np.concatenate(rows_y)


def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device, n: int, n_classes: int) -> np.ndarray:
    model.eval()
    out = np.zeros((n, n_classes), dtype=np.float32)
    with torch.no_grad():
        for seq, point, row_index in loader:
            proba = torch.softmax(model(seq.to(device), point.to(device)), dim=1).cpu().numpy()
            out[row_index.numpy()] = proba
    return out


def run_seed(seed: int, args) -> list[dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    args.seed_for_loader = seed
    df, class_names = load_force(args.data_dir, args.target)
    df = sample_by_well(df, args.max_rows_per_well, seed)
    df, all_features = build_features(df, include_missing=True)
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    train_wells, interp_wells, extra_wells, _ = split_wells_by_space(df, args.train_fraction, args.interp_test_wells, seed)
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
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_frame = df[df["WELL"].isin(train_wells)].copy()
    train_dataset = WindowDataset(train_frame, seq_all, point_all, labels, args.window)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler(train_dataset, labels, n_classes, args.class_weight_power, seed),
        num_workers=0,
    )
    model = make_model(args.model, len(seq_features), len(point_features), n_classes, args).to(device)
    if args.model == "ddpm_mscnn":
        train_centers = np.array([center for _, center in train_dataset.samples], dtype=np.int64)
        seq_aug, point_aug, labels_aug = augment_minority(seq_all[train_centers], point_all[train_centers], labels[train_centers], args, device)
        aug_frame = pd.DataFrame({"WELL": "aug", "DEPTH_MD": np.arange(len(labels_aug))})
        seq_all_train = seq_aug
        point_all_train = point_aug
        train_dataset = WindowDataset(aug_frame, seq_all_train, point_all_train, labels_aug, args.window)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler(train_dataset, labels_aug, n_classes, args.class_weight_power, seed),
            num_workers=0,
        )
        train_labels_for_weight = labels_aug
    else:
        train_labels_for_weight = labels[train_mask]
    weight = class_weights(train_labels_for_weight, n_classes, args.class_weight_power).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        frame = df[df["WELL"].isin(wells)].copy()
        dataset = WindowDataset(frame, seq_all, point_all, None, args.window)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        proba_all = predict_proba(model, loader, device, len(df), n_classes)
        ordered = frame.sort_values(["WELL", "DEPTH_MD"])
        proba = proba_all[ordered.index.to_numpy()]
        y = ordered["TARGET"].to_numpy(dtype=np.int64)
        rows.extend(selective_rows(y, proba, seed, split, f"{args.model}_margin", args.coverages))
        full = metrics(y, proba.argmax(axis=1))
        full.update({"seed": seed, "split": split, "method": f"{args.model}_full", "coverage": 1.0, "kept_rows": len(y)})
        rows.append(full)
    for row in rows:
        row.update({"window": args.window, "epochs": args.epochs, "hidden": args.hidden, "train_time": train_time})
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
    for coverage in sorted(set(raw["coverage"]) - {1.0}):
        cand = raw[(raw["split"] == "extrapolation") & (raw["coverage"] == coverage)].set_index("seed")
        for method in sorted(cand["method"].unique()):
            c = cand[cand["method"] == method]
            for baseline in ["ProfileAnchor-Seq", "Random forest"]:
                b = base[(base["method"] == baseline) & (base["split"] == "extrapolation") & (base["coverage"] == coverage)].set_index("seed")
                common = sorted(set(c.index) & set(b.index))
                if not common:
                    continue
                for metric in ["Accuracy", "Balanced Accuracy", "MCC", "F1_macro", "F1_weighted"]:
                    cv = c.loc[common, metric]
                    bv = b.loc[common, metric]
                    diff = cv - bv
                    rows.append(
                        {
                            "method": method,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/force2020"))
    parser.add_argument("--target", default="FORCE_2020_LITHOFACIES_LITHOLOGY")
    parser.add_argument("--model", choices=["att_cnn", "recurrent_transformer", "ddpm_mscnn"], default="att_cnn")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42])
    parser.add_argument("--coverages", nargs="+", type=float, default=DEFAULT_COVERAGES)
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--interp-test-wells", type=int, default=10)
    parser.add_argument("--max-rows-per-well", type=int, default=None)
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--diffusion-epochs", type=int, default=3)
    parser.add_argument("--diffusion-ratio", type=float, default=1.0)
    parser.add_argument("--diffusion-target-quantile", type=float, default=0.75)
    parser.add_argument("--noise-levels", type=int, default=8)
    parser.add_argument("--noise-min", type=float, default=0.03)
    parser.add_argument("--noise-max", type=float, default=0.35)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--paired-csv", type=Path, default=None)
    parser.add_argument("--reference-summary", type=Path, default=Path("results/profile_anchor_seq_force_11seed_summary.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        torch.manual_seed(0)
        seq = torch.randn(6, 11, 5)
        point = torch.randn(6, 7)
        for name in ["att_cnn", "recurrent_transformer", "ddpm_mscnn"]:
            args.model = name
            model = make_model(name, 5, 7, 4, args)
            proba = torch.softmax(model(seq, point), dim=1).detach().numpy()
            rows = selective_rows(np.array([0, 1, 2, 3, 1, 0]), proba, 0, "self_check", f"{name}_margin", [0.5])
            if proba.shape != (6, 4) or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
                raise RuntimeError(f"{name} produced invalid probabilities.")
            if not rows or rows[0]["kept_rows"] != 3:
                raise RuntimeError(f"{name} selective check failed.")
        print("recent_lithology_baselines self-check passed")
        return
    if args.out_csv is None:
        args.out_csv = Path(f"results/recent_{args.model}_force_11seed.csv")
    if args.summary_csv is None:
        args.summary_csv = Path(f"results/recent_{args.model}_force_11seed_summary.csv")
    if args.paired_csv is None:
        args.paired_csv = Path(f"results/recent_{args.model}_force_11seed_paired.csv")
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
