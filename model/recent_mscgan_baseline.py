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


class TensorWindowDataset(Dataset):
    def __init__(self, seq: np.ndarray, point: np.ndarray, labels: np.ndarray):
        self.seq = seq.astype(np.float32, copy=False)
        self.point = point.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.seq[idx]), torch.from_numpy(self.point[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim: int, n_classes: int, output_dim: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(n_classes, hidden)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden * 2, output_dim),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, self.embed(y)], dim=1))


class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(n_classes, hidden)
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden, hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden * 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.embed(y)], dim=1)).squeeze(1)


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


def class_weights(labels: np.ndarray, n_classes: int, power: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (counts.sum() / (n_classes * counts)) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def sampler(labels: np.ndarray, n_classes: int, power: float, seed: int) -> WeightedRandomSampler:
    weights = class_weights(labels, n_classes, power).numpy()
    sample_weights = np.array([weights[int(label)] for label in labels], dtype=np.float64)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)


def materialize_windows(dataset: WindowDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_rows = []
    point_rows = []
    label_rows = []
    for idx in range(len(dataset)):
        seq, point, label = dataset[idx]
        seq_rows.append(seq.numpy())
        point_rows.append(point.numpy())
        label_rows.append(int(label))
    return np.stack(seq_rows), np.stack(point_rows), np.asarray(label_rows, dtype=np.int64)


def train_generator(flat: np.ndarray, labels: np.ndarray, n_classes: int, args, device: torch.device) -> ConditionalGenerator:
    generator = ConditionalGenerator(args.noise_dim, n_classes, flat.shape[1], args.gan_hidden).to(device)
    discriminator = ConditionalDiscriminator(flat.shape[1], n_classes, args.gan_hidden).to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=args.gan_lr, betas=(0.5, 0.99), weight_decay=args.weight_decay)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.gan_lr, betas=(0.5, 0.99), weight_decay=args.weight_decay)
    data = TensorWindowDataset(flat.reshape((flat.shape[0], 1, flat.shape[1])), np.zeros((flat.shape[0], 1), dtype=np.float32), labels)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed_for_loader))
    loss_fn = nn.BCEWithLogitsLoss()
    generator.train()
    discriminator.train()
    for _ in range(args.gan_epochs):
        for real_seq, _, y in loader:
            real = real_seq.squeeze(1).to(device)
            y = y.to(device)
            z = torch.randn(real.shape[0], args.noise_dim, device=device)
            fake = generator(z, y).detach()
            loss_d = loss_fn(discriminator(real, y), torch.ones(real.shape[0], device=device)) + loss_fn(
                discriminator(fake, y), torch.zeros(real.shape[0], device=device)
            )
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            z = torch.randn(real.shape[0], args.noise_dim, device=device)
            fake = generator(z, y)
            adv = loss_fn(discriminator(fake, y), torch.ones(real.shape[0], device=device))
            recon_target = real[torch.randperm(real.shape[0], device=device)]
            moment = torch.mean((fake.mean(dim=0) - real.mean(dim=0)) ** 2) + torch.mean((fake.std(dim=0) - real.std(dim=0)) ** 2)
            loss_g = adv + args.moment_weight * moment + args.reconstruction_weight * torch.mean((fake - recon_target) ** 2)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
    return generator


def augment_with_cgan(seq: np.ndarray, point: np.ndarray, labels: np.ndarray, n_classes: int, args, device: torch.device):
    counts = np.bincount(labels, minlength=n_classes)
    valid = counts[counts > 0]
    if len(valid) == 0 or args.augment_ratio <= 0:
        return seq, point, labels, 0
    target = int(np.quantile(valid, args.target_quantile))
    max_new = int(len(labels) * args.max_augmented_multiplier) - len(labels)
    if target <= 0 or max_new <= 0:
        return seq, point, labels, 0
    flat = seq.reshape(seq.shape[0], -1)
    generator = train_generator(flat, labels, n_classes, args, device)
    rng = np.random.default_rng(args.seed_for_loader)
    rows_seq = [seq]
    rows_point = [point]
    rows_y = [labels]
    total_new = 0
    generator.eval()
    with torch.no_grad():
        for cls, count in enumerate(counts):
            if count == 0:
                continue
            need = max(0, min(target - int(count), int(count * args.augment_ratio)))
            need = min(need, max_new - total_new)
            if need <= 0:
                continue
            z = torch.randn(need, args.noise_dim, device=device)
            y = torch.full((need,), cls, dtype=torch.long, device=device)
            fake_seq = generator(z, y).cpu().numpy().reshape((need, *seq.shape[1:]))
            src = np.flatnonzero(labels == cls)
            fake_point = point[rng.choice(src, size=need, replace=True)].copy()
            rows_seq.append(fake_seq)
            rows_point.append(fake_point)
            rows_y.append(np.full(need, cls, dtype=np.int64))
            total_new += need
            if total_new >= max_new:
                break
    return np.vstack(rows_seq), np.vstack(rows_point), np.concatenate(rows_y), total_new


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
    seq_imputer = SimpleImputer(strategy="median")
    point_imputer = SimpleImputer(strategy="median")
    seq_scaler = StandardScaler()
    point_scaler = StandardScaler()
    seq_train = seq_scaler.fit_transform(seq_imputer.fit_transform(df.loc[train_mask, seq_features]))
    point_train = point_scaler.fit_transform(point_imputer.fit_transform(df.loc[train_mask, all_features]))
    seq_all = seq_scaler.transform(seq_imputer.transform(df[seq_features]))
    point_all = point_scaler.transform(point_imputer.transform(df[all_features]))
    labels = df["TARGET"].to_numpy(dtype=np.int64)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_frame = df[df["WELL"].isin(train_wells)].copy()
    train_dataset = WindowDataset(train_frame, seq_all, point_all, labels, args.window)
    seq_windows, point_windows, labels_train = materialize_windows(train_dataset)
    seq_aug, point_aug, labels_aug, synthetic_rows = augment_with_cgan(seq_windows, point_windows, labels_train, n_classes, args, device)
    train_aug = TensorWindowDataset(seq_aug, point_aug, labels_aug)
    loader = DataLoader(
        train_aug,
        batch_size=args.batch_size,
        sampler=sampler(labels_aug, n_classes, args.class_weight_power, seed),
        num_workers=0,
    )
    model = MultiscaleCNN(len(seq_features), len(all_features), n_classes, args.hidden, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(labels_aug, n_classes, args.class_weight_power).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start = time.time()
    model.train()
    for _ in range(args.epochs):
        for seq, point, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(seq.to(device), point.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
    train_time = time.time() - start
    rows = []
    for split, wells in [("interpolation", interp_wells), ("extrapolation", extra_wells)]:
        frame = df[df["WELL"].isin(wells)].copy()
        dataset = WindowDataset(frame, seq_all, point_all, None, args.window)
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        proba_all = predict_proba(model, test_loader, device, len(df), n_classes)
        ordered = frame.sort_values(["WELL", "DEPTH_MD"])
        proba = proba_all[ordered.index.to_numpy()]
        y = ordered["TARGET"].to_numpy(dtype=np.int64)
        rows.extend(selective_rows(y, proba, seed, split, "mscgan_mscnn_margin", args.coverages))
        full = metrics(y, proba.argmax(axis=1))
        full.update({"seed": seed, "split": split, "method": "mscgan_mscnn_full", "coverage": 1.0, "kept_rows": len(y)})
        rows.append(full)
    for row in rows:
        row.update({"window": args.window, "epochs": args.epochs, "hidden": args.hidden, "synthetic_rows": synthetic_rows, "train_time": train_time})
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
    cand_all = raw[(raw["split"] == "extrapolation") & (raw["method"] == "mscgan_mscnn_margin")]
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
                        "method": "mscgan_mscnn_margin",
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
    seq = rng.normal(size=(24, 9, 4)).astype(np.float32)
    point = rng.normal(size=(24, 5)).astype(np.float32)
    labels = np.array([0] * 10 + [1] * 8 + [2] * 6, dtype=np.int64)
    device = torch.device("cpu")
    args.seed_for_loader = 0
    seq_aug, point_aug, labels_aug, synthetic_rows = augment_with_cgan(seq, point, labels, 3, args, device)
    model = MultiscaleCNN(4, 5, 3, args.hidden, args.dropout)
    out = torch.softmax(model(torch.tensor(seq_aug[:6]), torch.tensor(point_aug[:6])), dim=1).detach().numpy()
    rows = selective_rows(labels_aug[:6], out, 0, "self_check", "mscgan_mscnn_margin", [0.5])
    if out.shape != (6, 3) or not np.allclose(out.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("MS-CGAN probabilities are invalid.")
    if synthetic_rows <= 0 or rows[0]["kept_rows"] != 3:
        raise RuntimeError("MS-CGAN selective self-check failed.")
    print("recent_mscgan_baseline self-check passed")


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
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gan-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--gan-epochs", type=int, default=3)
    parser.add_argument("--gan-hidden", type=int, default=128)
    parser.add_argument("--noise-dim", type=int, default=64)
    parser.add_argument("--augment-ratio", type=float, default=1.0)
    parser.add_argument("--target-quantile", type=float, default=0.75)
    parser.add_argument("--max-augmented-multiplier", type=float, default=1.5)
    parser.add_argument("--moment-weight", type=float, default=0.2)
    parser.add_argument("--reconstruction-weight", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-csv", type=Path, default=Path("results/recent_mscgan_mscnn_force_11seed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/recent_mscgan_mscnn_force_11seed_summary.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("results/recent_mscgan_mscnn_force_11seed_paired.csv"))
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
