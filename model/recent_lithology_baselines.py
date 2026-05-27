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


class TensorWindowDataset(Dataset):
    def __init__(self, seq: np.ndarray, point: np.ndarray, labels: np.ndarray):
        self.seq = seq.astype(np.float32, copy=False)
        self.point = point.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.seq[idx]),
            torch.from_numpy(self.point[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


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


class ReFormerStyle(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float, heads: int, layers: int, window: int):
        super().__init__()
        self.token = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )
        self.pos = nn.Parameter(torch.randn(1, window, hidden) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.att = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.point = nn.Sequential(nn.Linear(point_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, seq, point):
        x = self.token(seq.transpose(1, 2)).transpose(1, 2)
        if x.shape[1] != self.pos.shape[1]:
            pos = torch.nn.functional.interpolate(self.pos.transpose(1, 2), size=x.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        else:
            pos = self.pos
        x = self.encoder(x + pos)
        weights = torch.softmax(self.att(x).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        return self.head(torch.cat([pooled, self.point(point)], dim=1))


class BoostedTransformerStage(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float, heads: int, layers: int, window: int):
        super().__init__()
        self.token = nn.Linear(seq_dim, hidden)
        self.pos = nn.Parameter(torch.randn(1, window, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.point = nn.Sequential(nn.Linear(point_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, seq, point):
        x = self.token(seq)
        if x.shape[1] != self.pos.shape[1]:
            pos = torch.nn.functional.interpolate(self.pos.transpose(1, 2), size=x.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        else:
            pos = self.pos
        x = self.encoder(x + pos)
        return self.head(torch.cat([x.mean(dim=1), self.point(point)], dim=1))


class AdaBoostTransformerEnsemble(nn.Module):
    def __init__(self, stages: list[nn.Module], alphas: list[float]):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))

    def forward(self, seq, point):
        logits = torch.stack([stage(seq, point) for stage in self.stages], dim=0)
        weights = self.alphas.view(-1, 1, 1)
        return (weights * logits).sum(dim=0) / torch.clamp(self.alphas.sum(), min=1e-6)


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


class MultiscaleFeatureFusionCNN(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        branch_hidden = max(8, hidden // 4)
        self.local = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(seq_dim, branch_hidden, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(branch_hidden),
                    nn.GELU(),
                    nn.Conv1d(branch_hidden, branch_hidden, kernel_size=1),
                    nn.GELU(),
                )
                for k in (3, 5, 7, 11)
            ]
        )
        merged = branch_hidden * 4
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(merged, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )
        self.cross_scale = nn.Sequential(
            nn.Conv1d(merged, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=max(1, hidden // 16)),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.point = nn.Sequential(nn.Linear(point_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, seq, point):
        x = seq.transpose(1, 2)
        branches = [layer(x) for layer in self.local]
        merged = torch.cat(branches, dim=1)
        weights = torch.softmax(self.fusion_gate(merged), dim=1).unsqueeze(-1).unsqueeze(-1)
        stacked = torch.stack(branches, dim=1)
        gated = (stacked * weights).sum(dim=1)
        fused = self.cross_scale(merged).mean(dim=2)
        local = gated.mean(dim=2)
        if local.shape[1] != fused.shape[1]:
            local = torch.nn.functional.pad(local, (0, fused.shape[1] - local.shape[1]))
        return self.head(torch.cat([0.5 * (fused + local), self.point(point)], dim=1))


class LMAFNetStyle(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        branch_hidden = max(8, hidden // 4)
        kernels = (3, 5, 9, 15)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(seq_dim, branch_hidden, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(branch_hidden),
                    nn.GELU(),
                    nn.Conv1d(branch_hidden, branch_hidden, kernel_size=1),
                    nn.GELU(),
                )
                for k in kernels
            ]
        )
        merged = branch_hidden * len(kernels)
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(merged, max(8, merged // 2)),
            nn.GELU(),
            nn.Linear(max(8, merged // 2), len(kernels)),
        )
        self.vertical_gate = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, len(kernels), kernel_size=1),
        )
        self.point = nn.Sequential(nn.Linear(point_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(branch_hidden + hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, seq, point):
        x = seq.transpose(1, 2)
        branches = torch.stack([branch(x) for branch in self.branches], dim=1)
        branch_context = torch.cat([branches[:, i] for i in range(branches.shape[1])], dim=1)
        global_gate = torch.softmax(self.branch_gate(branch_context), dim=1).unsqueeze(-1).unsqueeze(-1)
        vertical_gate = torch.softmax(self.vertical_gate(x), dim=1).unsqueeze(2)
        fused = (branches * global_gate * vertical_gate).sum(dim=1).mean(dim=2)
        return self.head(torch.cat([fused, self.point(point)], dim=1))


class MultiModelFusionNet(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float, heads: int, layers: int):
        super().__init__()
        self.att_branch = AttentionCNN(seq_dim, point_dim, n_classes, hidden, dropout)
        self.rnn_branch = RecurrentTransformer(seq_dim, point_dim, n_classes, hidden, dropout, heads, layers)
        self.ms_branch = MultiscaleCNN(seq_dim, point_dim, n_classes, hidden, dropout)
        self.gate = nn.Sequential(
            nn.Linear(point_dim + seq_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, seq, point):
        summary = torch.cat([point, seq.mean(dim=1)], dim=1)
        weights = torch.softmax(self.gate(summary), dim=1)
        logits = torch.stack([self.att_branch(seq, point), self.rnn_branch(seq, point), self.ms_branch(seq, point)], dim=1)
        return (logits * weights.unsqueeze(-1)).sum(dim=1)


class SerialEnsembleNet(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float, heads: int, layers: int):
        super().__init__()
        self.stage1 = MultiscaleCNN(seq_dim, point_dim, n_classes, hidden, dropout)
        self.stage2 = AttentionCNN(seq_dim, point_dim, n_classes, hidden, dropout)
        self.stage3 = RecurrentTransformer(seq_dim, point_dim, n_classes, hidden, dropout, heads, layers)
        self.gate = nn.Sequential(
            nn.Linear(point_dim + seq_dim + n_classes * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, seq, point):
        logits1 = self.stage1(seq, point)
        logits2 = self.stage2(seq, point)
        logits3 = self.stage3(seq, point)
        summary = torch.cat([point, seq.mean(dim=1), torch.softmax(logits1, dim=1), torch.softmax(logits2, dim=1)], dim=1)
        weights = torch.softmax(self.gate(summary), dim=1)
        logits = torch.stack([logits1, logits2, logits3], dim=1)
        return (logits * weights.unsqueeze(-1)).sum(dim=1)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class SVATCNStyle(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.proj = nn.Conv1d(seq_dim, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            TCNBlock(hidden, kernel=3, dilation=1, dropout=dropout),
            TCNBlock(hidden, kernel=3, dilation=2, dropout=dropout),
            TCNBlock(hidden, kernel=5, dilation=4, dropout=dropout),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, max(8, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(8, hidden // 2), hidden),
            nn.Sigmoid(),
        )
        self.temporal_gate = nn.Sequential(nn.Conv1d(hidden, hidden // 2, kernel_size=1), nn.GELU(), nn.Conv1d(hidden // 2, 1, kernel_size=1))
        self.point = nn.Sequential(nn.Linear(point_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, seq, point):
        x = self.blocks(self.proj(seq.transpose(1, 2)))
        x = x * self.channel_gate(x).unsqueeze(-1)
        weights = torch.softmax(self.temporal_gate(x).squeeze(1), dim=1).unsqueeze(1)
        pooled = (x * weights).sum(dim=2)
        return self.head(torch.cat([pooled, self.point(point)], dim=1))


class SpatialStratigraphicDynamicRangeAttention(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.seq_proj = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.range_proj = nn.Sequential(
            nn.Linear(seq_dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.strat_gate = nn.Sequential(
            nn.Conv1d(hidden + seq_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )
        self.spatial = nn.Sequential(nn.Linear(point_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.compat = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, seq, point):
        x = seq.transpose(1, 2)
        seq_repr = self.seq_proj(x)
        stats = torch.cat(
            [
                seq.mean(dim=1),
                seq.std(dim=1, unbiased=False),
                seq.amax(dim=1) - seq.amin(dim=1),
                seq[:, seq.shape[1] // 2, :],
            ],
            dim=1,
        )
        seq_repr = seq_repr * self.range_proj(stats).unsqueeze(-1)
        gate_input = torch.cat([seq_repr, x], dim=1)
        weights = torch.softmax(self.strat_gate(gate_input).squeeze(1), dim=1).unsqueeze(1)
        strat = (seq_repr * weights).sum(dim=2)
        spatial = self.spatial(point)
        fused = torch.cat([strat * self.compat(torch.cat([strat, spatial], dim=1)), spatial], dim=1)
        return self.head(fused)


class ClassWiseCorrelationFilter(nn.Module):
    def __init__(self, seq_dim: int, n_classes: int, hidden: int, window: int):
        super().__init__()
        self.class_filters = nn.Parameter(torch.randn(n_classes, seq_dim, window) * 0.02)
        self.scale = nn.Parameter(torch.ones(n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self.proj = nn.Sequential(
            nn.Linear(n_classes + seq_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, seq):
        x = torch.nn.functional.normalize(seq.transpose(1, 2), dim=(1, 2))
        if x.shape[-1] != self.class_filters.shape[-1]:
            x = torch.nn.functional.interpolate(x, size=self.class_filters.shape[-1], mode="linear", align_corners=False)
        filters = torch.nn.functional.normalize(self.class_filters, dim=(1, 2))
        corr = (x.unsqueeze(1) * filters.unsqueeze(0)).sum(dim=(2, 3))
        summary = seq.mean(dim=1)
        return self.proj(torch.cat([corr * self.scale + self.bias, summary], dim=1))


class CWSCFStyle(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float, window: int):
        super().__init__()
        self.filter = ClassWiseCorrelationFilter(seq_dim, n_classes, hidden, window)
        self.local = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.point = nn.Sequential(nn.Linear(point_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(hidden + point_dim, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, seq, point):
        filter_logits = self.filter(seq)
        local = self.local(seq.transpose(1, 2)).mean(dim=2)
        point_repr = self.point(point)
        neural_logits = self.head(torch.cat([local, point_repr], dim=1))
        weights = torch.softmax(self.gate(torch.cat([local, point], dim=1)), dim=1)
        return weights[:, :1] * filter_logits + weights[:, 1:] * neural_logits


class ShrinkageBlock2D(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(8, channels // 2)),
            nn.ReLU(),
            nn.Linear(max(8, channels // 2), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = self.conv(x)
        threshold = residual.abs().mean(dim=(2, 3), keepdim=True) * self.scale(residual).unsqueeze(-1).unsqueeze(-1)
        residual = torch.sign(residual) * torch.relu(residual.abs() - threshold)
        return torch.relu(x + residual)


class DRSNGAFStyle(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        base = max(16, hidden // 3)
        self.stem = nn.Sequential(
            nn.Conv2d(seq_dim, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(ShrinkageBlock2D(base, dropout), ShrinkageBlock2D(base, dropout))
        self.image_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.point = nn.Sequential(nn.Linear(point_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def gaf(self, seq):
        x = seq.transpose(1, 2).clamp(-1.0, 1.0)
        phi = torch.acos(x)
        return torch.cos(phi.unsqueeze(-1) + phi.unsqueeze(-2))

    def forward(self, seq, point):
        image = self.gaf(seq)
        image_repr = self.image_head(self.blocks(self.stem(image)))
        return self.head(torch.cat([image_repr, self.point(point)], dim=1))


class GeologyDrivenHybridNet(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        image_channels = 5
        base = max(16, hidden // 3)
        self.image = nn.Sequential(
            nn.Conv2d(1, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.GELU(),
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.seq_proj = nn.Linear(seq_dim * image_channels, hidden)
        self.bilstm = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.channel_gate = nn.Sequential(
            nn.Linear(seq_dim * image_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, seq_dim * image_channels),
            nn.Sigmoid(),
        )
        self.point = nn.Sequential(nn.Linear(point_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(base + hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def multiscale(self, seq):
        x = seq.transpose(1, 2)
        smooth3 = torch.nn.functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        smooth7 = torch.nn.functional.avg_pool1d(x, kernel_size=7, stride=1, padding=3)
        high3 = x - smooth3
        high7 = smooth3 - smooth7
        stacked = torch.stack([x, smooth3, smooth7, high3, high7], dim=2)
        return stacked

    def forward(self, seq, point):
        multi = self.multiscale(seq)
        image_repr = self.image(multi.flatten(1, 2).unsqueeze(1))
        tokens = multi.permute(0, 3, 1, 2).flatten(2)
        gate = self.channel_gate(tokens.mean(dim=1)).unsqueeze(1)
        tokens = tokens * gate
        seq_repr, _ = self.bilstm(self.seq_proj(tokens))
        seq_repr = seq_repr.mean(dim=1)
        return self.head(torch.cat([image_repr, seq_repr, self.point(point)], dim=1))


class DiffusionDenoiser(nn.Module):
    def __init__(self, dim: int, hidden: int, noise_levels: int):
        super().__init__()
        self.embed = nn.Embedding(noise_levels, hidden)
        self.net = nn.Sequential(nn.Linear(dim + hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, level):
        return self.net(torch.cat([x, self.embed(level)], dim=1))


class MRSSLEncoder(nn.Module):
    def __init__(self, seq_dim: int, hidden: int, dropout: float, heads: int, layers: int, window: int):
        super().__init__()
        self.time_token = nn.Sequential(
            nn.Linear(seq_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.freq_token = nn.Sequential(
            nn.Linear(seq_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.pos = nn.Parameter(torch.randn(1, window, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.time_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        freq_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.freq_encoder = nn.TransformerEncoder(freq_layer, num_layers=max(1, layers // 2))

    def forward(self, seq):
        time = self.time_token(seq) + self.pos[:, : seq.shape[1]]
        time = self.time_encoder(time)
        freq_amp = torch.fft.rfft(seq, dim=1).abs()
        freq_amp = torch.nn.functional.interpolate(freq_amp.transpose(1, 2), size=seq.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        freq = self.freq_token(freq_amp) + self.pos[:, : seq.shape[1]]
        freq = self.freq_encoder(freq)
        return time, freq


class MRSSLStyle(nn.Module):
    def __init__(self, seq_dim: int, point_dim: int, n_classes: int, hidden: int, dropout: float, heads: int, layers: int, window: int):
        super().__init__()
        self.encoder = MRSSLEncoder(seq_dim, hidden, dropout, heads, layers, window)
        self.point = nn.Sequential(nn.Linear(point_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, seq, point):
        time, freq = self.encoder(seq)
        return self.head(torch.cat([time.mean(dim=1), freq.mean(dim=1), self.point(point)], dim=1))


class MRSSLPretrainer(nn.Module):
    def __init__(self, seq_dim: int, hidden: int, dropout: float, heads: int, layers: int, window: int):
        super().__init__()
        self.encoder = MRSSLEncoder(seq_dim, hidden, dropout, heads, layers, window)
        self.time_decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, seq_dim))
        self.freq_decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, seq_dim))

    def forward(self, seq):
        time, freq = self.encoder(seq)
        return self.time_decoder(time), self.freq_decoder(freq), time.mean(dim=1), freq.mean(dim=1)


def class_weights(labels: np.ndarray, n_classes: int, power: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (counts.sum() / (n_classes * counts)) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def sampler(dataset: WindowDataset, labels: np.ndarray, n_classes: int, power: float, seed: int) -> WeightedRandomSampler:
    weights = class_weights(labels, n_classes, power).numpy()
    if hasattr(dataset, "samples"):
        sample_weights = np.array([weights[int(labels[center])] for _, center in dataset.samples], dtype=np.float64)
    else:
        sample_weights = np.array([weights[int(label)] for label in labels], dtype=np.float64)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)


def make_model(name: str, seq_dim: int, point_dim: int, n_classes: int, args):
    if name == "att_cnn":
        return AttentionCNN(seq_dim, point_dim, n_classes, args.hidden, args.dropout)
    if name == "recurrent_transformer":
        return RecurrentTransformer(seq_dim, point_dim, n_classes, args.hidden, args.dropout, args.heads, args.layers)
    if name == "reformer":
        return ReFormerStyle(seq_dim, point_dim, n_classes, args.hidden, args.dropout, args.heads, args.layers, args.window)
    if name == "adaboost_transformer":
        return BoostedTransformerStage(seq_dim, point_dim, n_classes, args.hidden, args.dropout, args.heads, args.layers, args.window)
    if name == "ddpm_mscnn":
        return MultiscaleCNN(seq_dim, point_dim, n_classes, args.hidden, args.dropout)
    if name == "mffcnn":
        return MultiscaleFeatureFusionCNN(seq_dim, point_dim, n_classes, args.hidden, args.dropout)
    if name == "lmafnet":
        return LMAFNetStyle(seq_dim, point_dim, n_classes, args.hidden, args.dropout)
    if name == "multimodel_fusion":
        return MultiModelFusionNet(seq_dim, point_dim, n_classes, args.hidden, args.dropout, args.heads, args.layers)
    if name == "serial_ensemble":
        return SerialEnsembleNet(seq_dim, point_dim, n_classes, args.hidden, args.dropout, args.heads, args.layers)
    if name == "sva_tcn":
        return SVATCNStyle(seq_dim, point_dim, n_classes, args.hidden, args.dropout)
    if name == "ssdra":
        return SpatialStratigraphicDynamicRangeAttention(seq_dim, point_dim, n_classes, args.hidden, args.dropout)
    if name == "cwscf":
        return CWSCFStyle(seq_dim, point_dim, n_classes, args.hidden, args.dropout, args.window)
    if name == "drsn_gaf":
        return DRSNGAFStyle(seq_dim, point_dim, n_classes, args.hidden, args.dropout)
    if name == "geology_hybrid":
        return GeologyDrivenHybridNet(seq_dim, point_dim, n_classes, args.hidden, args.dropout)
    if name == "mrssl":
        return MRSSLStyle(seq_dim, point_dim, n_classes, args.hidden, args.dropout, args.heads, args.layers, args.window)
    raise ValueError(f"Unknown recent baseline: {name}")


def train_one_stage(model: nn.Module, loader: DataLoader, weight: torch.Tensor, args, device: torch.device):
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for _ in range(args.epochs):
        for seq, point, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(seq.to(device), point.to(device)), y.to(device))
            loss.backward()
            optimizer.step()


def predict_tensor(model: nn.Module, seq: np.ndarray, point: np.ndarray, args, device: torch.device, n_classes: int) -> np.ndarray:
    dataset = TensorWindowDataset(seq, point, np.zeros(seq.shape[0], dtype=np.int64))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    out = np.zeros((seq.shape[0], n_classes), dtype=np.float32)
    pos = 0
    model.eval()
    with torch.no_grad():
        for batch_seq, batch_point, _ in loader:
            proba = torch.softmax(model(batch_seq.to(device), batch_point.to(device)), dim=1).cpu().numpy()
            out[pos : pos + proba.shape[0]] = proba
            pos += proba.shape[0]
    return out


def train_adaboost_transformer(train_dataset: WindowDataset, n_classes: int, seq_dim: int, point_dim: int, args, device: torch.device, seed: int):
    seq_windows, point_windows, labels_train = materialize_windows(train_dataset)
    sample_weights = np.ones(len(labels_train), dtype=np.float64) / max(1, len(labels_train))
    class_weight_base = class_weights(labels_train, n_classes, args.class_weight_power).to(device)
    stages = []
    alphas = []
    start = time.time()
    for stage_idx in range(args.boosting_stages):
        stage = BoostedTransformerStage(seq_dim, point_dim, n_classes, args.hidden, args.dropout, args.heads, args.layers, args.window).to(device)
        generator = torch.Generator().manual_seed(seed + 1009 * stage_idx)
        sampler_stage = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)
        loader = DataLoader(TensorWindowDataset(seq_windows, point_windows, labels_train), batch_size=args.batch_size, sampler=sampler_stage, num_workers=0)
        train_one_stage(stage, loader, class_weight_base, args, device)
        proba = predict_tensor(stage, seq_windows, point_windows, args, device, n_classes)
        pred = proba.argmax(axis=1)
        miss = (pred != labels_train).astype(np.float64)
        err = float(np.clip(np.sum(sample_weights * miss), 1e-4, 1.0 - 1e-4))
        alpha = float(np.log((1.0 - err) / err) + np.log(max(1, n_classes - 1)))
        alpha = max(alpha, 1e-3)
        sample_weights *= np.exp(alpha * miss)
        sample_weights /= sample_weights.sum()
        stages.append(stage)
        alphas.append(alpha)
    return AdaBoostTransformerEnsemble(stages, alphas).to(device), time.time() - start


def pretrain_mrssl(model: MRSSLStyle, train_dataset: Dataset, args, device: torch.device, seed: int):
    if args.pretrain_epochs <= 0:
        return
    pretrainer = MRSSLPretrainer(
        model.encoder.time_token[0].in_features,
        args.hidden,
        args.dropout,
        args.heads,
        args.layers,
        args.window,
    ).to(device)
    pretrainer.encoder.load_state_dict(model.encoder.state_dict())
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pretrainer.train()
    for _ in range(args.pretrain_epochs):
        for seq, _, *_ in loader:
            seq = seq.to(device)
            mask = torch.rand(seq.shape, device=device) < args.mask_ratio
            masked = seq.masked_fill(mask, 0.0)
            target_freq = torch.fft.rfft(seq, dim=1).abs()
            target_freq = torch.nn.functional.interpolate(target_freq.transpose(1, 2), size=seq.shape[1], mode="linear", align_corners=False).transpose(1, 2)
            pred_time, pred_freq, time_repr, freq_repr = pretrainer(masked)
            loss_time = ((pred_time - seq) ** 2)[mask].mean() if mask.any() else torch.mean((pred_time - seq) ** 2)
            loss_freq = torch.mean((pred_freq - target_freq) ** 2)
            time_repr = torch.nn.functional.normalize(time_repr, dim=1)
            freq_repr = torch.nn.functional.normalize(freq_repr, dim=1)
            logits = time_repr @ freq_repr.T / args.contrastive_temperature
            labels = torch.arange(logits.shape[0], device=device)
            loss_contrast = 0.5 * (
                torch.nn.functional.cross_entropy(logits, labels)
                + torch.nn.functional.cross_entropy(logits.T, labels)
            )
            loss = loss_time + args.freq_loss_weight * loss_freq + args.contrastive_weight * loss_contrast
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.encoder.load_state_dict(pretrainer.encoder.state_dict())


def train_denoiser(seq_windows: np.ndarray, args, device: torch.device) -> DiffusionDenoiser:
    flat = seq_windows.reshape(seq_windows.shape[0], -1)
    dim = flat.shape[1]
    denoiser = DiffusionDenoiser(dim, args.hidden, args.noise_levels).to(device)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data = torch.tensor(flat, dtype=torch.float32)
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


def augment_minority(seq_windows: np.ndarray, point_train: np.ndarray, labels: np.ndarray, args, device: torch.device):
    if args.diffusion_ratio <= 0:
        return seq_windows, point_train, labels
    counts = np.bincount(labels)
    target = int(np.quantile(counts[counts > 0], args.diffusion_target_quantile))
    denoiser = train_denoiser(seq_windows, args, device)
    rows_seq, rows_point, rows_y = [seq_windows], [point_train], [labels]
    rng = np.random.default_rng(args.seed_for_loader)
    denoiser.eval()
    with torch.no_grad():
        for cls, count in enumerate(counts):
            need = max(0, min(target - count, int(count * args.diffusion_ratio)))
            if need <= 0:
                continue
            src = np.where(labels == cls)[0]
            pick = rng.choice(src, size=need, replace=True)
            shape = seq_windows.shape[1:]
            flat = torch.tensor(seq_windows[pick].reshape(need, -1), dtype=torch.float32, device=device)
            level = torch.full((need,), args.noise_levels - 1, dtype=torch.long, device=device)
            noisy = flat + torch.randn_like(flat) * args.noise_max
            seq_new = denoiser(noisy, level).cpu().numpy().reshape((need, *shape))
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
    if args.model == "adaboost_transformer":
        model, train_time = train_adaboost_transformer(train_dataset, n_classes, len(seq_features), len(point_features), args, device, seed)
        train_labels_for_weight = labels[train_mask]
    elif args.model == "ddpm_mscnn":
        seq_windows, point_windows, labels_aug_base = materialize_windows(train_dataset)
        seq_aug, point_aug, labels_aug = augment_minority(seq_windows, point_windows, labels_aug_base, args, device)
        train_dataset = TensorWindowDataset(seq_aug, point_aug, labels_aug)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler(train_dataset, labels_aug, n_classes, args.class_weight_power, seed),
            num_workers=0,
        )
        train_labels_for_weight = labels_aug
    else:
        train_labels_for_weight = labels[train_mask]
    if args.model == "mrssl":
        pretrain_mrssl(model, train_dataset, args, device, seed)
    if args.model != "adaboost_transformer":
        weight = class_weights(train_labels_for_weight, n_classes, args.class_weight_power).to(device)
        start = time.time()
        train_one_stage(model, train_loader, weight, args, device)
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
    parser.add_argument(
        "--model",
        choices=[
            "att_cnn",
            "recurrent_transformer",
            "reformer",
            "ddpm_mscnn",
            "mffcnn",
            "lmafnet",
            "multimodel_fusion",
            "serial_ensemble",
            "sva_tcn",
            "ssdra",
            "cwscf",
            "drsn_gaf",
            "geology_hybrid",
            "mrssl",
            "adaboost_transformer",
        ],
        default="att_cnn",
    )
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
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--freq-loss-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-temperature", type=float, default=0.2)
    parser.add_argument("--boosting-stages", type=int, default=3)
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
        for name in [
            "att_cnn",
            "recurrent_transformer",
            "reformer",
            "ddpm_mscnn",
            "mffcnn",
            "lmafnet",
            "multimodel_fusion",
            "serial_ensemble",
            "sva_tcn",
            "ssdra",
            "cwscf",
            "drsn_gaf",
            "geology_hybrid",
            "mrssl",
            "adaboost_transformer",
        ]:
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
