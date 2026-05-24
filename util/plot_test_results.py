
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRIC_LABELS = {
    "Accuracy_mean": "Accuracy",
    "F1_weighted_mean": "Weighted F1",
    "Balanced Accuracy_mean": "Balanced accuracy",
    "F1_macro_mean": "Macro F1",
    "MCC_mean": "MCC",
}


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "coverage" not in df.columns or "method" not in df.columns:
        raise ValueError(f"{path} must contain 'method' and 'coverage' columns")
    return df


def plot_force(force_summary: Path, out_dir: Path) -> None:
    df = load_summary(force_summary)
    if "split" in df.columns:
        df = df[df["split"].astype(str).eq("extrapolation")].copy()
    methods = [
        "ProfileAnchor-Seq",
        "rf",
        "geoshift_seq",
        "cat",
        "xgb",
        "lgbm",
    ]
    labels = {
        "ProfileAnchor-Seq": "ProfileAnchor",
        "rf": "Random forest",
        "geoshift_seq": "Tree-sequence",
        "cat": "CatBoost",
        "xgb": "XGBoost",
        "lgbm": "LightGBM",
    }
    df = df[df["method"].isin(methods)].copy()
    if df.empty:
        raise ValueError(f"No expected FORCE methods found in {force_summary}")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    for ax, metric in zip(axes, ["Accuracy_mean", "F1_weighted_mean"]):
        if metric not in df.columns:
            raise ValueError(f"Missing metric column {metric} in {force_summary}")
        for method in methods:
            sub = df[df["method"].eq(method)].sort_values("coverage")
            if sub.empty:
                continue
            ax.plot(sub["coverage"] * 100, sub[metric], marker="o", linewidth=1.5, label=labels[method])
        ax.set_xlabel("Accepted coverage (%)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        fig.savefig(out_dir / f"force_test_release_curves.{suffix}", bbox_inches="tight", dpi=240)
    plt.close(fig)


def plot_figshare(method_summary: Path, baseline_summary: Path, out_dir: Path) -> None:
    method = load_summary(method_summary)
    baseline = load_summary(baseline_summary)
    df = pd.concat([method, baseline], ignore_index=True, sort=False)
    if "split" in df.columns:
        df = df[df["split"].astype(str).eq("external")].copy()
    methods = [
        "ProfileAnchor-Seq",
        "rf",
        "cat",
        "xgb",
        "lgbm",
        "mlp",
        "STNet-like margin",
        "GCN-like margin",
    ]
    labels = {
        "ProfileAnchor-Seq": "ProfileAnchor",
        "rf": "Random forest",
        "cat": "CatBoost",
        "xgb": "XGBoost",
        "lgbm": "LightGBM",
        "mlp": "MLP",
        "STNet-like margin": "STNet-like",
        "GCN-like margin": "GCN-like",
    }
    df = df[df["method"].isin(methods)].copy()
    if df.empty:
        raise ValueError("No expected Figshare methods found")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    for ax, metric in zip(axes, ["Accuracy_mean", "F1_weighted_mean"]):
        if metric not in df.columns:
            raise ValueError(f"Missing metric column {metric}")
        for method_name in methods:
            sub = df[df["method"].eq(method_name)].sort_values("coverage")
            if sub.empty:
                continue
            ax.plot(sub["coverage"] * 100, sub[metric], marker="o", linewidth=1.5, label=labels[method_name])
        ax.set_xlabel("Accepted coverage (%)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        fig.savefig(out_dir / f"figshare_test_release_curves.{suffix}", bbox_inches="tight", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("test_figures"))
    parser.add_argument("--force-summary", type=Path, default=None)
    parser.add_argument("--figshare-method-summary", type=Path, default=None)
    parser.add_argument("--figshare-baseline-summary", type=Path, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    force_summary = args.force_summary or args.results_dir / "profile_anchor_seq_force_11seed_summary.csv"
    figshare_method = args.figshare_method_summary or args.results_dir / "profile_anchor_seq_figshare_11seed_summary.csv"
    figshare_baselines = (
        args.figshare_baseline_summary
        or args.results_dir / "figshare_structural_external_baselines_11seed_extcov_fast_summary.csv"
    )

    plot_force(force_summary, args.out_dir)
    plot_figshare(figshare_method, figshare_baselines, args.out_dir)
    print(f"Wrote test figures to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
