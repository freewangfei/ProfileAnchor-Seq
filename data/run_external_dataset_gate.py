
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from data.audit_external_welllog_dataset import audit_csv


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
RUNNER = ROOT / "model" / "external_welllog_profile_anchor_runner.py"


def audit_value(audit: pd.DataFrame, item: str) -> str:
    hit = audit.loc[audit["item"] == item, "value"]
    if hit.empty:
        return "MISSING"
    return str(hit.iloc[0])


def split_csv_value(value: str) -> list[str]:
    if not value or value == "MISSING":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def build_runner_command(args: argparse.Namespace, audit: pd.DataFrame) -> list[str]:
    well_col = audit_value(audit, "well_column")
    depth_col = audit_value(audit, "depth_column")
    label_col = audit_value(audit, "label_column")
    logs = args.logs or split_csv_value(audit_value(audit, "common_log_columns"))
    if len(logs) < args.min_logs:
        raise RuntimeError(f"Need at least {args.min_logs} detected log columns; found {logs}")

    cmd = [
        str(PYTHON),
        str(RUNNER),
        str(args.csv),
        "--dataset-name",
        args.dataset_name,
        "--well-col",
        well_col,
        "--depth-col",
        depth_col,
        "--label-col",
        label_col,
        "--logs",
        *logs,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--test-wells",
        str(args.test_wells),
        "--coverages",
        *[str(cov) for cov in args.coverages],
        "--models",
        *args.models,
        "--anchor-models",
        *args.anchor_models,
        "--anchor-weights",
        *[str(weight) for weight in args.anchor_weights],
        "--release-scores",
        *args.release_scores,
        "--n-estimators",
        str(args.n_estimators),
        "--gbdt-max-iter",
        str(args.gbdt_max_iter),
        "--max-depth",
        str(args.max_depth),
        "--learning-rate",
        str(args.learning_rate),
        "--n-jobs",
        str(args.n_jobs),
        "--mlp-max-iter",
        str(args.mlp_max_iter),
    ]
    if args.disable_gbdt_early_stopping:
        cmd.append("--disable-gbdt-early-stopping")
    if args.out_csv:
        cmd.extend(["--out-csv", str(args.out_csv)])
    if args.summary_csv:
        cmd.extend(["--summary-csv", str(args.summary_csv)])
    if args.paired_csv:
        cmd.extend(["--paired-csv", str(args.paired_csv)])
    if args.split_manifest_csv:
        cmd.extend(["--split-manifest-csv", str(args.split_manifest_csv)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--audit-out", type=Path, default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--min-logs", type=int, default=4)
    parser.add_argument("--logs", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--test-wells", type=int, default=3)
    parser.add_argument("--coverages", nargs="+", type=float, default=[0.01, 0.03, 0.05, 0.08, 0.10, 0.20, 0.40])
    parser.add_argument("--models", nargs="+", default=["rf", "xgb", "lgbm", "cat", "mlp"])
    parser.add_argument("--anchor-models", nargs="+", default=["rf", "xgb", "lgbm", "cat", "mlp"])
    parser.add_argument("--anchor-weights", nargs="+", type=float, default=[0.44])
    parser.add_argument(
        "--release-scores",
        nargs="+",
        default=[
            "external_profile_anchor",
            "rank_fusion_mean",
            "rank_fusion_trimmed",
            "robust_anchor_margin",
            "second_stage_rank_fusion_mean_x_robust_anchor_margin",
            "second_stage_weighted_rank_fusion_mean_x_robust_anchor_margin",
        ],
    )
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--gbdt-max-iter", type=int, default=120)
    parser.add_argument("--disable-gbdt-early-stopping", action="store_true")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--mlp-max-iter", type=int, default=160)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--paired-csv", type=Path, default=None)
    parser.add_argument("--split-manifest-csv", type=Path, default=None)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)
    if not RUNNER.exists():
        raise FileNotFoundError(RUNNER)

    audit = audit_csv(args.csv)
    audit_out = args.audit_out or ROOT / "results" / f"{args.dataset_name}_schema_gate.csv"
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_out, index=False)
    print(audit.to_string(index=False))
    print(f"Wrote {audit_out}")

    if audit_value(audit, "same_task_schema_gate") != "PASS":
        print("Schema gate failed; external runner was not launched.", file=sys.stderr)
        return 1
    if args.audit_only:
        print("Schema gate passed; audit-only mode requested.")
        return 0

    cmd = build_runner_command(args, audit)
    print("Launching external runner:")
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
