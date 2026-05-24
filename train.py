import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(args: list[str]) -> int:
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["force", "figshare", "external"], default="force")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.dataset == "force":
        return run(["model/profile_anchor_reliability_geoshift_seq.py", *extra])

    if args.dataset == "figshare":
        csv = args.csv or Path(
            "datasets/external_processed/figshare_crosswell_6667646/figshare_crosswell_standard.csv"
        )
        command = ["model/external_geoshift_rba_fixed_runner.py", str(csv)]
        if args.dataset_name:
            command.extend(["--dataset-name", args.dataset_name])
        return run([*command, *extra])

    if args.csv is None:
        raise SystemExit("--csv is required for --dataset external")
    command = ["data/run_external_dataset_gate.py", str(args.csv), "--dataset-name", args.dataset_name or "external"]
    if args.audit_only:
        command.append("--audit-only")
    return run([*command, *extra])


if __name__ == "__main__":
    raise SystemExit(main())
