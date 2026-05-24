import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("test_figures"))
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    command = [
        sys.executable,
        "util/plot_test_results.py",
        "--results-dir",
        str(args.results_dir),
        "--out-dir",
        str(args.out_dir),
        *extra,
    ]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
