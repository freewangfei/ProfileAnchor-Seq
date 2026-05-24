
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np


def normalized(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 1e-12, None)
    return x / x.sum(axis=1, keepdims=True)


def margin(p: np.ndarray) -> np.ndarray:
    part = np.partition(np.asarray(p), -2, axis=1)
    return part[:, -1] - part[:, -2]


def rank_percentile(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(score.size, dtype=np.float64)
    denom = max(score.size - 1, 1)
    return ranks / denom


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = normalized(p)
    q = normalized(q)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * (np.log(p + 1e-12) - np.log(m + 1e-12)), axis=1) + 0.5 * np.sum(
        q * (np.log(q + 1e-12) - np.log(m + 1e-12)), axis=1
    )


def synthetic_posteriors(n: int, c: int, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    tree = rng.dirichlet(np.full(c, 0.8), size=n)
    sequence = rng.dirichlet(np.full(c, 0.8), size=n)
    anchors = [rng.dirichlet(np.full(c, 0.8), size=n) for _ in range(k)]
    return tree, sequence, anchors


def release_score(tree: np.ndarray, sequence: np.ndarray, anchors: list[np.ndarray], tree_weight: float) -> np.ndarray:
    coupled = normalized(tree_weight * tree + (1.0 - tree_weight) * sequence)
    anchor_pool = normalized(np.mean(np.stack(anchors, axis=0), axis=0))
    anchor_margins = np.stack([margin(p) for p in anchors], axis=1)
    anchor_preds = np.stack([p.argmax(axis=1) for p in anchors], axis=1)
    vote_share = np.empty(coupled.shape[0], dtype=np.float64)
    for i in range(coupled.shape[0]):
        vote_share[i] = np.bincount(anchor_preds[i], minlength=coupled.shape[1]).max() / len(anchors)
    agreement = (coupled.argmax(axis=1) == anchor_pool.argmax(axis=1)).astype(np.float64)
    min_margin = np.minimum.reduce([margin(coupled), margin(anchor_pool), anchor_margins.min(axis=1)])
    disagreement = rank_percentile(js_divergence(coupled, anchor_pool))
    robust_anchor = 0.45 * margin(coupled) + 0.20 * min_margin + 0.15 * vote_share + 0.10 * agreement - 0.10 * disagreement
    rank_fusion = np.mean(
        np.stack(
            [
                rank_percentile(margin(coupled)),
                rank_percentile(margin(tree)),
                rank_percentile(margin(sequence)),
                rank_percentile(margin(anchor_pool)),
                rank_percentile(anchor_margins.mean(axis=1)),
                rank_percentile(anchor_margins.min(axis=1)),
                rank_percentile(vote_share),
                rank_percentile(agreement),
                rank_percentile(-js_divergence(coupled, anchor_pool)),
            ],
            axis=1,
        ),
        axis=1,
    )
    return 0.50 * rank_percentile(rank_fusion) + 0.50 * rank_percentile(robust_anchor)


def accepted_indices(score: np.ndarray, coverage: float) -> np.ndarray:
    n = score.size
    keep = max(1, int(round(n * coverage)))
    if keep >= n:
        return np.argsort(score)[::-1]
    idx = np.argpartition(score, n - keep)[n - keep :]
    return idx[np.argsort(score[idx])[::-1]]


def time_run(n: int, c: int, k: int, coverage: float, repeats: int, seed: int, tree_weight: float) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    tree, sequence, anchors = synthetic_posteriors(n, c, k, rng)
    score = release_score(tree, sequence, anchors, tree_weight)
    accepted_indices(score, coverage)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        score = release_score(tree, sequence, anchors, tree_weight)
        keep = accepted_indices(score, coverage)
        elapsed = time.perf_counter() - start
        if keep.size == 0:
            raise RuntimeError("empty accepted set")
        times.append(elapsed)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "target_intervals": n,
        "classes": c,
        "anchor_models": k,
        "coverage": coverage,
        "repeats": repeats,
        "mean_seconds": float(arr.mean()),
        "std_seconds": float(arr.std(ddof=0)),
        "intervals_per_second": float(n / arr.mean()),
        "accepted_intervals": int(max(1, round(n * coverage))),
    }


def write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", nargs="+", type=int, default=[5000, 20000, 50000])
    parser.add_argument("--classes", type=int, default=12)
    parser.add_argument("--anchors", type=int, default=5)
    parser.add_argument("--coverage", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tree-weight", type=float, default=0.75)
    parser.add_argument("--out-json", type=Path, default=Path("results/inference_benchmark.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("results/inference_benchmark.csv"))
    args = parser.parse_args()

    rows = [
        time_run(n, args.classes, args.anchors, args.coverage, args.repeats, args.seed + i, args.tree_weight)
        for i, n in enumerate(args.intervals)
    ]
    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
        },
        "benchmark": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(rows, args.out_csv)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
