
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from model.external_reliability_budget_anchor_runner import (
    Policy,
    fit_models,
    load_external,
    model_proba,
    parse_policy,
    policy_rows,
    posterior_bank,
    score_bank,
    smooth_by_well,
    source_prior,
)
from model.external_welllog_profile_anchor_runner import metric_row, selective_rows, split_wells, summarize


METHOD = "ProfileAnchor-Seq"
DEFAULT_SCHEDULE = {
    0.01: "rba_hgb_bal|consensus_margin|uniform|x2.2|m0.08",
    0.03: "rba_hgb_bal|margin|uniform|x2.2|m0.08",
    0.05: "rba_hgb_bal|margin|uniform|x2.2|m0.08",
    0.08: "rba_pool_all|margin|uniform|x2.2|m0.08",
    0.10: "rba_pool_all|margin|uniform|x2.2|m0.08",
    0.20: "rba_pool_all|margin|uniform|x2.2|m0.08",
    0.40: "rba_hgb_bal|margin|global|x1.0|m0.00",
}


def scheduled_policy(coverage: float) -> Policy:
    key = round(float(coverage), 2)
    try:
        return parse_policy(DEFAULT_SCHEDULE[key])
    except KeyError as exc:
        known = ", ".join(f"{c:.2f}" for c in sorted(DEFAULT_SCHEDULE))
        raise ValueError(f"No ProfileAnchor-Seq policy is defined for coverage {coverage}; known: {known}") from exc


def run_seed(df: pd.DataFrame, features: list[str], n_classes: int, seed: int, args):
    train_wells, test_wells = split_wells(df, seed, args.test_wells)
    source = df[df["WELL"].isin(train_wells)].copy()
    target = df[df["WELL"].isin(test_wells)].copy()

    models, classes = fit_models(source, features, n_classes, seed, args)
    post = {
        name: smooth_by_well(
            target,
            model_proba(model, target[features], classes, n_classes),
            args.smooth_window,
            args.smooth_blend,
        )
        for name, model in models.items()
    }
    banks = posterior_bank(post)
    prior = source_prior(source, n_classes, args.prior_smoothing)
    y = target["TARGET"].to_numpy(dtype=int)

    rows = []
    for name, p in post.items():
        rows.extend(selective_rows(y, p, seed, "external", name, args.coverages))
        full = metric_row(y, p.argmax(axis=1))
        full.update({"seed": seed, "split": "external", "method": name, "coverage": 1.0, "kept_rows": len(y)})
        rows.append(full)

    manifest_rows = []
    for coverage in args.coverages:
        policy = scheduled_policy(coverage)
        if policy.posterior not in banks:
            raise RuntimeError(f"Posterior {policy.posterior} is unavailable for seed {seed}")
        p = banks[policy.posterior]
        scores = score_bank(post, p)
        if policy.score not in scores:
            raise RuntimeError(f"Score {policy.score} is unavailable for seed {seed}")
        rows.extend(policy_rows(y, p, scores[policy.score], seed, METHOD, [coverage], prior, policy))
        manifest_rows.append(
            {
                "seed": seed,
                "coverage": coverage,
                "train_wells": ",".join(sorted(train_wells)),
                "test_wells": ",".join(sorted(test_wells)),
                "fixed_policy": DEFAULT_SCHEDULE[round(float(coverage), 2)],
            }
        )
    return rows, manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--dataset-name", default="profile_anchor_seq_figshare_11seed")
    parser.add_argument("--well-col", default="WELL")
    parser.add_argument("--depth-col", default="DEPTH_MD")
    parser.add_argument("--label-col", default="LITHOLOGY")
    parser.add_argument("--logs", nargs="+", default=["GR", "AC", "DEN", "PEF", "LLD", "LLS", "SP", "CALI"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42])
    parser.add_argument("--test-wells", type=int, default=3)
    parser.add_argument("--coverages", nargs="+", type=float, default=sorted(DEFAULT_SCHEDULE))
    parser.add_argument("--models", nargs="+", default=["rf_deep", "et_deep", "hgb_bal", "cat", "xgb", "lgbm"])
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--smooth-blend", type=float, default=0.82)
    parser.add_argument("--prior-smoothing", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--gbdt-max-iter", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-max-iter", type=int, default=4000)
    parser.add_argument("--knn-neighbors", type=int, default=15)
    parser.add_argument("--mlp-max-iter", type=int, default=60)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    args = parser.parse_args()

    out_base = Path("results")
    args.out_csv = args.out_csv or out_base / f"{args.dataset_name}.csv"
    args.summary_csv = args.summary_csv or out_base / f"{args.dataset_name}_summary.csv"
    args.manifest_csv = args.manifest_csv or out_base / f"{args.dataset_name}_manifest.csv"

    df, features, class_names = load_external(args.csv, args.well_col, args.depth_col, args.label_col, args.logs)
    rows = []
    manifests = []
    for seed in args.seeds:
        seed_rows, manifest_rows = run_seed(df, features, len(class_names), seed, args)
        rows.extend(seed_rows)
        manifests.extend(manifest_rows)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        pd.DataFrame(manifests).to_csv(args.manifest_csv, index=False)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    summary.to_csv(args.summary_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.manifest_csv}")


if __name__ == "__main__":
    main()
