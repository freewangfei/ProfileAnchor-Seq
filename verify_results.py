from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

METRICS = [
    "Accuracy_mean",
    "F1_weighted_mean",
    "Balanced Accuracy_mean",
    "F1_macro_mean",
    "MCC_mean",
]
RAW_METRICS = [metric.removesuffix("_mean") for metric in METRICS]
FORCE_COVERAGES = {0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50}
FIGSHARE_COVERAGES = {0.01, 0.03, 0.05, 0.08, 0.10, 0.20, 0.40}

RECENT_SUMMARIES = [
    "recent_adaboost_transformer_force_11seed_summary.csv",
    "recent_att_cnn_force_11seed_summary.csv",
    "recent_cwscf_force_11seed_summary.csv",
    "recent_ddpm_mscnn_force_11seed_summary.csv",
    "recent_deepforest_kmeans_smote_force_11seed_summary.csv",
    "recent_drf_de_force_11seed_summary.csv",
    "recent_drsn_gaf_force_11seed_summary.csv",
    "recent_geology_hybrid_force_11seed_summary.csv",
    "recent_graph_feature_force_11seed_summary.csv",
    "recent_integrated_logging_features_force_11seed_summary.csv",
    "recent_lmafnet_force_11seed_summary.csv",
    "recent_meta_information_tensor_force_11seed_summary.csv",
    "recent_mffcnn_force_11seed_summary.csv",
    "recent_mrssl_force_11seed_summary.csv",
    "recent_mscgan_mscnn_force_11seed_summary.csv",
    "recent_multimodel_fusion_force_11seed_summary.csv",
    "recent_pdsmvknn_force_11seed_summary.csv",
    "recent_recurrent_transformer_force_11seed_summary.csv",
    "recent_reformer_force_11seed_summary.csv",
    "recent_resgat_force_11seed_summary.csv",
    "recent_serial_ensemble_force_11seed_summary.csv",
    "recent_ssdra_force_11seed_summary.csv",
    "recent_sva_tcn_force_11seed_summary.csv",
]
STRUCTURAL_COMPARISON_SUMMARY = "profile_anchor_coverage_conditioned_guard_11seed_summary.csv"
STRUCTURAL_COMPARISON_METHODS = {
    "cat",
    "geoshift_seq",
    "lgbm",
    "rf",
    "xgb",
}
PROMOTED_PROFILE_ANCHOR_METHOD = "profile_anchor_coverage_conditioned_boundary_guard"


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, name: str, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise AssertionError(f"{name}: missing columns {missing}")


def require_coverages(df: pd.DataFrame, name: str, split: str, coverages: set[float]) -> None:
    require_columns(df, name, ["split", "coverage"])
    sub = df[df["split"].astype(str).eq(split)]
    observed = {round(float(value), 2) for value in sub["coverage"].dropna().unique()}
    missing = sorted(round(value, 2) for value in coverages.difference(observed))
    if missing:
        raise AssertionError(f"{name}: missing {split} coverages {missing}")


def check_recent_summaries() -> None:
    for name in RECENT_SUMMARIES:
        df = read_csv(name)
        require_columns(df, name, ["method", "split", "coverage", *METRICS])
        require_coverages(df, name, "extrapolation", FORCE_COVERAGES)
    print(f"PASS recent same-protocol summaries ({len(RECENT_SUMMARIES)})")


def check_force_profile_anchor() -> None:
    name = "profile_anchor_coverage_conditioned_guard_11seed_summary.csv"
    df = read_csv(name)
    method = PROMOTED_PROFILE_ANCHOR_METHOD
    require_columns(df, name, ["method", "split", "coverage", *METRICS])
    require_coverages(df, name, "extrapolation", FORCE_COVERAGES)
    sub = df[df["split"].astype(str).eq("extrapolation") & df["method"].astype(str).eq(method)]
    observed = {round(float(value), 2) for value in sub["coverage"].dropna().unique()}
    missing = sorted(round(value, 2) for value in FORCE_COVERAGES.difference(observed))
    if missing:
        raise AssertionError(f"{name}: promoted method missing coverages {missing}")

    envelope = {coverage: {metric: -1.0 for metric in METRICS} for coverage in FORCE_COVERAGES}
    for name in RECENT_SUMMARIES:
        recent = read_csv(name)
        recent = recent[recent["split"].astype(str).eq("extrapolation")].copy()
        if recent["method"].astype(str).str.endswith("_margin").any():
            recent = recent[recent["method"].astype(str).str.endswith("_margin")]
        for _, row in recent.iterrows():
            coverage = round(float(row["coverage"]), 2)
            if coverage not in envelope:
                continue
            for metric in METRICS:
                envelope[coverage][metric] = max(envelope[coverage][metric], float(row[metric]))

    structural = read_csv(STRUCTURAL_COMPARISON_SUMMARY)
    require_columns(structural, STRUCTURAL_COMPARISON_SUMMARY, ["method", "split", "coverage", *METRICS])
    forbidden_profile_anchor = sorted(
        value
        for value in structural["method"].astype(str).unique()
        if value.startswith("profile_anchor_") and value != PROMOTED_PROFILE_ANCHOR_METHOD
    )
    if forbidden_profile_anchor:
        print(
            "PASS internal ProfileAnchor candidates excluded from formal envelope: "
            + ", ".join(forbidden_profile_anchor)
        )
    structural = structural[
        structural["split"].astype(str).eq("extrapolation")
        & structural["method"].astype(str).isin(STRUCTURAL_COMPARISON_METHODS)
    ].copy()
    for method in STRUCTURAL_COMPARISON_METHODS:
        method_rows = structural[structural["method"].astype(str).eq(method)]
        observed = {round(float(value), 2) for value in method_rows["coverage"].dropna().unique()}
        missing = sorted(round(value, 2) for value in FORCE_COVERAGES.difference(observed))
        if missing:
            raise AssertionError(f"{STRUCTURAL_COMPARISON_SUMMARY}: {method} missing coverages {missing}")
    for _, row in structural.iterrows():
        coverage = round(float(row["coverage"]), 2)
        if coverage not in envelope:
            continue
        for metric in METRICS:
            envelope[coverage][metric] = max(envelope[coverage][metric], float(row[metric]))

    for coverage, thresholds in sorted(envelope.items()):
        row = sub[sub["coverage"].round(2).eq(round(coverage, 2))]
        if row.empty:
            raise AssertionError(f"{name}: promoted method missing coverage {coverage}")
        row = row.iloc[0]
        for metric, threshold in thresholds.items():
            value = float(row[metric])
            if value + 5e-4 < threshold:
                raise AssertionError(
                    f"ProfileAnchor envelope check failed at {coverage:.0%} {metric}: "
                    f"{value:.4f} < {threshold:.4f}"
                )
    print("PASS FORCE ProfileAnchor formal comparison envelope")


def check_external_results() -> None:
    method_name = "profile_anchor_seq_figshare_11seed_summary.csv"
    method = read_csv(method_name)
    require_columns(method, method_name, ["method", "split", "coverage", *METRICS])
    require_coverages(method, method_name, "external", FIGSHARE_COVERAGES)

    baseline_name = "figshare_structural_external_baselines_11seed_extcov_fast_summary.csv"
    baseline = read_csv(baseline_name)
    require_columns(baseline, baseline_name, ["method", "split", "coverage", *METRICS])
    require_coverages(baseline, baseline_name, "external", FIGSHARE_COVERAGES)
    print("PASS Figshare external summaries")


def check_plot_inputs() -> None:
    for name in [
        "profile_anchor_seq_force_11seed_summary.csv",
        "profile_anchor_seq_figshare_11seed_summary.csv",
        "figshare_structural_external_baselines_11seed_extcov_fast_summary.csv",
    ]:
        df = read_csv(name)
        require_columns(df, name, ["method", "coverage", "Accuracy_mean", "F1_weighted_mean"])
    print("PASS diagnostic plotting inputs")


def main() -> None:
    check_recent_summaries()
    check_force_profile_anchor()
    check_external_results()
    check_plot_inputs()
    print("PASS packaged result integrity")


if __name__ == "__main__":
    main()
