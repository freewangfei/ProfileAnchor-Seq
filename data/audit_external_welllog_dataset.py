
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


WELL_ALIASES = ["WELL", "well", "Well", "Well Name", "WELL_NAME"]
DEPTH_ALIASES = ["DEPTH_MD", "Depth", "DEPTH", "depth"]
LABEL_ALIASES = ["lithology_name", "Lithology", "LITHOLOGY", "Facies", "TARGET", "label"]
COMMON_LOGS = [
    "GR",
    "RHOB",
    "NPHI",
    "DTC",
    "AC",
    "DTS",
    "PE",
    "PEF",
    "ILD_log10",
    "LLD",
    "LLS",
    "DEN",
    "SP",
    "DeltaPHI",
    "PHIND",
    "CALI",
    "RDEP",
    "RMED",
]
COORDS = ["X_LOC", "Y_LOC", "Z_LOC", "X", "Y", "Z", "LAT", "LON"]


def pick_column(columns: list[str], aliases: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def audit_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    columns = df.columns.tolist()
    well_col = pick_column(columns, WELL_ALIASES)
    depth_col = pick_column(columns, DEPTH_ALIASES)
    label_col = pick_column(columns, LABEL_ALIASES)
    log_cols = [c for c in COMMON_LOGS if c in columns]
    coord_cols = [c for c in COORDS if c in columns]

    rows = [
        {"item": "file", "value": str(path)},
        {"item": "rows", "value": len(df)},
        {"item": "columns", "value": len(columns)},
        {"item": "well_column", "value": well_col or "MISSING"},
        {"item": "depth_column", "value": depth_col or "MISSING"},
        {"item": "label_column", "value": label_col or "MISSING"},
        {"item": "common_log_columns", "value": ",".join(log_cols) if log_cols else "MISSING"},
        {"item": "coordinate_columns", "value": ",".join(coord_cols) if coord_cols else "MISSING"},
    ]

    if well_col:
        wells = df[well_col].dropna().astype(str)
        rows.append({"item": "n_wells", "value": wells.nunique()})
        rows.append({"item": "min_rows_per_well", "value": int(wells.value_counts().min())})
        rows.append({"item": "median_rows_per_well", "value": float(wells.value_counts().median())})
    if label_col:
        labels = df[label_col].dropna().astype(str)
        rows.append({"item": "n_labels", "value": labels.nunique()})
        rows.append({"item": "min_rows_per_label", "value": int(labels.value_counts().min())})
        rows.append({"item": "label_counts", "value": labels.value_counts().to_dict()})
    if depth_col and well_col:
        ordered = (
            df[[well_col, depth_col]]
            .dropna()
            .sort_values([well_col, depth_col])
            .groupby(well_col)[depth_col]
            .apply(lambda s: bool(s.is_monotonic_increasing))
        )
        rows.append({"item": "depth_order_valid_wells", "value": int(ordered.sum())})
        rows.append({"item": "depth_order_total_wells", "value": int(len(ordered))})

    gate = all([well_col, depth_col, label_col]) and len(log_cols) >= 4
    if well_col:
        gate = gate and df[well_col].nunique() >= 5
    if label_col:
        gate = gate and df[label_col].nunique() >= 3
    rows.append({"item": "same_task_schema_gate", "value": "PASS" if gate else "FAIL"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/external_dataset_schema_audit.csv"))
    args = parser.parse_args()
    audit = audit_csv(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.out, index=False)
    print(audit.to_string(index=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
