
from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_DIR = Path("datasets/external_raw/figshare_crosswell_6667646")
OUT_DIR = Path("datasets/external_processed/figshare_crosswell_6667646")
OUT_CSV = OUT_DIR / "figshare_crosswell_standard.csv"
SUMMARY_CSV = OUT_DIR / "figshare_crosswell_schema_summary.csv"


RENAME = {
    "TopDepth": "TOP_DEPTH",
    "BotDepth": "BOT_DEPTH",
    "_CAL": "CALI",
    "_GR": "GR",
    "_SP": "SP",
    "_LLD": "LLD",
    "_LLS": "LLS",
    "_AC": "AC",
    "_DEN": "DEN",
    "_PEF": "PEF",
    "Lith_Section": "LITHOLOGY",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        if path.name.startswith("figshare_"):
            continue
        df = pd.read_csv(path).rename(columns=RENAME)
        df["WELL"] = path.stem
        df["DEPTH_MD"] = 0.5 * (df["TOP_DEPTH"] + df["BOT_DEPTH"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No raw CSV files found in {RAW_DIR}")

    out = pd.concat(frames, ignore_index=True)
    cols = [
        "WELL",
        "DEPTH_MD",
        "TOP_DEPTH",
        "BOT_DEPTH",
        "CALI",
        "GR",
        "SP",
        "LLD",
        "LLS",
        "AC",
        "DEN",
        "PEF",
        "LITHOLOGY",
    ]
    out = out[cols].sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    per_well = out.groupby("WELL").agg(rows=("WELL", "size"), top=("DEPTH_MD", "min"), bottom=("DEPTH_MD", "max"))
    label_counts = out["LITHOLOGY"].value_counts().rename_axis("lithology").reset_index(name="rows")
    summary_rows = [
        {"item": "rows", "value": len(out)},
        {"item": "wells", "value": out["WELL"].nunique()},
        {"item": "labels", "value": out["LITHOLOGY"].nunique()},
        {"item": "logs", "value": "CALI,GR,SP,LLD,LLS,AC,DEN,PEF"},
        {"item": "min_rows_per_well", "value": int(per_well["rows"].min())},
        {"item": "median_rows_per_well", "value": float(per_well["rows"].median())},
        {"item": "label_counts", "value": label_counts.to_dict(orient="records")},
    ]
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(out)} rows, {out['WELL'].nunique()} wells)")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
