# Data Directory

This directory contains dataset preparation, schema auditing, complete-well splitting, and source-only evaluation helpers for ProfileAnchor-Seq. Raw well-log datasets are not redistributed with this repository because they are governed by their original data licenses.

Recommended local layout from the repository root:

```text
datasets/
  force2020/
    train.csv
  external_raw/
    figshare_crosswell_6667646/
      *.csv
  external_processed/
    figshare_crosswell_6667646/
      figshare_crosswell_standard.csv
```

FORCE 2020:

- Competition repository: https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition
- Zenodo record: https://doi.org/10.5281/zenodo.4351156

Figshare cross-well lithology dataset:

- DOI: https://doi.org/10.6084/m9.figshare.6667646.v1

After downloading the Figshare files, build the standardized CSV with:

```bash
python data/prepare_figshare_crosswell_dataset.py
```

The standardized Figshare CSV is written to:

```text
datasets/external_processed/figshare_crosswell_6667646/figshare_crosswell_standard.csv
```

## FORCE 2020

Download the FORCE 2020 lithofacies dataset from the competition repository or Zenodo record, then place `train.csv` under:

```text
datasets/force2020/train.csv
```

The FORCE runners use complete-well spatial splits. Target wells are never used for fitting, release-score calibration, or hyperparameter selection.

Run the main FORCE workflow from the repository root:

```bash
python train.py --dataset force
```

Pass additional runner arguments after `--`:

```bash
python train.py --dataset force -- \
  --data-dir datasets/force2020 \
  --seeds 0 1 2 \
  --coverages 0.01 0.02 0.05 0.10 0.20
```

## Figshare Cross-Well Dataset

Download the Figshare dataset from the DOI above and place the raw CSV files under:

```text
datasets/external_raw/figshare_crosswell_6667646/
```

Build the standardized external CSV:

```bash
python data/prepare_figshare_crosswell_dataset.py \
  --raw-dir datasets/external_raw/figshare_crosswell_6667646 \
  --out-csv datasets/external_processed/figshare_crosswell_6667646/figshare_crosswell_standard.csv
```

Run the Figshare source-only external workflow:

```bash
python train.py --dataset figshare \
  --csv datasets/external_processed/figshare_crosswell_6667646/figshare_crosswell_standard.csv
```

## External CSV Schema

External well-log CSV files must contain:

- one well identifier column, such as `WELL`, `well`, `well_id`, or `Well`;
- one depth column, such as `DEPTH`, `depth`, `DEPT`, or `Depth`;
- one lithology label column, such as `label`, `lithology`, `Lithology`, or `FORCE_2020_LITHOFACIES_LITHOLOGY`;
- at least three numeric log-curve columns.

Typical log columns include `GR`, `RHOB`, `NPHI`, `DTC`, `DTS`, `PEF`, `CALI`, `SP`, `LLD`, `LLS`, and `AC`. Missing curves are allowed if enough numeric logs remain for training and evaluation.

Audit an external file without running a model:

```bash
python train.py --dataset external \
  --csv /path/to/external_lithology.csv \
  --dataset-name new_external \
  --audit-only
```

Run the source-only external gate and evaluation:

```bash
python train.py --dataset external \
  --csv /path/to/external_lithology.csv \
  --dataset-name new_external
```

The external gate checks schema validity before launching model evaluation. If the audit fails, fix the column names, remove nonnumeric log fields, or provide a CSV with complete well, depth, label, and log-curve information.

## Relevant Files

- `prepare_figshare_crosswell_dataset.py`: converts downloaded Figshare files into the standardized external CSV used by the code.
- `audit_external_welllog_dataset.py`: validates well, depth, label, and numeric log columns.
- `run_external_dataset_gate.py`: runs the audit and launches the external ProfileAnchor-Seq workflow when the schema is valid.
- `spatial_multimethod_group_benchmark.py`: provides FORCE loading, feature construction, complete-well splits, and conventional baseline utilities.
