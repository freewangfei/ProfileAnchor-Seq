# ProfileAnchor-Seq Code

ProfileAnchor-Seq is a source-only cross-well lithofacies identification pipeline for well-log interpretation. It trains on labelled source wells, predicts complete lithofacies profiles in target wells, and ranks target intervals for coverage-controlled automatic release without using target-well labels during fitting, inference, or release selection.

This repository contains the runnable method, reproduced comparison algorithms, data preparation scripts, training entry points, testing entry points, and diagnostic plotting code.

Public repository: https://github.com/freewangfei/ProfileAnchor-Seq

The code supports:

- FORCE 2020 complete-well spatial evaluation.
- Figshare cross-well lithology validation.
- External well-log CSV schema checks and source-only evaluation.
- Reproduced tabular, sequence, spatial-temporal, graph-like, and frequency-style baselines.
- Test figures that summarize release behavior from saved result files.

## Method at a Glance

ProfileAnchor-Seq first predicts a complete lithofacies profile for every target well. It then ranks intervals for automatic release using only source-trained evidence: spatial tree support, local sequence response, heterogeneous profile-anchor agreement, and within-well reliability ranking.

![ProfileAnchor method overview](readme_assets/fig1.pdf)

## Diagnostic Examples

The main output is a complete lithofacies prediction and a coverage-controlled automatic-release subset. The curves below show how accepted accuracy and weighted F1 change as more target-well intervals are released.

![FORCE release curves](readme_assets/fig6.pdf)

The method can also be inspected along well tracks. Accepted intervals concentrate in stable parts of target wells while uncertain transitions remain available for review.

![Target-well lithofacies profiles](readme_assets/fig7.pdf)

External validation uses the Figshare cross-well lithology dataset with a different log suite and label dictionary.

![Figshare release curves](readme_assets/fig9.pdf)

## Installation

Use the project conda environment:

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate yolo
```

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Directory Layout

```text
profile_anchor_code/
  train.py
  test.py
  model/
  data/
  util/
  results/
  test_figures/
  readme_assets/
  requirements.txt
```

`train.py` dispatches the FORCE, Figshare, or generic external training/evaluation workflow.

`test.py` generates diagnostic release-curve figures from saved result summaries.

`model/` contains the ProfileAnchor-Seq implementation and reproduced comparison runners.

- `profile_anchor_reliability_geoshift_seq.py`: main FORCE ProfileAnchor-Seq runner.
- `external_geoshift_rba_fixed_runner.py`: fixed source-only ProfileAnchor runner for Figshare-style external CSVs.
- `external_reliability_budget_anchor_runner.py`: external runner with source-side release-policy selection.
- `external_welllog_profile_anchor_runner.py`: generic external ProfileAnchor and baseline implementation.
- `figshare_structural_external_baselines.py`: reproduced external structural baselines.
- `spatial_lithofacies_selective_geoshift_seq.py`: tree-sequence selective baseline.
- `spatial_lithofacies_stnet_like.py`: STNet-like sequence/spatial-temporal baseline.
- `spatial_lithofacies_stnet_view_fusion.py`: STNet-style view-fusion baseline.
- `spatial_lithofacies_tree_stnet_posterior_fusion.py`: tree and STNet posterior-fusion baseline.
- `spatial_tree_smote_aligned_lithofacies.py`: SMOTE-aligned tree baseline.

`data/` contains dataset preparation, schema checks, FORCE loading, and well-level split utilities.

- `prepare_figshare_crosswell_dataset.py`: converts downloaded Figshare CSV files into a standard external CSV.
- `audit_external_welllog_dataset.py`: checks whether a CSV has well, depth, label, and log columns.
- `run_external_dataset_gate.py`: audits an external CSV and launches the generic external runner if the schema is valid.
- `spatial_multimethod_group_benchmark.py`: FORCE loader, feature construction, well split, and conventional baselines.

`util/` contains reusable metric, selective-release, feature-processing, and diagnostic plotting utilities.

- `selective_multimethod_lithofacies.py`: selective-release metrics and posterior utilities.
- `spatial_lithofacies_feature_ablation_smote.py`: feature selection and SMOTE-related helpers.
- `spatial_lithofacies_feature_view_fusion.py`: feature-view fusion helpers.
- `plot_test_results.py`: creates diagnostic test figures from summary CSV files.
- `benchmark_inference.py`: measures release-score and accepted-set selection throughput from synthetic posterior arrays.

`results/` contains small example summary files used by `test.py --plot-only`. Full training runs write new CSV files here by default. These files are algorithm outputs and diagnostic inputs.

`test_figures/` is the default output directory for diagnostic test figures.

`readme_assets/` contains representative images used in this README.

## Datasets

### FORCE 2020

- GitHub: `https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition`
- Zenodo: `https://doi.org/10.5281/zenodo.4351156`
- Default local path used by the scripts: `datasets/force2020`
- Required file: `train.csv`

### Figshare Cross-Well Lithology Identification

- DOI: `https://doi.org/10.6084/m9.figshare.6667646.v1`
- Expected raw path: `datasets/external_raw/figshare_crosswell_6667646`
- Expected processed CSV: `datasets/external_processed/figshare_crosswell_6667646/figshare_crosswell_standard.csv`

Build the processed Figshare CSV after downloading the raw files:

```bash
python data/prepare_figshare_crosswell_dataset.py
```

The processed external CSV should contain a well ID column, a depth column, a lithology label column, and conventional log curves such as `GR`, `AC`, `DEN`, `PEF`, `LLD`, `LLS`, `SP`, and `CALI`.

## Training

Run the FORCE ProfileAnchor-Seq training and evaluation pipeline:

```bash
python train.py --dataset force
```

Pass model-specific arguments after `--`:

```bash
python train.py --dataset force -- --seeds 0 1 2 --coverages 0.01 0.02 0.05 0.10 0.20
```

Run the Figshare external pipeline:

```bash
python train.py --dataset figshare \
  --csv datasets/external_processed/figshare_crosswell_6667646/figshare_crosswell_standard.csv
```

Audit and run a new external dataset:

```bash
python train.py --dataset external \
  --csv /path/to/external_lithology.csv \
  --dataset-name new_external
```

Audit only:

```bash
python train.py --dataset external \
  --csv /path/to/external_lithology.csv \
  --dataset-name new_external \
  --audit-only
```

## Testing and Figures

Generate diagnostic test figures from saved result summary CSV files:

```bash
python test.py --plot-only
```

Use explicit output paths:

```bash
python test.py --plot-only \
  --results-dir results \
  --out-dir test_figures
```

Generated files:

```text
test_figures/force_test_release_curves.pdf
test_figures/force_test_release_curves.png
test_figures/figshare_test_release_curves.pdf
test_figures/figshare_test_release_curves.png
```

This command uses the small example summaries under `results/` and writes PDF/PNG figures under `test_figures/`.

## Engineering Check

Measure the inference-side release scoring cost with synthetic posterior arrays:

```bash
python util/benchmark_inference.py \
  --intervals 5000 20000 50000 \
  --classes 12 \
  --anchors 5 \
  --coverage 0.05 \
  --repeats 5
```

The command writes `results/inference_benchmark.json` and `results/inference_benchmark.csv`. It measures posterior coupling, anchor-consensus scoring, percentile ranking, and accepted-set selection. It does not train models.

## Reproduced Comparison Runs

FORCE conventional and selective baselines:

```bash
python util/selective_multimethod_lithofacies.py \
  --data-dir datasets/force2020 \
  --seeds 0 1 2 3 4 5 6 7 8 9 42 \
  --coverages 0.01 0.02 0.03 0.05 0.08 0.10 0.20 0.30 0.40 0.50
```

FORCE STNet-like baseline:

```bash
python model/spatial_lithofacies_stnet_like.py \
  --data-dir datasets/force2020 \
  --seeds 0 1 2 3 4 5 6 7 8 9 42 \
  --window 31 \
  --epochs 8 \
  --batch-size 512
```

FORCE STNet view-fusion baseline:

```bash
python model/spatial_lithofacies_stnet_view_fusion.py \
  --data-dir datasets/force2020 \
  --seeds 0 1 2 3 4 5 6 7 8 9 42 \
  --window 31 \
  --epochs 8 \
  --batch-size 512
```

For a quick smoke run, reduce the seed list to `--seeds 0 1 2`. The reported comparisons use the 11-seed complete-well protocol shown above.

Figshare structural baselines:

```bash
python model/figshare_structural_external_baselines.py \
  datasets/external_processed/figshare_crosswell_6667646/figshare_crosswell_standard.csv \
  --manifest results/figshare_complete_well_split_manifest_11seed_unique.csv \
  --logs GR AC DEN PEF LLD LLS SP CALI \
  --coverages 0.01 0.03 0.05 0.08 0.10 0.20 0.40
```

## Direct Method Runners

FORCE main runner:

```bash
python model/profile_anchor_reliability_geoshift_seq.py --help
```

Figshare ProfileAnchor runner:

```bash
python model/external_geoshift_rba_fixed_runner.py \
  datasets/external_processed/figshare_crosswell_6667646/figshare_crosswell_standard.csv
```

## Outputs

Default training outputs are written under `results/`:

```text
*_summary.csv
*_paired.csv
*_manifest.csv
```

Default diagnostic figures are written under `test_figures/`:

```text
force_test_release_curves.pdf
force_test_release_curves.png
figshare_test_release_curves.pdf
figshare_test_release_curves.png
```

## Minimal Checks

Check imports and syntax:

```bash
python -m compileall .
```

Check figure generation:

```bash
python test.py --plot-only
```
