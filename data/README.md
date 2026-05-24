# Data Directory

This directory contains data preparation and loading code. Raw datasets are not redistributed with this repository.

Recommended local layout:

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
