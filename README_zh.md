# ProfileAnchor-Seq 中文说明

ProfileAnchor-Seq 是一个面向常规测井曲线的跨井岩性识别代码库。它从带标签的源井训练模型，对目标井输出完整岩性剖面，对目标井层段进行覆盖率可控的自动释放排序，并提供数据处理、训练、测试、对比运行、结果检查和诊断图生成脚本。

本仓库包含：

- ProfileAnchor-Seq 主方法；
- 训练和测试入口；
- FORCE 2020 与 Figshare 数据处理工具；
- 对比算法运行脚本；
- 用于快速检查的结果摘要文件；
- 诊断图和推理耗时测试工具。

## 代码功能

ProfileAnchor-Seq 结合三类信息：

- 缺失感知的空间树模型支持；
- 局部深度窗口中的序列响应；
- 由多个异构模型形成的 profile-anchor 一致性约束。

模型输出包括完整目标井岩性预测，以及在给定覆盖率下可自动释放的层段子集。

![方法概览](readme_assets/method_overview.png)

以下图片展示方法流程和保存结果生成的诊断输出。

![FORCE 释放曲线](readme_assets/force_release_curves.png)

![目标井剖面轨迹](readme_assets/well_profile_tracks.png)

![跨井迁移诊断](readme_assets/transfer_diagnostics.png)

![Figshare 释放曲线](readme_assets/figshare_release_curves.png)

## 环境配置

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate your_env
pip install -r requirements.txt
```

代码可以在 CPU 上运行。完整 11 个随机种子的运行耗时较长，快速检查时可以先减少种子数量。

## 目录结构

```text
profile_anchor_code/
  train.py
  test.py
  verify_results.py
  model/
  data/
  util/
  results/
  readme_assets/
  requirements.txt
  REPRODUCIBILITY.md
```

- `train.py`：统一训练入口，分发 FORCE、Figshare 和通用外部 CSV 流程。
- `test.py`：从保存的结果摘要生成诊断图。
- `verify_results.py`：检查结果文件和必要覆盖率。
- `model/`：ProfileAnchor-Seq 主方法和对比算法。
- `data/`：数据集准备、格式检查、FORCE 加载和井级划分工具。
- `util/`：选择性释放指标、特征处理、绘图和推理耗时测试。

## 数据集

FORCE 2020：

- GitHub：`https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition`
- Zenodo：`https://doi.org/10.5281/zenodo.4351156`
- 默认本地路径：`datasets/force2020`
- 必需文件：`train.csv`

Figshare Cross-Well Lithology Identification：

- DOI：`https://doi.org/10.6084/m9.figshare.6667646.v1`
- 原始数据路径：`datasets/external_raw/figshare_crosswell_6667646`
- 处理后 CSV：`datasets/external_processed/figshare_crosswell_6667646/figshare_crosswell_standard.csv`

下载原始文件后，运行：

```bash
python data/prepare_figshare_crosswell_dataset.py
```

通用外部 CSV 至少应包含井名列、深度列、岩性标签列和常规测井曲线列。可先运行格式检查：

```bash
python data/audit_external_welllog_dataset.py /path/to/external_lithology.csv
```

## 训练主方法

运行 FORCE 主方法：

```bash
python train.py --dataset force
```

在 `--` 后传入模型参数：

```bash
python train.py --dataset force -- \
  --data-dir datasets/force2020 \
  --seeds 0 1 2 \
  --coverages 0.01 0.02 0.05 0.10 0.20
```

运行 Figshare 数据集：

```bash
python train.py --dataset figshare \
  --csv datasets/external_processed/figshare_crosswell_6667646/figshare_crosswell_standard.csv
```

运行新的外部数据集：

```bash
python train.py --dataset external \
  --csv /path/to/external_lithology.csv \
  --dataset-name external_case
```

只检查数据格式，不训练：

```bash
python train.py --dataset external \
  --csv /path/to/external_lithology.csv \
  --dataset-name external_case \
  --audit-only
```

## 测试和诊断图

从已有结果摘要生成诊断图：

```bash
python test.py --plot-only
```

指定输入和输出路径：

```bash
python test.py --plot-only \
  --results-dir results \
  --out-dir test_figures
```

生成文件：

```text
test_figures/force_test_release_curves.pdf
test_figures/force_test_release_curves.png
test_figures/figshare_test_release_curves.pdf
test_figures/figshare_test_release_curves.png
```

检查打包结果：

```bash
python verify_results.py
```

## 运行对比方法

比较不同方法时，建议使用相同的数据路径、种子列表和覆盖率。标准覆盖率为：

```text
0.01 0.02 0.03 0.05 0.08 0.10 0.20 0.30 0.40 0.50
```

运行主方法：

```bash
python model/profile_anchor_reliability_geoshift_seq.py \
  --data-dir datasets/force2020 \
  --seeds 0 1 2 3 4 5 6 7 8 9 42 \
  --coverages 0.01 0.02 0.03 0.05 0.08 0.10 0.20 0.30 0.40 0.50
```

运行常规和选择性基线：

```bash
python util/selective_multimethod_lithofacies.py \
  --data-dir datasets/force2020 \
  --max-rows-per-well 800 \
  --seeds 0 1 2 3 4 5 6 7 8 9 42 \
  --coverages 0.01 0.02 0.03 0.05 0.08 0.10 0.20 0.30 0.40 0.50
```

运行 STNet-like 基线：

```bash
python model/spatial_lithofacies_stnet_like.py \
  --data-dir datasets/force2020 \
  --max-rows-per-well 800 \
  --seeds 0 1 2 3 4 5 6 7 8 9 42 \
  --window 31 \
  --epochs 8 \
  --batch-size 512

python model/spatial_lithofacies_stnet_view_fusion.py \
  --data-dir datasets/force2020 \
  --max-rows-per-well 800 \
  --seeds 0 1 2 3 4 5 6 7 8 9 42 \
  --window 31 \
  --epochs 8 \
  --batch-size 512
```

岩性识别机制族由 `model/recent_lithology_baselines.py` 运行：

```bash
python model/recent_lithology_baselines.py \
  --model att_cnn \
  --data-dir datasets/force2020 \
  --max-rows-per-well 800 \
  --seeds 0 1 2 3 4 5 6 7 8 9 42 \
  --coverages 0.01 0.02 0.03 0.05 0.08 0.10 0.20 0.30 0.40 0.50
```

可选 `--model` 包括：

```text
att_cnn
recurrent_transformer
reformer
adaboost_transformer
mrssl
geology_hybrid
drsn_gaf
sva_tcn
cwscf
ssdra
serial_ensemble
lmafnet
multimodel_fusion
mffcnn
ddpm_mscnn
```

其他对比方法：

```bash
python model/integrated_logging_features_baseline.py --data-dir datasets/force2020
python model/meta_information_tensor_baseline.py --data-dir datasets/force2020
python model/deepforest_kmeans_smote_baseline.py --data-dir datasets/force2020
python model/recent_graph_attention_baseline.py --data-dir datasets/force2020
python model/graph_feature_extraction_baseline.py --data-dir datasets/force2020
python model/recent_mscgan_baseline.py --data-dir datasets/force2020
python model/recent_drf_de_baseline.py --data-dir datasets/force2020
python model/recent_pdsmvknn_baseline.py --data-dir datasets/force2020
```

Figshare 结构化基线：

```bash
python model/figshare_structural_external_baselines.py \
  datasets/external_processed/figshare_crosswell_6667646/figshare_crosswell_standard.csv \
  --manifest results/figshare_complete_well_split_manifest_11seed_unique.csv \
  --logs GR AC DEN PEF LLD LLS SP CALI \
  --coverages 0.01 0.03 0.05 0.08 0.10 0.20 0.40
```

## 正确性检查

编译 Python 文件：

```bash
python -m compileall .
```

检查结果文件：

```bash
python verify_results.py
```

检查测试图生成：

```bash
python test.py --plot-only
```

运行无需 FORCE 数据集的自检：

```bash
python model/integrated_logging_features_baseline.py --self-check
python model/recent_drf_de_baseline.py --self-check
python model/deepforest_kmeans_smote_baseline.py --self-check
python model/meta_information_tensor_baseline.py --self-check
python model/graph_feature_extraction_baseline.py --self-check
python model/recent_lithology_baselines.py --self-check
python model/recent_graph_attention_baseline.py --self-check
python model/recent_mscgan_baseline.py --self-check
```

运行推理侧释放评分耗时测试：

```bash
python util/benchmark_inference.py \
  --intervals 1000 2000 \
  --classes 12 \
  --anchors 5 \
  --coverage 0.05 \
  --repeats 2 \
  --out-json test_figures/inference_benchmark_smoke.json \
  --out-csv test_figures/inference_benchmark_smoke.csv
```

## 主要文件

主方法：

- `model/profile_anchor_reliability_geoshift_seq.py`

外部数据集：

- `model/external_geoshift_rba_fixed_runner.py`
- `model/external_reliability_budget_anchor_runner.py`
- `model/external_welllog_profile_anchor_runner.py`
- `data/run_external_dataset_gate.py`

对比方法：

- `util/selective_multimethod_lithofacies.py`
- `model/spatial_lithofacies_selective_geoshift_seq.py`
- `model/spatial_lithofacies_stnet_like.py`
- `model/spatial_lithofacies_stnet_view_fusion.py`
- `model/spatial_lithofacies_tree_stnet_posterior_fusion.py`
- `model/spatial_tree_smote_aligned_lithofacies.py`
- `model/recent_lithology_baselines.py`
- `model/recent_graph_attention_baseline.py`
- `model/recent_mscgan_baseline.py`
- `model/recent_drf_de_baseline.py`
- `model/recent_pdsmvknn_baseline.py`
- `model/deepforest_kmeans_smote_baseline.py`
- `model/graph_feature_extraction_baseline.py`
- `model/integrated_logging_features_baseline.py`
- `model/meta_information_tensor_baseline.py`

工具：

- `data/spatial_multimethod_group_benchmark.py`
- `data/prepare_figshare_crosswell_dataset.py`
- `data/audit_external_welllog_dataset.py`
- `util/spatial_lithofacies_feature_ablation_smote.py`
- `util/spatial_lithofacies_feature_view_fusion.py`
- `util/plot_test_results.py`
- `util/benchmark_inference.py`
