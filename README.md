# EEG2MM-BENCH

一个统一的多模态 EEG 基准，覆盖 EEG-to-Image、EEG-to-Video、EEG-to-Text
三条生成任务，以及基于 EEGBench 已有轨道的检索评测。框架只实现一遍，EEG
编码器可插拔；基线保持各自固定的编码器。

## 安装

```bash
pip install eeg2mm-bench
pip install eeg2mm-bench[labram]   # LaBraM，需要 braindecode==1.3.2
pip install eeg2mm-bench[luna]     # LUNA，需要 rotary-embedding-torch
```

不同任务与编码器依赖的包不同，按用途分开的环境说明在 `environments/` 下。
仓库不创建环境，用哪个解释器由使用者提供。

## 用法

```bash
python -m eeg2x list
python -m eeg2x env

python -m eeg2x run --dataset thingseeg --model only_clip --encoder luna \
                    --subjects all --stages inference
python -m eeg2x preprocess --dataset nod --encoder labram --subjects sub-01
python -m eeg2x verify --dataset thingseeg --encoder atm_s --subject sub-01

# 从零训练：编码器 + ATM 头，再训练替换扩散先验的 Linear 投影头。
# 已有 checkpoint 时默认跳过，--overwrite 强制重训。
python -m eeg2x run --dataset thingseeg --model only_clip --encoder atm_s \
                    --subjects sub-02 --stages train embed project generate evaluate

# 噪声对照：冻结权重，把测试集 EEG 换成 N(0,1)。
# 产物落在 <干净根>/noise_control/，绝不覆盖干净结果。
python -m eeg2x run --dataset thingseeg --model only_clip --encoder luna \
                    --subjects sub-01 --noise gaussian

# 检索：枚举轨道或从已有预测重算 subject-agg 指标。
python -m eeg2x retrieval --list
python -m eeg2x retrieval --dataset things_eeg --model labram --seed 33 --mode evaluate
```

正式运行要求环境变量 `PYTHONHASHSEED=2000`，全局种子固定为 2000。

## 说明

* 三条生成任务共用同一套编码器实现；检索任务复用 EEGBench 的模型与数据，
  不复制它的模型层。
* 训练与推理产物只写本仓库内的 `artifacts/` 与 `h5/`，保证可追溯。
* LUNA 的逐层衰减优化器只在 ThingsEEG 上生效，其他数据集回落到统一配方；
  这是有意的偏离，原因与代价见 `docs/LUNA_TRAINING_DEVIATION.md`。
* **braindecode 必须是 1.3.2**：1.6 及以上版本的 LaBraM 架构不同，无法加载
  已有的 checkpoint。
* 预处理产物与运行产物必须落在 benchmark 仓库内，这是可追溯性要求。
* 评测报告带 `task_contract` 字段：从产物 HDF5 读出的采样率、通道表、时间窗与
  预处理步骤，以及该任务实际使用的指标与方向。框架的 `run_config.json` 记录同一份契约。
