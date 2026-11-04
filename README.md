# EEG2MM-BENCH

统一框架，编码器可插拔；基线编码器固定。当前覆盖 EEG-to-Image 重建。

A unified EEG-to-Image reconstruction framework. The framework is implemented
once and the EEG encoder is swappable (ATM-S, LaBraM, REVE, LUNA, ENIGMA);
baselines (complete ATM, AVDE, ENIGMA) keep a fixed encoder by construction.

## 安装

```bash
pip install eeg2x-benchmark
pip install eeg2x-benchmark[labram]   # LaBraM，需要 braindecode==1.3.2
pip install eeg2x-benchmark[luna]     # LUNA，需要 rotary-embedding-torch
```

## 用法

```bash
python -m eeg2x list
python -m eeg2x run --dataset thingseeg --model only_clip --encoder luna \
                    --subjects all --stages inference
python -m eeg2x preprocess --dataset nod --encoder labram --subjects sub-01
python -m eeg2x verify --dataset thingseeg --encoder atm_s --subject sub-01

# 从零训练：编码器 + ATM 头，再训练替换扩散先验的 Linear 投影头。
# 已有 checkpoint 时默认跳过，--overwrite 强制重训。
python -m eeg2x run --dataset thingseeg --model only_clip --encoder atm_s \
                    --subjects sub-02 --stages train embed project generate evaluate
python -m eeg2x run ... --stages train --epochs 2 --projection-epochs 5   # 冒烟测试

# 噪声对照：冻结权重，把测试集 EEG 换成 N(0,1)。
# 产物落在 <干净根>/noise_control/，绝不覆盖干净结果；--distinct-images 是省时档。
python -m eeg2x run --dataset thingseeg --model only_clip --encoder luna \
                    --subjects sub-01 --noise gaussian
python -m eeg2x run --dataset thingseeg --model avde --subjects sub-08 \
                    --noise gaussian --distinct-images 10
```

正式运行要求环境变量 `PYTHONHASHSEED=2000`，全局种子固定为 2000。

## 说明

* 编码器模型数学逐字沿用 EEG2X-Benchmark v2，未做改动。
* 训练产物只写 `EEG2X-Benchmark-v3` 根目录；读 v2 的 HDF5 与检查点仍允许。
* LUNA 的逐层衰减优化器**只在 ThingsEEG 上生效**，其他数据集回落到统一配方
  （Grootswager2019 的负结果见 `实验报告12.txt` 第九节）。
* **braindecode 必须是 1.3.2**：1.6 及以上版本的 LaBraM 架构不同，无法加载
  v2 的 checkpoint。
* 预处理产物与运行产物必须落在 benchmark 仓库内，这是可追溯性要求。
* 评测报告带 `task_contract` 字段：从产物 HDF5 读出的采样率、通道表、时间窗与
  预处理步骤，以及该任务实际使用的指标与方向。框架的 `run_config.json` 记录同一份契约。
* LUNA 的训练配方有意偏离框架统一配方，原因与代价见
  `docs/LUNA_TRAINING_DEVIATION.md`。

详细的中文介绍见仓库根目录的 `jieshao_benchmark_v3.txt`。
