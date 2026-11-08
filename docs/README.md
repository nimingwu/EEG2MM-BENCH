# v3 文档索引

本目录的文档**继承自 v2**（`EEG2X-Benchmark-v2-test/`），内容未改写——它们是
协议条款与实验结案的记录，不是 v3 的代码说明。v3 的代码结构见仓库根的
`README.md` 与各模块的 docstring。

## 各文件的当前效力

| 文件 | 效力 | 说明 |
| --- | --- | --- |
| `DATA_PROTOCOL.md` | **有效** | v3 沿用同一套数据协议：全局 seed=2000、versioned profile、HDF5 v2 schema、stimulus manifest 行对齐、`require_benchmark_output` 守卫。v3 的 `eeg2x/protocol.py`、`h5_schema.py`、`profiles.py` 逐字迁自它描述的代码。 |
| `ADDING_A_DATASET.md` | **有效** | 接入新数据集的检查表；v3 的入口是 `eeg2x/datasets.py` + `eeg2x/preprocessing/<dataset>.py`。 |
| `LUNA_TRAINING_DEVIATION.md` | **有效** | LUNA 在 ThingsEEG 上只覆盖学习率为 1e-4，其余训练设置与其他编码器一致；记录适用范围与历史。v3 用 `EncoderSpec.learning_rate` / `learning_rate_datasets` 表达（见 `eeg2x/encoders/luna/__init__.py`）。 |
| `NOISE_EXPERIMENT.md` | **有效** | 噪声对照的实现说明。v3 的共享实现在 `eeg2x/noise_control.py`，三个注入点已全部接线：框架 `eeg2x/frameworks/only_clip/embed.py`（五个可插拔编码器共用，含 LUNA 的有效样本长度）、ENIGMA `eeg2x/baselines/enigma/pipeline.py:_predict_embeddings`、AVDE `eeg2x/baselines/avde/inference.py:generate_stage`。命令行入口是 `run --noise gaussian`。完整 ATM 尚未移植，它的注入点随其推理入口一起待补。 |

## 与 v3 代码的对应关系

| 文档里的概念 | v2 位置 | v3 位置 |
| --- | --- | --- |
| 数据协议 / fingerprint | `core/data_protocol.py` | `eeg2x/protocol.py` |
| HDF5 v2 校验 | `core/h5_protocol.py`、`core/h5_schema.py` | `eeg2x/h5_protocol.py`、`eeg2x/h5_schema.py` |
| versioned profiles | `core/preprocessing_profiles.py` | `eeg2x/profiles.py` |
| profile → 预处理几何 | 各 runner 的 `GEOMETRY` 表 | `eeg2x/preprocessing/__init__.py:GEOMETRY` |
| 产物路径守卫 | `core/paths.py` | `eeg2x/rawdata.py`（多根，读 v2 / 写 v3） |
| 数据集注册表 | 散在各 runner 的 `DATASETS` | `eeg2x/datasets.py` |
| 任务 schema | `eeg2*_task_interface.py` | `eeg2x/tasks/{image,text,speech}.py` |

## 尚未迁入的文档

v2 根目录还有 `README.md`（629 行，含大量已失效的 runner 说明）、
`readme_old.md`、`ATM_S_ThingsEEG_sub-08_实验报告.md`。这些描述的是 v2 的
runner 结构，**不适用于 v3**，因此没有迁入：

* `README.md` 里列举的 `run_atm_reve_benchmark.py` / `run_atm_luna_benchmark.py` /
  `run_nod_reve_luna_benchmark.py` 调用的是**已被移除的 Diffusion Prior**，在 v2
  里就已失效；v3 没有这些脚本。
* `ATM_S_ThingsEEG_sub-08_实验报告.md` 记录的是 v2 sub-08 的实验，属结案材料，
  应留在 v2。

v3 的 `README.md` 描述 `benchmark.py` 的四个子命令与目录结构。
