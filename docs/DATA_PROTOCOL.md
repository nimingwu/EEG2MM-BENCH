# EEG2X-Benchmark v2 数据处理协议

v2 将 EEG 数值处理和刺激/事件对齐视为同一个可追溯的数据契约。模型和训练超参数不属于本协议；模型继续读取原有 `data`、`eeg`、标签和文本字段。

## 共同元数据

带 v2 属性的 HDF5 必须包含 `schema_version=2.0.0`、`data_protocol=eeg2x-data-protocol`、`dataset_id`、`task_id`、`subject_id`、`split`、`sampling_rate`、`source_sampling_rate`、`eeg_units`、`n_epochs`、`n_channels`、`epoch_type`、唯一的 `channel_names`，以及 preprocessing profile/state 的 canonical JSON 和 SHA-256 fingerprint。`split=test` 不能包含 `fit_scope=source` 的步骤；归一化、白化等有拟合状态的操作必须明确写为 `fit_scope=train`，并在切分后拟合。MNN 矩阵、RobustScaler median/IQR 等拟合产物随 HDF5 保存，并在 state 中记录数组 hash，使 test 文件可以证明引用的是 train 状态。

## 刺激 manifest

每一行 EEG epoch 对应一行刺激 manifest：`stimulus_id`、`stimulus_modality`、`stimulus_uri`、`stimulus_sha256`、`stimulus_onset_sec`、`stimulus_duration_sec`、`stimulus_alignment_source` 和 `stimulus_transform_json`。`stimulus_id` 由数据集 ID 和原生稳定 ID 组成，不得使用绝对路径。刺激缺失、顺序错位或行数不一致必须失败，不能用零向量静默补行。

刺激的专门处理由 profile 的 `align_stimulus` 步骤和 manifest 共同记录：图片记录 trigger/event 对齐，连续语音记录 TextGrid/TSV 对齐，ROAMM 记录 fixation/coordinates 对齐，Brennan 记录 `trl` 与词表对齐。因此“添加一个刺激文件”不等同于“完成刺激协议”。

CLIP/wav2vec2 等刺激编码器必须以 manifest 为行序权威。embedding 输出旁边写 `.manifest.json`，记录 encoder model/revision、stimulus ID 到 embedding row 的映射和 EEG preprocessing fingerprint。缺失刺激或 `word_order` 不一致时在模型加载前失败，不能跳过图片或插入零向量。

## EEG 处理原则

1. 先声明原始采样率，再进行精确有理数重采样。`256→250` 和 `512→250` 使用 polyphase anti-alias filter，并校验输出长度。
2. 读取 BrainVision/EDF/CNT 时只保留 EEG 通道，统一通道名并拒绝规范化后的重复通道；MNE Volt 输入显式转换为 microvolt。
3. epoch 时间戳统一在输出采样时钟上转换，同时保留秒级原始对齐信息。
4. profile 是模型无关的；EEGNet、LaBraM 等差异必须显式注册为不同 profile，而不能由模型名称隐式改变输入。
5. `eeg_units` 描述写入 HDF5 的数值，而不是读取前的原始单位。MNN 输出标记为 `dimensionless_whitened`，RobustScaler 输出标记为 `dimensionless_robust_scaled`；原始文件不能证明物理单位时使用 `dataset_native`，不得猜测为 microvolt。

## 全局随机性与可复现性

EEG2X-Benchmark-v2 的全局随机种子固定为 `2000`。该规则适用于整个
benchmark，而不只是某一个数据集、某一种 encoder 或某一个运行脚本。

凡是代码中存在随机性，均必须使用该统一种子，或者使用由该种子通过明确、
可复现规则派生出的子随机流。禁止在正式 benchmark 路径中保留其他随机种子。
具体包括但不限于：

1. Python `random`；
2. NumPy 全局 RNG 和 `numpy.random.default_rng`；
3. PyTorch CPU/CUDA RNG；
4. DataLoader 的 `generator`、worker 初始化和随机采样器；
5. scikit-learn 等库的 `random_state`；
6. train/validation/test 或 class split；
7. repetition/trial 随机选择；
8. 图像增强、随机 crop 和其他随机预处理；
9. 模型初始化、训练过程和 batch 顺序；
10. 图像生成和推理过程。

命令行参数如果暴露 `--seed`，其默认值必须为 `2000`。正式 benchmark 运行
时传入的 seed 也必须是 `2000`；不接受通过参数把正式运行切换到其他 seed。
代码中原来没有固定种子的随机过程必须补充固定种子，原来使用 `42`、`0`、
`7`、`8`、`33`、`20200220` 或其他数值的随机过程必须改为 `2000`。

对于需要并行或按样本产生独立随机状态的操作，允许使用：

    deterministic_subseed = f(seed=2000, stable_sample_id)

但 `seed` 的唯一根值仍必须是 `2000`，派生方法、稳定样本 ID 和最终 seed
必须写入 artifact manifest 或 preprocessing state。派生子 seed 不是新的
benchmark seed。

固定随机种子不等于所有 CUDA 算子天然逐位确定。使用 GPU 时还必须在合理
范围内开启 deterministic 设置、固定 `PYTHONHASHSEED=2000`，并记录设备、
软件版本以及无法保证确定性的算子。若某个算法本身不使用随机数，也应在
配置或 manifest 中记录 `seed=2000`，以保证整条运行链路的配置一致。

随机种子属于数据协议的可追溯性要求。所有由随机过程决定的 HDF5、特征、
checkpoint、生成图像和评测报告，都必须记录：

    seed = 2000

以及必要时的随机过程名称、派生规则和输入 artifact fingerprint。旧的、
使用其他 seed 生成的派生结果不能作为统一 `seed=2000` benchmark 的正式输入；
需要按本协议重新生成。

## 添加新数据集

实现 loader adapter，产出 legacy 字段和 v2 元数据；注册 profile 和任务 pattern；为刺激 manifest 写稳定 native ID；运行 `validate_h5_file(..., require_protocol_v2=True)` 及 CPU tests。只有 HDF5 schema、GT 提取和评测分支都接通后，才把任务加入 registry。

三个主 loader 默认严格失败：只要用户所选的数据集缺失、没有发现受试者、任一受试者处理失败，或写出的 HDF5 未通过协议校验，命令就返回非零状态。`--allow_partial` 仅供诊断和长批次中保留成功样本；以该选项生成的目录不是完整 benchmark 产物，不能作为正式发布数据。

协议固定的是可追溯字段、逐行对齐、拟合边界和机器校验，不固定所有数据集的滤波器、epoch 长度、通道数或刺激模态变换。数据集特有规则属于各自 versioned profile。

## ThingsEEG 双协议

ThingsEEG 检索使用 `thingseeg-image-nice-v1`：从刺激后 `[0, 1.0)` 秒取 epoch，不做 baseline correction，并在每个 session 内只用训练 split 拟合 MNN。正式检索评测文件名为 `thingseeg_sub-XX_test_epoched.h5`。

ATM 图像生成使用 `thingseeg-image-atm-v1`：从原始 `raw_eeg_training.npy` / `raw_eeg_test.npy` 读取 `[-0.2, 1.0)` 秒，减去 `[-0.2, 0)` 秒通道均值，重采样到 250 Hz 后裁出 `[0, 1.0)` 秒。按本协议统一的 `seed=2000` 在每个 session 中为每个训练 image condition 选 2 个 repetition、为每个测试 concept 选 20 个 repetition，并在每个 session 内只用选定的训练 split 拟合 MNN。训练保留选定的单 trial，测试按 concept 对所有 session 的 repetition 求均值。正式生成评测文件名为 `thingseeg_atm_sub-XX_test_epoched.h5`。

两个 profile 均禁止读取 `Preprocessed_data*`、原论文项目的 `ViT-H-14_features_*.pt`、VAE/GIT 特征、EEG embedding 或训练 checkpoint。当前仓库只能经根目录的 `ThingsEEG` 符号链接访问原始 EEG 和刺激图片；冻结基础模型缓存使用 `/datas/hf-cache`。由 benchmark 产生的 HDF5、特征、checkpoint、图像、日志和评测报告全部写入本仓库内部。

## ThingsEEG REVE 200 Hz 协议

REVE-base 的预训练时间尺度是 200 Hz，patch size 为 200 samples。不得把
`63 x 250` 输入直接交给 REVE，也不得在 encoder `forward()` 中隐式重采样。
ATM-REVE 使用两个独立、可 fingerprint 的 profile：

- `thingseeg-image-reve-nice-v1` 完全保留 NICE 检索的 `[0,1.0)` epoch、
  train-only per-session MNN 和 concept-average 测试逻辑，仅把精确 polyphase
  重采样目标改为 200 Hz，输出 `63 x 200`。
- `thingseeg-image-reve-atm-v1` 完全保留 ATM 生成的 `[-0.2,1.0)` source
  epoch、`[-0.2,0)` baseline、repetition selection、train-only per-session MNN
  和 concept-average 测试逻辑，仅把重采样和 `[0,1.0)` crop 输出改为
  200 Hz / `63 x 200`。

两个 REVE profile 均从符号链接下的原始 `raw_eeg_*.npy` 独立生成 HDF5，不能
复用或二次转换既有 250 Hz HDF5。论文 Low-Level 分支并不是被替换的 ATM-S
encoder，仍按原结构接收 `63 x 250`；ATM-REVE runner 因而从相同原始数据在
隔离目录生成一份 `thingseeg-image-atm-v1` 输入，并在使用前校验其正式测试
`stimulus_id` 行序与 REVE 生成 profile 完全一致。

图像目标也从原始图片重算。CLIP 严格使用发布代码的 OpenCLIP
`ViT-H-14/laion2b_s32b_b79k`、未归一化 1024 维 `encode_image` 输出和
`preprocess_train`（`RandomResizedCrop(224, scale=(0.9,1.0))`）。为使中断续跑
可重现，每张图片的 crop 随机状态由 `seed` 与稳定 `stimulus_id` 的 SHA-256
确定，并写入 feature manifest；这不改变发布代码的图像变换分布。

## ThingsEEG LUNA 256 Hz / 280 点输入协议

LUNA-Base 以 256 Hz EEG 预训练，patch size 为 40。ATM/NICE 的严格一秒语义在
256 Hz 下对应 256 个有效样本，不能被 40 整除；直接交给官方 temporal patch
embed 会失败。ATM-LUNA 采用不改变有效时间窗的最小适配：

1. 从符号链接下的原始 `raw_eeg_training.npy` / `raw_eeg_test.npy` 生成严格
   `[0,1.0)` 的 256 点有效 EEG。
2. 完成任务原有的 MNN；测试数据还先完成原任务规定的 concept aggregation。
3. 按官方 `ChannelWiseNormalize` 对每个样本、每个通道的 256 个有效点做 z-score，
   标准差 correction 为 1，epsilon 为 `1e-8`。
4. 只在右侧补 24 个零，得到七个 40 点 patch，即 HDF5 data shape
   `N x 63 x 280`；`sampling_rate=256` 独立记录，padding 不改变物理采样率。

两个 versioned profile 分别保持原任务逻辑：

- `thingseeg-image-luna-nice-v1` 保留 NICE 检索的 `[0,1.0)` epoch、train-only
  per-session MNN 与跨 session concept-average 测试；仅将目标采样率改为 256 Hz，
  并在 MNN/最终 concept average 后增加上述 z-score/padding。
- `thingseeg-image-luna-atm-v1` 保留 ATM 生成的 `[-0.2,1.0)` source epoch、
  `[-0.2,0)` baseline、论文 repetition selection、train-only per-session MNN
  与跨 session concept average；在聚合后增加上述 LUNA 输入适配。

LUNA 使用官方 `standard_1005` 三维电极位置逻辑；ATM 的 63 个固定通道必须全部
有有限坐标，坐标在模型内按官方方式逐样本归一化并做 NeRF positional encoding。
模型 encoder 只接受 `63 x 280` 且最后 24 点严格为零，避免 profile 与 checkpoint
契约被绕过。

论文 Low-Level 分支仍是独立的 250 Hz EEG-to-VAE-latent 模型，不由 LUNA 替换。
ATM-LUNA runner 因此从同一批原始文件另建 `thingseeg-image-atm-v1` 输入，并在训练
Low-Level 前校验它与 LUNA 生成 profile 的正式测试 `stimulus_id` 行序完全一致。
LUNA 高层 profile、Low-Level profile 和图像特征都不能复用 EEG_Image_decode 的
预处理/CLIP 文件；图像特征必须由 benchmark 根据 HDF5 manifest 从原图重新提取。

**训练侧的例外（不在本协议范围内，仅作交叉引用）**：统一框架的所有编码器共用一套
AdamW、平坦学习率、无 warmup、无调度的配方。唯一的数据集级差异是 LUNA 在 ThingsEEG
上把学习率设为 `1e-4`（框架默认 `3e-4`），其他设置与其他编码器完全相同；其他数据集
LUNA 仍用 `3e-4`。本协议第 3 行明确不规定训练超参，因此这**不是协议例外，而是
框架一致性的缺口**；见 [LUNA_TRAINING_DEVIATION.md](LUNA_TRAINING_DEVIATION.md)。

## 数据集划分：统一契约，不统一规则

各数据集的原始结构不同（概念数、类别数、序列数、block 数），因此**划分规则由各
数据集自己的 profile 决定**，协议不强制统一。但协议强制两件事：

1. **划分方式必须机器可读地记录在 HDF5 的 preprocessing state 里**：`split_unit`
   （划分单位名）、划分所用的 seed，以及被留出那一侧的稳定 ID 清单。当前实际记录：
   ds004018 记 `train_stimuli` / `test_stimuli`，NOD 记 `train_class_ids` /
   `test_class_ids`，ds005106 记 `train_blocks` / `test_blocks`（完整 block 内含
   全部 200 个刺激，故 block 清单唯一的确定了测试刺激集合）。测试侧的聚合口径由
   `test_aggregation` 记录。
2. **测试集与训练集的语义关系必须显式声明**，且必须在结果表里可见。判据是"测试的
   刺激/类别是否在训练中出现过"：否 → unseen-stimulus/class；是 → seen-stimulus
   （同一刺激的新试次）。**两种语义下的绝对指标不可横向排名**，只能组内比较。
   该语义目前记录在本节的下表中（尚未作为 HDF5 布尔字段写入；如需机器校验，
   后续版本再加）。

当前四个数据集的语义分属两组：

| 数据集 | 划分单位 | 测试刺激是否在训练中出现 | 测试行数 |
| --- | --- | --- | --- |
| ThingsEEG2 | 概念（数据集自带 train/test） | 否 | 200 |
| NOD-EEG | `class_id`（800/200 class-disjoint） | 否 | 200 |
| Grootswagers2019 / ds004018 | `stimulus`（180/20） | 否 | 20 |
| Grootswagers2025 / ds005106 | `block` | **是** | 200 |

### Grootswagers2019（ds004018）

v2 起划分单位为**刺激**：200 个刺激中按 seed 2000 无放回留 20 个作测试，其余 180
个训练；**序列不再参与划分**，因此每个留出刺激在 40 个序列 × 2 个速度上共 80 次
重复上取平均，每个训练刺激贡献 80 条试次（14400 条训练行）。

比例取 180/20 是为了复现 ThingsEEG2 的 1654/200 ≈ 90/10，使该数据集也成为
**unseen-stimulus** 重建，从而能与 ThingsEEG2 / NOD 同组比较。

必须随结果一并记录的代价：测试样本数只有 20，双向识别类指标的留出刺激抽样标准误
约 0.057（旧序列级 30/10 划分、200 个测试刺激时为 0.021）。因此该数据集的单被试
数值误差棒较宽；编码器/模型之间的比较应以同一批被试、同一批刺激的配对差值为准，
绝对值只作参考。FID 在 20 张图上不稳定，评测会告警，正式报告不使用该列。
划分噪声可用 `scripts/bootstrap_stimulus_noise.py` 对留出刺激重采样实测。

### Grootswagers2025（ds005106）

划分单位是 **block**（每个完整 block 把 200 张图各呈现一次）。只有完整 block 有
资格进入测试，测试 block 数 = `max(1, round(25% × 完整 block 数))`；不完整的 block
只贡献训练试次。

因此测试集覆盖全部 200 个刺激，**而这 200 个刺激在训练 block 里都出现过**——
该数据集是 **seen-stimulus** 重建，与上表其余三个数据集不是同一种任务。

同时必须记录的第二个限制：完整测试 block 通常只有 1 个，所以每个测试刺激平均只有
**1 个试次**（`min_test_repetitions_per_stimulus` 多为 1），测试 SNR 远低于
ThingsEEG2 的 80 次重复。加上训练试次中位数仅 ~784 条（ThingsEEG2 为 66160），
该数据集的定位是**数据量不足的补充数据集**，不适合与 ThingsEEG2 / NOD 并列解读。
