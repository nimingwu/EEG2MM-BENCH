# 噪声对照（Noise Control）：实现说明

「**冻结权重 + 纯噪声 EEG**」这个对照条件在 EEG2X-Benchmark 里怎么实现、代码在哪、
与上游 `fuxian-GLIM/EEG2X-Benchmark-v2-GS`（其 `NOISE_EXPERIMENT.md` 是源头）差在哪。
路径均相对本目录。

---

## 1. 它回答什么问题

不是一个鲁棒性指标，而是一个**对照**：模型到底有没有在用 EEG？

把整段 EEG 换成标准正态噪声，图像目标 / 标签 / prompt / 所有超参一律不动，照常推理
与评测。若此时指标纹丝不动，说明模型靠的是先验或别的通路，不是脑电信号。

**只针对重建（`image_generation`）**，检索任务没有接线（一行都没改）。

## 2. 与上游 GLIM 版的差异

| | GLIM (`EEG2X-Benchmark-v2-GS`) | 本仓库 |
| --- | --- | --- |
| 任务 | 文本重建（sentence_generation） | 图像重建（image_generation） |
| 入口 | `generate_sentence_predictions.py`（纯推理脚本） | `run_noise_control.py`（专用驱动） |
| 为什么入口不同 | 上游训练与推理本来就是分开的 | **本仓库训练和推理混在 `run_*.py` 里**，所以驱动显式只做「训练完之后」的那一半 |
| 注入点 | 2 个（两族模型 forward 签名不同） | **4 个**（四个模型族的 EEG→预测 路径互不相通） |
| 其余 | — | 三条约定、种子策略、验证断言全部照抄 |

**不重训**是这里最关键的一条：所有 checkpoint（编码器 / 投影头 / Diffusion Prior /
低层 VAE 编码器 / 语义 GIT / AVDE 的 VAE+VAR）都从**干净产物目录读**，只有测试集
EEG 被替换。

## 3. 代码位置

| 文件 | 作用 |
| --- | --- |
| `model/noise_control.py` | 共享实现：CLI 参数、噪声抽样、`_noise` 路径、旁挂 meta、等价性断言 |
| `model/atm/training.py:extract_atm_embeddings` | **注入点 1**：统一框架的 5 个编码器 + 完整 ATM 的编码器段 |
| `model/enigma/pipeline.py:_predict_embeddings` | **注入点 2**：ENIGMA baseline |
| `model/avde/inference.py:generate_stage` | **注入点 3**：AVDE baseline |
| `run_noise_control.py` | 驱动：定位干净 checkpoint → 只跑测试侧推理 → 写 `noise_control/` → 评测 |
| `scripts/run_noise_control.sh` | 一行跑完全部模型族 |

四个注入点按**模型族**分组（上游文档 §8.6：别按模型逐个实现，迟早会漂移）。
新增一个走统一框架的编码器时噪声条件是**白拿的**——它走同一个
`extract_atm_embeddings`。

## 4. 数据流

```
干净:  h5 测试 EEG ────────────────────────→ 冻结模型 → SDXL/VQ-VAE → images/ → 8 指标
噪声:  h5 测试 EEG ─→ randn(shape) ────────→ 冻结模型 → SDXL/VQ-VAE → images/ → 8 指标
                       ↑ 唯一改动
```

评测端**一行都没改**：`benchmark_eval` 只吃「predictions.json + h5 参考」，不关心
EEG 是怎么来的。

只有 **test** split 被替换。train split 保持干净——投影头 / Diffusion Prior 是在干净
数据上训完的，不重训，所以对照隔离出的正是「冻结模型在没有 EEG 时会怎样」。

## 5. 三条约定（照抄上游）

1. **绝不覆盖干净结果。** 噪声产物写在 `<干净根>/noise_control/` 下，调用方照常传干净
   路径；结构上不可能互相覆盖。
2. **条件元数据不放进预测 JSON。** 评测把 predictions.json 的每个顶层 key 当样本 id，
   混进 `{"noise": "gaussian"}` 会被当成一条样本。条件写旁挂的 `noise_meta.json`。
3. **种子固定且留痕。** 一个 `torch.Generator` 跑完全程 ⇒ 同 batch size、同数据顺序下
   重跑得到同一份噪声；种子随产物写进 meta。

上游文档 §8.7 的两个坑按推荐方式规避：用 `torch.randn`（`torch.empty` 不接受
`generator`）；**在 CPU 上采样再 `.to(device)`**（CPU 的 `torch.Generator` 不能直接给
CUDA 张量抽样，且 CPU 采样能保证换设备后抽到的是同一份噪声）。

## 6. 怎么用

```bash
bash scripts/run_noise_control.sh              # 全部模型族，ThingsEEG sub-08
bash scripts/run_noise_control.sh atm labram   # 只跑指定模型
PYTHONHASHSEED=2000 python3 run_noise_control.py --model all --noise-seed 2000
```

干净条件仍然走原来的 `run_*.py`；本驱动**只跑噪声条件**，`--noise none` 会被拒绝
（所以不存在「以为是噪声结果其实是干净结果」这种情况）。

产物（以统一框架的 `atm` 为例）：

```
artifacts/only_clip_linear_seed2000/atm/sub-08/          # 干净（不动）
└── noise_control/
    ├── embeddings/{train,test}_embeddings.h5     # 只有 test 是噪声推出来的
    ├── projection/test_clip_predictions.h5
    ├── generation/clip/{images,predictions.json}
    ├── evaluation/generation_clip.json           # 8 指标，直接与干净条件对照
    └── noise_meta.json                           # 条件溯源（约定 2/3）
```

## 7. 验证

1. **逐字节断言（自动）**：每个模型跑完后比较干净与噪声的 `images/`，
   `noise_control.validate_noise_outputs` 断言不得有任何一张图逐字节相同。
   这条断言本身也做了反向验证（把干净目录与它自己比，必须触发）。
2. **单元自检**：同种子同形状 ⇒ 同一份噪声；不同种子 ⇒ 不同；`generator=None`
   ⇒ 原样返回；形状/dtype/设备在注入后不变。
3. **实测**：见第 8 节。

## 8. 结果（ThingsEEG sub-08）

四项语义指标（clip_similarity / alexnet_2 / alexnet_5 / inception）在噪声条件下应塌回
**0.5000**（benchmark 的语义指标是 2AFC 双向识别，chance = 0.5，与 N 无关）；swav
越低越好，应上升（变差）。见本目录报告第十三/十五章与本文件的更新记录。

## 9. 与其它「加噪」实验的区别

| 实验 | 做法 | 本 benchmark |
| --- | --- | --- |
| **噪声对照**（本文档） | EEG 整段换成 `N(0,1)`，权重冻结 | ✅ 已实现 |
| SNR 分级加噪 | 噪声按信噪比缩放，得到性能-噪声曲线 | ❌ 未实现 |
| 训练期加噪增强 | 训练时注入噪声提升鲁棒性 | ❌ 未实现 |
