# EEG2MM-BENCH

一个统一的多模态 EEG 基准，覆盖 EEG-to-Image、EEG-to-Video、EEG-to-Text
三条生成任务，以及基于 EEGBench 已有轨道的检索评测。框架只实现一遍，EEG
编码器可插拔；基线保持各自固定的编码器。

## 环境

三条生成任务与检索使用的依赖互不兼容，尤其是 torch、diffusers、transformers
的版本，因此没有一个解释器能同时跑全部。仓库按用途提供六份固定版本的
requirements，放在 `environments/` 下；使用时为每一类各建一个虚拟环境。
仓库本身不需要安装，从仓库根目录直接运行即可。

| 环境 | 用途 | Python | 安装文件 |
| --- | --- | --- | --- |
| image | 图像框架的基础依赖（atm_s / enigma_cnn / reve）与 ENIGMA | 3.11 | `environments/image.txt` |
| image-braindecode | 追加 LaBraM 编码器（`braindecode==1.3.2`） | 3.11 | `environments/image-braindecode.txt` |
| image-luna | 追加 LUNA 编码器（`rotary-embedding-torch`） | 3.11 | `environments/image-luna.txt` |
| video | EEG-to-Video 框架（五个编码器）、EEGVid Orig、EEG2Video Orig | 3.10 | `environments/video.txt` |
| text | GLIM 文本框架（四个编码器）、GLIM Orig、EEG2Text Orig、RAG Orig | 3.9 | `environments/text.txt` |
| retrieval | EEGBench 检索 | 3.12 | `environments/retrieval.txt` |

图像拆成三份，是为了说明额外依赖：LaBraM 需要 `braindecode==1.3.2`，LUNA 需要
`rotary-embedding-torch`。v2 的实际做法是把三组装进同一个 `eeg2x-atm` 环境，
一次跑完 atm_s、labram、reve、luna；下面的安装步骤也按一个环境来写。本机
已复制好的历史环境在 `/datas/linshishangchuan/11-3-image-env/eeg2x-atm`，
可以直接使用；只跑单一编码器时，也可以只装对应的最小依赖。

```bash
cd EEG2X-Benchmark-v3.1

# 图像：一个环境装齐五个编码器（与 v2 的 eeg2x-atm 一致）
# 后两份是追加依赖，用 --no-deps 安装，避免 braindecode 把 mne 升到 1.11 以上。
python3.11 -m venv .venv-image
.venv-image/bin/pip install -U pip
.venv-image/bin/pip install -r environments/image.txt
.venv-image/bin/pip install --no-deps -r environments/image-braindecode.txt
.venv-image/bin/pip install --no-deps -r environments/image-luna.txt

# 视频
python3.10 -m venv .venv-video
.venv-video/bin/pip install -U pip
.venv-video/bin/pip install -r environments/video.txt

# 文本
python3.9 -m venv .venv-text
.venv-text/bin/pip install -U pip
.venv-text/bin/pip install -r environments/text.txt

# 检索：除 requirements 外还需 NeuroAI 四个包，步骤见 environments/README.md
python3.12 -m venv .venv-retrieval
.venv-retrieval/bin/pip install -U pip
.venv-retrieval/bin/pip install -r environments/retrieval.txt
```

如果机器上没有对应的小版本，先用 conda 建一个同版本的 base 解释器，再在
它上面执行 `python -m venv`。

用对应环境的解释器运行命令，不需要先激活环境：

```bash
.venv-image/bin/python benchmark.py run --dataset thingseeg --model only_clip \
    --encoder atm_s --subjects sub-01 --stages inference

.venv-image/bin/python benchmark.py run --dataset thingseeg --model only_clip \
    --encoder labram --subjects sub-01

.venv-image/bin/python benchmark.py run --dataset thingseeg --model only_clip \
    --encoder luna --subjects sub-01

.venv-video/bin/python benchmark.py run --dataset seed_dv --model eeg2video \
    --encoder eegnet --subjects sub-01

.venv-text/bin/python benchmark.py run --dataset zuco2 --model glim \
    --encoder labram --subjects YAC
```

运行前可以用 `benchmark.py env` 检查当前解释器能满足哪些环境、还缺哪些包；
加 `--json` 输出机器可读版本。选错环境时 CLI 会在导入框架之前报错，并指出
应该改用 `environments/` 下的哪一份文件。

环境之间只通过产物交接（HDF5、checkpoint、评测报告），不需要把依赖合并到
同一个解释器。检索是例外：它调用 EEGBench 自己的解释器，需要用
`EEG2X_EEGBENCH_PYTHON` 指向检索环境。AVDE 在 v2 也使用独立环境
`/datas/linshishangchuan/avde`（Python 3.10），复现历史结果时优先用它。
详见 `environments/README.md`。

## 用法

下面的命令用当前环境解释器执行，`python -m eeg2x` 与 `benchmark.py` 等价。

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

# 视频从零训练：Scheme D 两阶段。checkpoint 写到
# artifacts/<dataset>/eeg2video/<subject>/<encoder>/checkpoints；
# 之后的推理优先使用这份 v3 checkpoint，缺失时才回落到视频仓库的干净轨道。
python -m eeg2x run --dataset seed_dv --model eeg2video --encoder labram \
                    --subjects sub-01 --stages train \
                    --video-stage-one-epochs 200 --video-stage-two-epochs 150 \
                    --video-batch-size 32

# 文本从零训练：GLIM 的四个变体（glim/labram/luna/reve）。checkpoint 写到
# artifacts/<dataset>/glim/<encoder>/checkpoints，之后的推理优先使用它。
# 训练按 val/loss_total 保存 best checkpoint，并默认以 patience=15、min_delta=1e-4
# 早停；--text-early-stop-patience 0 可以关闭早停。
# 正式训练通常用 --text-devices 4；梯度累积默认把有效优化 batch 补到 576。
python -m eeg2x run --dataset zuco2 --model glim --encoder labram \
                    --subjects all --stages train inference \
                    --text-epochs 200 --text-devices 4

# 噪声对照：冻结权重，把测试集 EEG 换成 N(0,1)。
# 产物落在 <干净根>/noise_control/，绝不覆盖干净结果。
python -m eeg2x run --dataset thingseeg --model only_clip --encoder luna \
                    --subjects sub-01 --noise gaussian

# 检索：枚举轨道或从已有预测重算 subject-agg 指标。
python -m eeg2x retrieval --list
python -m eeg2x retrieval --dataset things_eeg --model labram --seed 33 --mode evaluate
# Dummy：读取 EEGBench 的 canonical 结果，不训练也不做 EEG 推理。
python -m eeg2x retrieval --dataset things_eeg --model dummy --mode evaluate
```

正式运行要求环境变量 `PYTHONHASHSEED=2000`，全局种子固定为 2000。

## 说明

* 三条生成任务共用同一套编码器实现；检索任务复用 EEGBench 的模型与数据，
  不复制它的模型层。
* 训练与推理产物只写本仓库内的 `artifacts/` 与 `h5/`，保证可追溯。
* LUNA 在 ThingsEEG 上只把学习率改为 `1e-4`，其他训练设置与其他编码器完全一致；
  其他数据集 LUNA 仍用框架默认的 `3e-4`。因此 LUNA 的 ThingsEEG 行不能与其他
  编码器的 `3e-4` 行直接并列比较，见 `docs/LUNA_TRAINING_DEVIATION.md`。
* **braindecode 必须是 1.3.2**：1.6 及以上版本的 LaBraM 架构不同，无法加载
  已有的 checkpoint。
* 预处理产物与运行产物必须落在 benchmark 仓库内，这是可追溯性要求。
* 评测报告带 `task_contract` 字段：从产物 HDF5 读出的采样率、通道表、时间窗与
  预处理步骤，以及该任务实际使用的指标与方向。框架的 `run_config.json` 记录同一份契约。
