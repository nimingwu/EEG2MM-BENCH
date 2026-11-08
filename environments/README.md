# 环境说明

v3.1 不假设整个 benchmark 能在一个 Python 环境里跑。不同框架与基线依赖不同的
包和版本，锁进一个环境只会互相牵制。本目录按用途给出若干份固定版本的说明：

| 文件 | 用途 | Python | 已知可用的解释器 |
| --- | --- | --- | --- |
| `image.txt` | Only-CLIP 框架的基础依赖（atm_s / reve / enigma_cnn）与 ENIGMA | 3.11 | `/datas/linshishangchuan/11-3-image-env/eeg2x-atm` |
| `image-braindecode.txt` | 在 `image.txt` 之上加 LaBraM（braindecode==1.3.2） | 3.11 | 同上（已包含） |
| `image-luna.txt` | 在 `image.txt` 之上加 LUNA（rotary-embedding-torch==0.8.9） | 3.11 | 同上（已包含） |
| `video.txt` | EEG-to-Video 框架与 EEGVid / EEG2Video Orig | 3.10 | `/datas/linshishangchuan/eeg2video` |
| `text.txt` | GLIM 文本框架与 GLIM Orig | 3.9 | `/datas/linshishangchuan/glim_2` |
| `retrieval.txt` | EEGBench 检索（NeuralBench 栈） | 3.12 | `/datas/linshishangchuan/11-3-jiansuo/eegbench-retrieval` |

`/datas/linshishangchuan/11-3-image-env/eeg2x-atm` 是从 v2 机器复制下来的
图像环境，三组依赖都已安装。v2 的统一框架在四个数据集上就是用这一个环境
跑完 atm、labram、reve、luna 的，ENIGMA 也在其中；AVDE 使用独立环境，见下文。

## 创建环境

图像按 v2 的做法用一个环境装齐三组依赖。`image-braindecode.txt` 与
`image-luna.txt` 是追加依赖，用 `--no-deps` 安装，避免 braindecode 的元数据
把 mne 升到 1.11 以上，破坏 v2 的预处理与数值：

```bash
python3.11 -m venv .venv-image
.venv-image/bin/pip install -U pip
.venv-image/bin/pip install -r environments/image.txt
.venv-image/bin/pip install --no-deps -r environments/image-braindecode.txt
.venv-image/bin/pip install --no-deps -r environments/image-luna.txt
```

只跑 atm_s、reve、enigma_cnn 或 ENIGMA 时，只装 `environments/image.txt`
即可；这样得到的是更小的环境，与 v2 的一体化环境不同。

视频与文本同理：

```bash
python3.10 -m venv .venv-video
.venv-video/bin/pip install -r environments/video.txt

python3.9 -m venv .venv-text
.venv-text/bin/pip install -r environments/text.txt
```

## AVDE 与历史环境

AVDE baseline 在 v2 使用独立环境 `/datas/linshishangchuan/avde`（Python 3.10，
自带 numpy 2.x、transformers 4.48 与 open_clip 2.30）。它和 image 环境不是
同一套依赖；v3.1 的 AVDE 入口在 image 环境里也能导入，但复现历史结果时优先
使用这个环境。

v2 的完整 ATM 方法还用过 `eeg2x-atm-reve`（REVE 分支）和 `.venv-atm-luna`
（LUNA 分支，建在 `eeg2x-atm-reve` 之上）。这两条是完整 ATM 的变体，不是
统一框架里可替换的编码器；v3.1 暂未迁入它们的推理入口。

## 检索环境

检索建在 EEGBench 与 NeuroAI 的四个包上，除 `retrieval.txt` 外还需要从源码
安装。NeuroAI 固定到提交 `096c01a1`，版本不一致时 NeuralBench 的接口会变：

```bash
python3.12 -m venv .venv-retrieval
.venv-retrieval/bin/pip install -U pip
.venv-retrieval/bin/pip install -r environments/retrieval.txt

git clone https://github.com/facebookresearch/neuroai.git
cd neuroai
git checkout 096c01a13f32efb3da9de6a015fa40dd01c4e450
../.venv-retrieval/bin/pip install -e neuralset-repo -e neuralfetch-repo \
                                -e neuraltrain-repo -e neuralbench-repo
```

EEGBench 本体不包含在本仓库中，默认从 `/EEG_bench_open` 读取，其
`EEGBench/src` 会被自动加入 `PYTHONPATH`，不需要额外安装。工作区在别处时用
`EEG2X_EEGBENCH_ROOT` 指向它；EEGBench 校验 `vendor/neuroai` 时，把该路径指向
上面的 clone 即可。从其他解释器调用检索 CLI 时，用 `EEG2X_EEGBENCH_PYTHON`
指定检索环境的解释器：

```bash
# 直接用检索环境运行
.venv-retrieval/bin/python benchmark.py retrieval --list
.venv-retrieval/bin/python benchmark.py retrieval \
    --dataset things_eeg --model labram --seed 33 --mode evaluate

# 或用别的解释器运行，由变量指定 EEGBench 的解释器
EEG2X_EEGBENCH_PYTHON=$PWD/.venv-retrieval/bin/python \
.venv-image/bin/python benchmark.py retrieval \
    --dataset things_eeg --model labram --seed 33 --mode infer
```

检索的推理需要 EEGBench 运行目录下的 EEG 数据缓存；缓存被清理后只能做基于
已有预测的离线评测，不能重建 DataLoader。

## 选择解释器

仓库不写死解释器路径。用哪个解释器由使用者提供，例如：

    /datas/linshishangchuan/glim_2/bin/python benchmark.py run --dataset zuco2 ...
    /datas/linshishangchuan/eeg2video/bin/python benchmark.py run --dataset seed_dv ...

运行前可以用 `benchmark.py env` 查看当前解释器装了哪些包、能跑哪些框架，加
`--json` 输出机器可读版本。缺包时 CLI 会明确报错并指出应该换用哪一份环境说明。

环境之间只通过产物交接：HDF5、checkpoint、评测报告。因此导入路径保持惰性：
braindecode 在构造 LaBraM 时导入，rotary_embedding_torch 在构造 LUNA 时导入，
torch_geometric 只在 eegvid_nice 分支导入，diffusers/SDXL 只在图像分支导入。
