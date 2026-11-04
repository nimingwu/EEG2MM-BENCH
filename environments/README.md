# 环境说明

v3.1 不假设整个 benchmark 能在一个 Python 环境里跑。不同框架与基线依赖不同的
包和版本，锁进一个环境只会互相牵制。本目录按用途给出若干份固定版本的说明：

| 文件 | 用途 | 已知可用的解释器 |
| --- | --- | --- |
| `image.txt` | Only-CLIP 框架（atm_s / reve / enigma_cnn）与 ENIGMA、AVDE | 旧图像环境 |
| `image-braindecode.txt` | 在 `image.txt` 之上加 LaBraM（braindecode==1.3.2） | 旧图像环境加 braindecode |
| `image-luna.txt` | 在 `image.txt` 之上加 LUNA（rotary-embedding-torch==0.8.9） | 旧图像环境加 rotary |
| `video.txt` | EEG-to-Video 框架与 EEGVid / EEG2Video Orig | `/datas/linshishangchuan/eeg2video` |
| `text.txt` | GLIM 文本框架与 GLIM Orig | `/datas/linshishangchuan/glim_2` |
| `retrieval.txt` | EEGBench 检索（NeuralBench 栈，Python 3.12） | 目前没有可用的本机环境 |

仓库本身不创建环境，也不写死解释器路径。用哪个解释器由使用者提供，例如：

    /datas/linshishangchuan/glim_2/bin/python benchmark.py run --dataset zuco2 ...
    /datas/linshishangchuan/eeg2video/bin/python benchmark.py run --dataset seed_dv ...

运行前可以用 `benchmark.py env` 查看当前解释器装了哪些包、能跑哪些框架。缺包时
CLI 会明确报错并指出应该换用哪一份环境说明。

环境之间只通过产物交接：HDF5、checkpoint、评测报告。因此导入路径保持惰性：
braindecode 在构造 LaBraM 时导入，rotary_embedding_torch 在构造 LUNA 时导入，
torch_geometric 只在 eegvid_nice 分支导入，diffusers/SDXL 只在图像分支导入。
