# preprocessing 结构

预处理有两条不同的数据契约，目录按它们分开，但入口只有一个：
`eeg2x.preprocessing.preprocess(dataset, encoder, subject, out_dir=..., data_root=...)`。

## 图像

`thingseeg.py`、`nod.py`、`grootswagers.py`、`grootswagers2025.py` 是四个数据集模块。
几何参数（目标采样率、输出样本数、是否 LUNA 适配）由 `__init__.py` 里的
`GEOMETRY` 表按 profile 编号给出，`_ADAPTERS` 负责分派。

* `ops.py` 是与数据集无关的通用原语：重采样、切 epoch、事件掩码、MNN 白化。
* `legacy_image.py` 只保留 ThingsEEG 专用的 OSF 读取、通道表与图像协议元数据；
  它从 v2 的 loader 迁入，NOD 旧分支与 CLI 入口已移除。

## 视频

四个数据集各一个模块：`seed_dv.py`、`ds005662.py`、`eeg_video_action.py`、
`ds008257.py`。`video_common.py` 放共享的 `_resize_rgb` 与资源目录。
视频的几何由各编码器分支在模块内部自己决定，所以 `_VIDEO_ADAPTERS` 的签名
比图像多一个 encoder 参数。

## 文本

`zuco1.py`、`zuco2.py`、`chisco.py`、`tmnred.py` 负责 dataframe 到 HDF5 的第二段，
写入 `<out_dir>/<encoder>/<dataset>_<encoder>_<subject>_<split>_sentences.h5`。
文本 preprocessing 在 GLIM 仓库原本是两段：先把原始 .mat/.pkl 做成
`seed2000_source` 下的 dataframe，再由这些模块写 H5。当前只迁移了第二段；
`data_root` 指向 dataframe，留空时使用 GLIM 仓库里的默认文件。

文本 profile 在 `eeg2x/profiles_text.py`，由 `profiles.py` 合并
`task_id == "sentence_generation"` 的条目。

## 资源

`resources/` 存放随包分发的刺激标注 JSON（ds005662、ds008257、
EEG-Video-Action）。`pyproject.toml` 已把 `eeg2x.preprocessing` 的
`resources/*.json` 声明为 package data。
