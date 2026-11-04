# 添加数据集检查表

1. 明确原始 EEG 时钟、单位、通道类型、事件时钟和刺激文件格式。
2. 在 `core/preprocessing_profiles.py` 注册 profile，列出有序步骤、参数和 `fit_scope`。
3. 在 loader 中使用 `resample_eeg`、`normalize_channel_names`、`seconds_to_sample`；不要用四舍五入比例替代真实重采样。
4. 保留模型依赖的旧根数据集/标签字段，同时调用 `write_common_h5_metadata`。
5. 为每个 epoch 写一条 `StimulusRecord`；manifest 顺序必须与 EEG 行一一对应。无法计算 hash 时可留空，但不能伪造 hash。
6. 在 `core/registry.py` 写准确的 `h5_patterns`、GT 字段和 schema；不要让一个任务扫描其它任务的文件。
7. 在 `benchmark_eval.py` 增加必要的 GT 分支，并优先读取 manifest 中的 URI/ID。
8. 用 CPU-only fixture 测试重采样、通道重复、profile fingerprint、manifest 行对齐、schema 缺失属性和混合 HDF5 目录过滤。
9. 若任务使用预计算刺激 embedding，必须从 HDF5 manifest 读取行顺序，保存 stimulus ID→row 映射和 encoder revision；缺文件或顺序不一致必须失败。
10. loader 默认使用严格模式：所选数据集缺失、没有受试者、任一受试者失败或最终 schema 校验失败时，进程必须以失败结束。`--allow_partial` 只用于定位坏样本或保留批处理中的成功部分，不能用于生成正式 benchmark 发布产物。

数据集适配器可以有不同的 EEG/刺激处理；统一的是记录方式、切分边界、失败可见性和行对齐，而不是强迫所有数据集使用同一滤波或同一 epoch 长度。
