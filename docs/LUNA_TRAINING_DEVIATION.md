# LUNA 的学习率偏离

统一框架的所有编码器共用一套训练配方：`AdamW(model.parameters())`，平坦学习率
`3e-4`，无 warmup、无学习率调度，`weight decay` 使用 PyTorch 默认值，epoch / batch /
seed 分别为 40 / 64 / 2000。编码器之间的指标差异因此可以归因于编码器本身。

**唯一的数据集级例外是 LUNA 在 ThingsEEG 上的学习率。** LUNA 在 ThingsEEG 上使用
`1e-4`；其他数据集仍使用框架默认的 `3e-4`。优化器类别、weight decay、warmup、调度、
epoch / batch / seed 与其他编码器完全一致。

## 实现

LUNA 的 `EncoderSpec` 声明：

```python
learning_rate=1e-4
learning_rate_datasets=("thingseeg",)
```

`eeg2x/frameworks/only_clip/train.py` 读取当前数据集的覆盖值；没有覆盖时使用
`FRAMEWORK_LEARNING_RATE = 3e-4`。训练仍然只有一条代码路径，`build_optimizer` 只
按有效学习率构造 AdamW，不启用逐层衰减或调度器。`run_config.json` 记录
`learning_rate` 与 `learning_rate_source`。

## 可比性

LUNA 在 ThingsEEG 上的学习率与其他编码器不同，因此该行不能与其他编码器的 `3e-4`
行当作完全相同的超参直接并列排名。LUNA 在其他数据集上的学习率仍为 `3e-4`。

## 历史

v2 与早期 v3 曾为 LUNA 使用论文 Table 6 的复杂微调配方（`peak_lr 1e-4`、
`min_lr 5e-6`、`warmup 5`、`weight_decay 0.05`、`layer_decay 0.5`），用于压制
平坦 `3e-4` 下的残差流漂移。该复杂设置已从 v3.1 移除；本版本只保留学习率覆盖。
旧配方的诊断、证据与历史结果属于 v2 / 实验报告12，不能与当前 lr-only 的运行直接
比较。
