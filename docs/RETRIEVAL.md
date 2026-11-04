# 检索任务（EEGBench）

检索是独立的一族任务，覆盖图像、文本、视频（以及音频）检索。代码来自
`/EEG_bench_open/EEGBench`，它建在 NeuralBench 上，模型、数据缓存与训练器
都与生成任务不同。

## 在 v3.1 里的位置

* `eeg2x/retrieval/registry.py` 登记 15 个数据集键（Grootswagers2019 拆成
  5 Hz 与 20 Hz 两个）、11 个模型与每个数据集的轨道布局；
* `eeg2x/retrieval/evaluate.py` 从 `predictions.npz` 与 `test_samples.jsonl`
  重算 subject-agg 的 top1/top5/top10 与中位排名，并与 `metrics.json` 对照；
* `eeg2x/retrieval/adapter.py` 通过 `EEG2X_EEGBENCH_PYTHON` 指定的解释器，
  调用 EEGBench 自己的 `eegbench.launch ... evaluate` 重跑推理；
* 命令行入口是 `benchmark.py retrieval`。

## 用法

    python benchmark.py retrieval --list
    python benchmark.py retrieval --dataset things_eeg --model labram --seed 33 --mode evaluate
    EEG2X_EEGBENCH_PYTHON=/path/to/python python benchmark.py retrieval \
        --dataset things_eeg --model labram --seed 33 --mode infer

`--mode evaluate` 只读已有轨道，不需要 NeuralBench，任何有 numpy 的解释器都能跑。
`--mode infer` 需要 EEGBench 自己的环境（Python 3.12 + neuralbench/neuraltrain/
neuralset/neuralfetch），见 `environments/retrieval.txt`；环境不可用时适配器会
明确报错，不会静默退回。

## 评测语义

同一 (subject, target_id) 的重复先取均值；候选集是全局 target_id 集合；预测与
目标都做 L2 归一化后按余弦相似度排序。参考实现的分组与并列处理没有完整公开，
因此与 `metrics.json` 的对照留 2e-3 的容差。

已核对：THINGS-EEG 与 Grootswagers2019 5Hz 的 labram seed-33 与参考逐位一致
（差约 2.5e-8，浮点）；ZuCo1 最大差 1.4e-3；ds005662 最大差 1.7e-4。

## 当前限制

`/EEG_bench_open` 的 vendor 目录只剩 `neuralbench`，`neuraltrain`、`neuralset`、
`neuralfetch` 的源码已被清理；本机也没有可 import 它们的 Python 3.12 环境。
所以 `--mode infer` 目前只能给出明确错误；真正的推理需要在恢复该环境后运行。
