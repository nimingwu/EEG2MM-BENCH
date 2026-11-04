"""任务接口：预测输出的 schema、校验器与 HDF5 行构造。

迁自 v2 根目录的三个 ``*_task_interface.py``（你的要求 5）。原文件是**契约模块**
而不是运行时实现——它们定义 ``TaskSpec``、``Sample``、输出 dataclass、校验函数
和 JSON schema 导出，不含模型、prompt 模板或生成逻辑。

**不要 from ... import ***：三个模块各自定义了同名的 ``EegSpec`` 与各自的
``TASK_REGISTRY``，通配导出会互相覆盖。请按需显式导入::

    from eeg2x.tasks.image import Sample, TaskSpec, validate_output
    from eeg2x.tasks.text import TextSample

与 v2 的差别（v3 已接线）：

* ``image.py`` 不再硬编码 250 Hz / 32 通道。EEG 几何随 (数据集 x 编码器 profile)
  变化，权威来源是产物 HDF5：:meth:`eeg2x.tasks.image.EegSpec.from_h5` 读出
  采样率、通道表、时间窗与预处理步骤。v2 的 32 通道常量对 ThingsEEG 就是错的
  （实际 63 通道），对 LaBraM/REVE/LUNA 的 200/256 Hz 也不成立。
* ``eeg2x.evaluation.api.evaluate_predictions`` 在评测时调用
  ``eeg_spec_from_h5_files``，把任务契约写进报告的 ``task_contract`` 字段；
  框架的 ``run_config.json`` 也记录同一份契约。
* 图像生成任务的指标清单只在 ``TASK_REGISTRY["image_generation"]`` 里定义，
  ``eeg2x.evaluation.registry`` 引用它，不再两处各写一份。

``text.py`` / ``speech.py`` 在 v2 里没有任何调用方（当时就是死代码），保留是
为了 video/text 阶段接入时有机器可读的 schema；它们尚未接线。
"""
__all__ = ["image", "speech", "text"]
