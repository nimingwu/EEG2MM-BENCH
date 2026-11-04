"""统一的命令行入口。

v2 有 15 个 ``run_*.py``，其中 7 个各自复制了同一段链路。v3 只留这一处：

    python benchmark.py list
    python benchmark.py run    --dataset thingseeg --model only_clip --encoder luna \\
                               --subjects all --stages inference
    python benchmark.py verify --dataset thingseeg --encoder luna --subject sub-01

约定：

* ``--subjects`` 默认 ``all``（数据集的全部被试）。
* ``--stages`` 默认 ``all``，等同于 ``inference``
  （``embed,project,generate,evaluate``），即复用已有检查点。
  **训练必须显式请求**：``--stages train ...``；只给 ``train`` 时训练完就停。
* ``--encoder`` **只对框架有效**；给 baseline 传会**报错**而不是忽略——baseline
  的编码器是固定的，这是设计约束，不是可选参数。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from .datasets import DATASETS, DatasetSpec, resolve as resolve_dataset
from .paths import V2_ROOT, V3_ROOT, artifact_dir

log = logging.getLogger("eeg2x.cli")

SEED = 2000

def _encoder_registry():
    """惰性导入编码器注册表：env 子命令不应强制加载 torch。"""
    from . import encoders
    return encoders


#: 框架（编码器可换）。三套框架并列：图像 Only-CLIP、视频 Scheme D、文本 GLIM。
FRAMEWORKS = ("only_clip", "eeg2video", "glim")
#: 视频框架可用的编码器。
VIDEO_ENCODERS = ("eegnet", "labram", "reve", "luna", "eegvid_nice")
#: 文本框架可用的编码器。
TEXT_ENCODERS = ("glim", "labram", "luna", "reve")
#: baseline（编码器固定）。
BASELINES = ("atm_full", "avde", "enigma", "dummy",
             "eeg2video_orig", "eegvid_orig", "glim_orig",
             "eeg2text_orig", "rag_orig")
#: 本里程碑已经能跑的模型。
IMPLEMENTED = ("only_clip", "eeg2video", "glim")
#: 已经搬过来的 baseline。
IMPLEMENTED_BASELINES = ("enigma", "avde", "eeg2video_orig", "eegvid_orig",
                         "glim_orig", "eeg2text_orig", "rag_orig")
#: 评测的 8 个指标（方向见下）。
IMAGE_METRICS = ("clip_similarity", "pixcorr", "ssim", "alexnet_2",
                 "alexnet_5", "inception", "swav", "fid")
LOWER_IS_BETTER = frozenset({"swav", "fid"})


def add_noise_arguments(parser: argparse.ArgumentParser) -> None:
    """给推理入口加 ``--noise`` / ``--noise-seed``（默认关闭）。

    共享实现在 :mod:`eeg2x.noise_control`；这里只把全局种子传进去，使噪声种子
    默认与 benchmark 种子一致。
    """
    from .noise_control import add_noise_arguments as _add
    _add(parser, seed=SEED)


def _validate_seed(seed: int) -> None:
    if seed != SEED:
        raise ValueError(f"EEG2X-Benchmark fixes the global seed at {SEED}, got {seed}")
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise RuntimeError(
            f"Launch with PYTHONHASHSEED={SEED}; current value is "
            f"{os.environ.get('PYTHONHASHSEED')!r}")


def _preflight(model: str, encoder: str | None = None) -> None:
    """缺包时在跑任何模型之前明确报错，并指出该换哪份环境说明。"""
    try:
        from .environments import require_environment
        require_environment(model, encoder)
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from exc


def _require_encoder(args: argparse.Namespace, *, model: str | None = None,
                     dataset: DatasetSpec | None = None) -> str:
    """解析 ``--encoder``，并对 baseline 明确报错。

    ``verify`` 子命令没有 ``--model``（它只验框架），所以允许显式传入。
    视频数据集使用另一张编码器名单：两套框架共享 labram/reve/luna 三个名字，
    但各自还有专属编码器（图像 enigma_cnn，视频 eegnet/eegvid_nice）。
    """
    model = model or getattr(args, "model", "only_clip")
    if model in BASELINES:
        if args.encoder is not None:
            raise SystemExit(
                f"error: {model!r} is a baseline -- its encoder is fixed and "
                f"cannot be swapped. Drop --encoder (got {args.encoder!r}).")
        raise SystemExit(
            f"error: baseline {model!r} is not wired in this milestone; "
            f"implemented: {list(IMPLEMENTED)}")
    if model not in FRAMEWORKS:
        raise SystemExit(
            f"error: unknown model {model!r}; frameworks={list(FRAMEWORKS)} "
            f"baselines={list(BASELINES)}")
    if dataset is not None and dataset.task == "video_generation":
        if args.encoder is None:
            raise SystemExit(
                f"error: {model!r} on {dataset.name} needs --encoder; "
                f"available: {list(VIDEO_ENCODERS)}")
        if args.encoder not in VIDEO_ENCODERS:
            raise SystemExit(
                f"error: unknown video encoder {args.encoder!r}; "
                f"available: {list(VIDEO_ENCODERS)}")
        return args.encoder
    if dataset is not None and dataset.task == "sentence_generation":
        if args.encoder is None:
            raise SystemExit(
                f"error: {model!r} on {dataset.name} needs --encoder; "
                f"available: {list(TEXT_ENCODERS)}")
        if args.encoder not in TEXT_ENCODERS:
            raise SystemExit(
                f"error: unknown text encoder {args.encoder!r}; "
                f"available: {list(TEXT_ENCODERS)}")
        return args.encoder
    if args.encoder is None:
        raise SystemExit(
            f"error: {model!r} needs --encoder; available: "
            f"{_encoder_registry().list_encoders()}")
    spec = _encoder_registry().resolve(args.encoder)
    return spec.name


def _subjects_for(dataset: DatasetSpec, requested: Sequence[str]) -> tuple[str, ...]:
    if list(requested) == ["all"]:
        return dataset.subjects
    unknown = [s for s in requested if s not in dataset.subjects]
    if unknown:
        raise SystemExit(
            f"error: {dataset.name} has no subject(s) {unknown}; "
            f"known ({len(dataset.subjects)}): {list(dataset.subjects)[:5]}...")
    return tuple(str(s) for s in requested)


def _run_baseline(args: argparse.Namespace, dataset: DatasetSpec,
                  subjects: Sequence[str]) -> Path:
    """baseline 分派：不接受 --encoder。"""
    _preflight(args.model)
    if args.encoder is not None:
        raise SystemExit(
            f"error: {args.model!r} is a baseline -- its encoder is fixed and "
            f"cannot be swapped. Drop --encoder (got {args.encoder!r}).")
    if args.model not in IMPLEMENTED_BASELINES:
        raise SystemExit(
            f"error: baseline {args.model!r} is not wired yet; "
            f"implemented: {list(IMPLEMENTED_BASELINES)}")
    if args.model == "avde":
        from .baselines.avde.runner import run_inference as avde_inference
        last = None
        for subject in subjects:
            log.info("===== %s / avde / %s =====", dataset.name, subject)
            last = avde_inference(dataset, subject, device=args.device,
                                  overwrite_outputs=args.overwrite,
                                  noise=args.noise, noise_seed=args.noise_seed,
                                  distinct_images=args.distinct_images)
            log.info("evaluation -> %s", last)
        return last
    if args.model == "enigma":
        from .baselines.enigma.runner import run_inference as enigma_inference
        last = None
        for subject in subjects:
            log.info("===== %s / enigma / %s =====", dataset.name, subject)
            last = enigma_inference(dataset, subject, device=args.device,
                                    overwrite_outputs=args.overwrite,
                                    noise=args.noise, noise_seed=args.noise_seed,
                                    distinct_images=args.distinct_images)
            log.info("evaluation -> %s", last)
        return last
    if args.model == "eeg2video_orig":
        if dataset.task != "video_generation":
            raise SystemExit(
                f"error: eeg2video_orig only applies to video datasets; "
                f"{dataset.name!r} is {dataset.task!r}")
        from .baselines.eeg2video_orig.runner import run_inference as paper_inference
        last = None
        for subject in subjects:
            log.info("===== %s / eeg2video_orig / %s =====", dataset.name, subject)
            last = paper_inference(dataset, subject, device=args.device,
                                   noise=args.noise, noise_seed=args.noise_seed,
                                   overwrite=args.overwrite)
            log.info("evaluation -> %s", last)
        return last
    if args.model == "glim_orig":
        if dataset.task != "sentence_generation":
            raise SystemExit(
                f"error: glim_orig only applies to text datasets; "
                f"{dataset.name!r} is {dataset.task!r}")
        from .baselines.glim_orig.runner import run_inference as glim_orig_inference
        log.info("===== %s / glim_orig =====", dataset.name)
        try:
            return glim_orig_inference(
                dataset, device=args.device, noise=args.noise,
                noise_seed=args.noise_seed, overwrite=args.overwrite)
        except KeyError as exc:
            raise SystemExit(f"error: {exc}") from exc
    if args.model in ("eeg2text_orig", "rag_orig"):
        if dataset.task != "sentence_generation":
            raise SystemExit(
                f"error: {args.model} only applies to text datasets; "
                f"{dataset.name!r} is {dataset.task!r}")
        if args.model == "eeg2text_orig":
            from .baselines.eeg2text_orig.runner import run_inference as fn
        else:
            from .baselines.rag_orig.runner import run_inference as fn
        log.info("===== %s / %s =====", dataset.name, args.model)
        try:
            return fn(dataset, device=args.device, noise=args.noise,
                      noise_seed=args.noise_seed, overwrite=args.overwrite)
        except KeyError as exc:
            raise SystemExit(f"error: {exc}") from exc
    if args.model == "eegvid_orig":
        if dataset.task != "video_generation":
            raise SystemExit(
                f"error: eegvid_orig only applies to video datasets; "
                f"{dataset.name!r} is {dataset.task!r}")
        from .baselines.eegvid_orig.runner import (
            available_subjects, run_inference as eegvid_inference,
        )
        allowed = available_subjects(dataset.name)
        if list(args.subjects) == ["all"]:
            chosen = allowed
        else:
            bad = [s for s in subjects if s not in allowed]
            if bad:
                raise SystemExit(
                    f"error: {dataset.name} EEGVid Orig. only retains "
                    f"{list(allowed)}; got {bad}")
            chosen = tuple(subjects)
        last = None
        for subject in chosen:
            log.info("===== %s / eegvid_orig / %s =====", dataset.name, subject)
            last = eegvid_inference(dataset, subject, device=args.device,
                                    noise=args.noise, noise_seed=args.noise_seed,
                                    overwrite=args.overwrite)
            log.info("evaluation -> %s", last)
        return last
    raise SystemExit(f"error: no dispatcher for {args.model!r}")


def _run_video(args: argparse.Namespace, dataset: DatasetSpec,
               subjects: Sequence[str]) -> Path:
    """视频框架：generate -> evaluate。编码器可换，基线在别处。"""
    if args.model != "eeg2video":
        raise SystemExit(
            f"error: {dataset.name!r} is a video dataset; use "
            f"--model eeg2video (got {args.model!r})")
    encoder = _require_encoder(args, model="eeg2video", dataset=dataset)
    _preflight("eeg2video", encoder)
    from .frameworks.eeg2video.pipeline import (
        INFERENCE_STAGES, resolve_paths, run_inference,
    )
    stages = (tuple(INFERENCE_STAGES)
              if args.stages in (["all"], ["inference"]) else tuple(args.stages))
    last: Path | None = None
    for subject in subjects:
        paths = resolve_paths(dataset, encoder, subject)
        log.info("===== %s / eeg2video / %s / %s =====",
                 dataset.name, encoder, subject)
        last = run_inference(
            paths, device=args.device,
            generator_dtype=args.generator_dtype,
            noise=args.noise, noise_seed=args.noise_seed,
            overwrite=args.overwrite, stages=stages)
        log.info("evaluation -> %s", last)
    if last is None:
        raise SystemExit("error: no subject was run")
    return last


def _run_text(args: argparse.Namespace, dataset: DatasetSpec,
              subjects: Sequence[str]) -> Path:
    """文本框架：一次处理该数据集 h5 目录下的整个 test split。"""
    if args.model != "glim":
        raise SystemExit(
            f"error: {dataset.name!r} is a text dataset; use --model glim "
            f"(got {args.model!r})")
    encoder = _require_encoder(args, model="glim", dataset=dataset)
    _preflight("glim", encoder)
    from .frameworks.glim.pipeline import resolve_paths, run_inference
    if list(args.subjects) != ["all"]:
        log.warning("text inference processes the whole test split of %s; "
                    "--subjects is ignored", dataset.name)
    paths = resolve_paths(dataset, encoder)
    return run_inference(paths, device=args.device, noise=args.noise,
                         noise_seed=args.noise_seed, overwrite=args.overwrite)


def _run(args: argparse.Namespace) -> Path:
    dataset = resolve_dataset(args.dataset)
    subjects = _subjects_for(dataset, args.subjects)
    if args.distinct_images is not None and args.noise == "none":
        raise SystemExit(
            "error: --distinct-images only applies to the noise control "
            "(predictions are meaningless there, so reusing images is allowed). "
            "Clean runs must generate every row.")
    if args.model in BASELINES:
        return _run_baseline(args, dataset, subjects)
    if dataset.task == "sentence_generation":
        return _run_text(args, dataset, subjects)
    if dataset.task == "video_generation":
        return _run_video(args, dataset, subjects)
    # 只有图像分支需要 diffusers/SDXL；glim 环境没有 diffusers，所以导入放在这里。
    if args.encoder:
        _preflight(args.model, args.encoder)
    encoder = _require_encoder(args, dataset=dataset)
    _preflight(args.model, encoder)
    from .frameworks.only_clip.pipeline import (
        INFERENCE_STAGES, resolve_paths, run_inference,
    )
    # ``all`` 与 ``inference`` 同义：默认只跑推理四步，训练必须显式请求。
    stages = (tuple(INFERENCE_STAGES)
              if args.stages in (["all"], ["inference"]) else tuple(args.stages))
    last: Path | None = None
    for subject in subjects:
        paths = resolve_paths(dataset, encoder, subject)
        log.info("===== %s / %s / %s =====", dataset.name, encoder, subject)
        last = run_inference(paths, device=args.device, overwrite=args.overwrite,
                             stages=stages, noise=args.noise,
                             noise_seed=args.noise_seed,
                             distinct_images=args.distinct_images,
                             epochs=args.epochs, batch_size=args.batch_size,
                             projection_epochs=args.projection_epochs)
        log.info("last artifact -> %s", last)
    if last is None:
        raise SystemExit("error: no subject was run")
    return last


def _read_metrics(path: Path) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return next(iter(report["datasets"].values()))["metrics"]


def _verify(args: argparse.Namespace) -> int:
    """跑一遍推理，再与 v2 的评测结果逐指标对比。"""
    from .frameworks.only_clip.pipeline import resolve_paths, run_inference

    encoder = _require_encoder(args, model="only_clip")
    dataset = resolve_dataset(args.dataset)
    subject = _subjects_for(dataset, [args.subject])[0]
    paths = resolve_paths(dataset, encoder, subject)
    # verify 必须对照 v2 的权重；本地训练出的 v3 权重会让对照失去意义。
    v3_report = run_inference(paths, device=args.device, overwrite=True,
                              noise=args.noise, noise_seed=args.noise_seed,
                              use_source_checkpoints=True)
    # 噪声条件与 v2 的噪声产物比，干净条件与干净产物比。
    from . import noise_control as _noise
    reference_root = _noise.resolve_noise_dir(paths.source, args.noise)
    reference = reference_root / "evaluation" / "generation_clip.json"
    if not reference.is_file():
        raise SystemExit(f"error: no v2 reference at {reference}")
    got = _read_metrics(v3_report)
    ref = _read_metrics(reference)
    condition = "" if args.noise == "none" else f" [{args.noise}]"
    print(f"\n=== v3 vs v2  {dataset.name} / {encoder} / {subject}{condition} ===")
    print(f"{'metric':18s} {'v2 (reference)':>16s} {'v3':>16s} {'diff':>12s}")
    worst = 0.0
    for name in IMAGE_METRICS:
        a = float(ref[name]); b = float(got[name])
        # fid 是量纲指标，用相对差；其余用绝对差。
        diff = abs(b - a) if name != "fid" else abs(b - a) / max(abs(a), 1e-9)
        worst = max(worst, diff)
        flag = "" if diff <= args.tolerance else "  <-- over tolerance"
        print(f"{name:18s} {a:16.6f} {b:16.6f} {diff:12.2e}{flag}")
    ok = worst <= args.tolerance
    print(f"\n{'PASS' if ok else 'FAIL'}: worst diff {worst:.2e} "
          f"(tolerance {args.tolerance:.0e})")
    return 0 if ok else 1


def _preprocess(args: argparse.Namespace) -> int:
    """跑预处理：一个数据集一个模块，按 profile 分支（见 eeg2x.preprocessing）。"""
    from .preprocessing import preprocess

    dataset = resolve_dataset(args.dataset)
    encoder = _require_encoder(args, dataset=dataset)
    subjects = _subjects_for(dataset, args.subjects)
    # 视频 loader 自己写 <out_dir>/<encoder>/，所以传数据集根，不传编码器目录；
    # 图像分支则是 <encoder>/<subject>。
    batch_task = dataset.task in ("video_generation", "sentence_generation")
    out_root = (V3_ROOT / "h5" / dataset.name
                if batch_task else V3_ROOT / "h5" / dataset.name / encoder)
    for subject in subjects:
        log.info("===== preprocess %s / %s / %s =====", dataset.name, encoder, subject)
        target = out_root if batch_task else out_root / subject
        preprocess(dataset, encoder, subject,
                   out_dir=target,
                   data_root=args.data_root,
                   overwrite=args.overwrite)
    log.info("HDF5 -> %s", out_root)
    return 0


def _retrieval(args: argparse.Namespace) -> int:
    from .retrieval import DATASETS, MODELS, evaluate_run, infer, iter_runs, resolve_run

    if args.list:
        rows = list(iter_runs(args.dataset))
        print(f"datasets ({len(DATASETS)}): {', '.join(sorted(DATASETS))}")
        print(f"models ({len(MODELS)}): {', '.join(MODELS)}")
        print(f"completed runs: {len(rows)}")
        for spec in rows[:20]:
            print(f"  {spec.dataset:18s} {spec.model:20s} seed-{spec.seed}")
        return 0
    if not args.dataset or not args.model:
        raise SystemExit("error: --dataset and --model are required (or use --list)")
    run = Path(args.run_dir) if args.run_dir else resolve_run(
        args.dataset, args.model, args.seed).directory
    if args.mode == "infer":
        try:
            run = infer(args.dataset, args.model, args.seed, device=args.device)
        except (RuntimeError, FileNotFoundError) as exc:
            raise SystemExit(f"error: {exc}") from exc
    result = evaluate_run(run)
    from .data import write_json
    from .paths import V3_ROOT, require_v3_output
    output = require_v3_output(
        V3_ROOT / "artifacts" / "retrieval" / args.dataset / args.model
        / f"seed-{args.seed}.json")
    write_json(output, result)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    print(f"=== retrieval {args.dataset} / {args.model} / seed-{args.seed} ===")
    print(f"  top1   = {result['test/full_retrieval/top1_acc_subject-agg']:.4f}")
    print(f"  top5   = {result['test/full_retrieval/top5_acc_subject-agg']:.4f}")
    print(f"  top10  = {result['test/full_retrieval/top10_acc_subject-agg']:.4f}")
    print(f"  median = {result['test/full_retrieval/median_rank_subject-agg']:.4f}")
    if "reference" in result:
        label = "match" if result["matches_reference"] else "DIFF"
        print(f"  vs metrics.json: max diff {result['max_abs_diff']:.2e} ({label})")
    print(f"  -> {output}")
    return 0


def _env(args: argparse.Namespace) -> int:
    from .environments import report

    data = report()
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    print(f"interpreter: {data['executable']}")
    print(f"python:      {data['python']}")
    print("environments:")
    for name, info in data["environments"].items():
        mark = "OK    " if info["ok"] else "MISSING"
        missing = "" if info["ok"] else "  " + ", ".join(info["missing"])
        print(f"  {name:18s} {mark}{missing}")
    print("runnable here:")
    for key, info in data["runnable"].items():
        mark = "yes" if info["ok"] else "no "
        print(f"  {key:22s} {mark}  envs={info['environments']}")
    return 0


def _list(_args: argparse.Namespace) -> int:
    print("frameworks (encoder swappable):")
    for name in FRAMEWORKS:
        mark = "" if name in IMPLEMENTED else "   [not wired yet]"
        print(f"  {name}{mark}")
    print("baselines (encoder fixed):")
    for name in BASELINES:
        mark = "" if name in IMPLEMENTED_BASELINES else "   [not wired yet]"
        print(f"  {name}{mark}")
    print("encoders:")
    for name in _encoder_registry().list_encoders():
        spec = _encoder_registry().ENCODERS[name]
        if spec.optimizer is None:
            opt = "framework default"
        elif spec.optimizer_datasets is None:
            opt = "own optimizer (all datasets)"
        else:
            opt = f"own optimizer ({', '.join(spec.optimizer_datasets)} only)"
        print(f"  {name:12s} feature_dim={str(spec.feature_dim):5s}  {opt}")
    print("video encoders:")
    print(f"  {', '.join(VIDEO_ENCODERS)}")
    print("datasets:")
    for spec in DATASETS.values():
        print(f"  {spec.name:18s} {len(spec.subjects):2d} subjects  [{spec.task}]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="跑一条链路")
    run.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    run.add_argument("--model", required=True, choices=[*FRAMEWORKS, *BASELINES])
    run.add_argument("--encoder", default=None)
    run.add_argument("--subjects", nargs="+", default=["all"])
    run.add_argument("--stages", nargs="+", default=["all"],
                     choices=["all", "inference", "embed", "project",
                              "generate", "evaluate", "train", "preprocess"])
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--seed", type=int, default=SEED)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--generator-dtype", default="float16",
                     choices=["float32", "fp32", "float16", "fp16",
                              "bfloat16", "bf16"],
                     help="视频生成器的精度；已发表的视频结果来自 fp16")
    add_noise_arguments(run)
    run.add_argument("--epochs", type=int, default=40,
                     help="train 阶段：编码器 + ATM 头的轮数")
    run.add_argument("--batch-size", type=int, default=64,
                     help="train 阶段：编码器训练的 batch size")
    run.add_argument("--projection-epochs", type=int, default=150,
                     help="train 阶段：Linear 投影头的轮数")
    run.add_argument(
        "--distinct-images", type=int, default=None,
        help="噪声对照省时档：只真出前 N 张图，其余复用第 (row %% N) 张。"
             "干净条件不允许——干净条件下每行是刺激特异的，不能复制")

    verify = sub.add_parser("verify", help="跑推理并与 v2 的评测结果对比")
    verify.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    verify.add_argument("--encoder", required=True)
    verify.add_argument("--subject", required=True)
    verify.add_argument("--device", default="cuda:0")
    add_noise_arguments(verify)
    verify.add_argument("--seed", type=int, default=SEED)
    verify.add_argument("--tolerance", type=float, default=2e-2,
                        help="absolute-diff tolerance (relative for fid). Cross-machine "
                             "fp drift measured at 8.5e-4 (embedding) -> 7.8e-3 (projection) "
                             "-> 1.2e-2 (inception, worst of 5 encoders); see requirements-v3.txt")
    verify.add_argument("--against", default="v2", choices=["v2"])

    pre = sub.add_parser("preprocess", help="按 profile 生成 HDF5")
    pre.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    pre.add_argument("--encoder", required=True)
    pre.add_argument("--subjects", nargs="+", default=["all"])
    pre.add_argument("--data-root", default=None)
    pre.add_argument("--seed", type=int, default=SEED)
    pre.add_argument("--overwrite", action="store_true")

    ret = sub.add_parser("retrieval", help="EEG 检索：重算或重跑 EEGBench 轨道")
    ret.add_argument("--dataset", default=None)
    ret.add_argument("--model", default=None)
    ret.add_argument("--seed", type=int, default=33)
    ret.add_argument("--mode", choices=["evaluate", "infer"], default="evaluate")
    ret.add_argument("--run-dir", default=None)
    ret.add_argument("--device", default="cuda")
    ret.add_argument("--list", action="store_true")
    ret.add_argument("--json", action="store_true")

    env = sub.add_parser("env", help="打印当前解释器可用的环境、包版本与模型")
    env.add_argument("--json", action="store_true", help="输出机器可读的 JSON")

    sub.add_parser("list", help="列出可用的模型/编码器/数据集")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    if args.command == "list":
        return _list(args)
    if args.command == "env":
        return _env(args)
    if args.command == "retrieval":
        return _retrieval(args)
    _validate_seed(args.seed)
    if args.command == "run":
        _run(args)
        return 0
    if args.command == "preprocess":
        return _preprocess(args)
    if args.command == "verify":
        return _verify(args)
    raise SystemExit(f"error: unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
