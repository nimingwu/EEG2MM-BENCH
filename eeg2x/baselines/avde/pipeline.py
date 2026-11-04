"""AVDE baseline orchestration for the EEG2X benchmark.

AVDE is a *baseline*: it keeps its own LaBraM encoder and its own VAR decoder
and is never combined with the swappable encoders of the benchmark's Only-CLIP
framework.  The two training stages run as ``torchrun --nproc_per_node=1``
subprocesses so the copied upstream trainers keep their distributed control
flow (AVDE calls ``torch.distributed`` barriers unconditionally) without any
single-process special case in the copied files.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from eeg2x import noise_control
from eeg2x.rawdata import BENCHMARK_ROOT, require_benchmark_output
from eeg2x.data import discover_subject_h5, h5_fingerprint, write_json
from eeg2x.targets.clip import extract_features
from eeg2x.baselines.avde import bootstrap


log = logging.getLogger(__name__)

SEED = 2000
#: Paper values (Tables 5 and 6 / Section 3.1).
STAGE1_EPOCHS = 50
STAGE1_BATCH_SIZE = 128
STAGE1_LR = 2e-3
STAGE2_EPOCHS = 30
STAGE2_BS = 512
STAGE2_TBLR = 2e-5
STAGE2_WPE = 0.1
STAGE2_ALNG = 1e-3
CFG_SCALE = 4.0
TOP_K = 900


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _run_stage(name: str, script: Path, arguments: list, *, log_path: Path,
               timeout: float) -> None:
    """Run one ``torchrun`` stage, streaming its output into the artifact tree.

    The caller has already decided the stage must run (its checkpoint is
    missing, or ``--overwrite`` was given), so the log is truncated rather than
    guarded: retrying a stage that crashed must not require redoing the stages
    that succeeded.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # ``python -m torch.distributed.run`` rather than the ``torchrun`` console
    # script: the AVDE environment is a relocated copy of a conda env, whose
    # console-script shebangs still point at the original prefix.
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
        "--nproc_per_node=1", "--master_port", str(_free_port()),
        str(script), *map(str, arguments),
    ]
    log.info("%s: %s", name, " ".join(command))
    started = time.time()
    with log_path.open("w", encoding="utf-8") as sink:
        sink.write("# " + " ".join(command) + "\n")
        sink.flush()
        result = subprocess.run(
            command, stdout=sink, stderr=subprocess.STDOUT, cwd=str(BENCHMARK_ROOT),
            env=bootstrap.torchrun_env(_free_port()), timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {result.returncode}; see {log_path}")
    log.info("%s finished in %.1f min", name, (time.time() - started) / 60.0)


def run_avde(
    *,
    dataset: str,
    subject: str,
    profile_id: str,
    h5_dir: str | Path,
    artifact_root: str | Path,
    feature_store: str | Path,
    device: str = "cuda:0",
    seed: int = SEED,
    overwrite: bool = False,
    stage1_epochs: int = STAGE1_EPOCHS,
    stage1_batch_size: int = STAGE1_BATCH_SIZE,
    stage1_lr: float = STAGE1_LR,
    stage1_workers: int = 8,
    stage2_epochs: int = STAGE2_EPOCHS,
    stage2_bs: int = STAGE2_BS,
    #: Micro-batch = ``stage2_bs / stage2_ac``.  The paper's effective batch is
    #: 512 and its learning rate is ``bs * tblr / 256``, so ``ac`` only splits
    #: the batch: 8 x 64 is the released run's own setting and fits a single
    #: 80 GB A100 (8 x 128 exhausts it during the first backward).
    stage2_ac: int = 8,
    stage2_workers: int = 6,
    stage2_tblr: float = STAGE2_TBLR,
    stage2_wpe: float = STAGE2_WPE,
    stage2_alng: float = STAGE2_ALNG,
    cfg_scale: float = CFG_SCALE,
    top_k: int = TOP_K,
    timeout: float = 24 * 3600.0,
    noise: str = "none",
    noise_seed: int = SEED,
    distinct_images: int | None = None,
) -> Path:
    """Stage 1 -> Stage 2 -> autoregressive reconstruction -> 8-metric report.

    ``noise="gaussian"`` 是噪声对照：两个阶段的 checkpoint 仍从干净根目录读
    （``overwrite=False``，绝不重训），只有测试集 EEG 被替换；生成与评测产物写
    到 ``root/noise_control/``，干净结果不可能被覆盖。
    """
    if seed != SEED:
        raise ValueError(
            f"EEG2X-Benchmark fixes the global seed at {SEED}, got {seed}")
    bootstrap.require_pretrained()
    root = require_benchmark_output(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    # 噪声条件的输出根：checkpoint / 训练目录仍在 root，生成与评测走独立子目录。
    output_root = noise_control.resolve_noise_dir(root, noise)
    if noise_control.noise_enabled(noise):
        output_root = require_benchmark_output(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
    h5_dir = require_benchmark_output(h5_dir)
    store = require_benchmark_output(feature_store)

    train_h5 = discover_subject_h5(h5_dir, subject, profile_id=profile_id, split="train")
    test_h5 = discover_subject_h5(h5_dir, subject, profile_id=profile_id, split="test")

    # Frozen CLIP image targets: the benchmark's store is the released
    # OpenCLIP ViT-H-14 encode_image output with the training transform and no
    # L2 normalization, which is exactly what AVDE's release uses.
    extract_features(
        [*train_h5, *test_h5], store, kinds=("clip_vit_h",),
        batch_size=32, device=device, overwrite=overwrite, seed=seed)

    stage1_dir = root / "stage1"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage1_best = stage1_dir / "checkpoint-best.pth"
    stage1_final = stage1_dir / "checkpoint-final.pth"
    if overwrite or not (stage1_best.is_file() or stage1_final.is_file()):
        # A rerun must not be able to fall back on the previous run's weights.
        for stale in (stage1_best, stage1_final):
            if stale.is_file():
                stale.unlink()
        _run_stage(
            "AVDE stage 1 (LaBraM contrastive fine-tuning)",
            bootstrap.AVDE_ROOT / "run_labram_finetuning.py",
            [
                "--output_dir", stage1_dir,
                "--description", f"avde_stage1_{dataset}_{subject}",
                "--model", "labram_base_patch200_200",
                "--finetune", bootstrap.LABRAM_CKPT,
                "--weight_decay", 0.05,
                "--batch_size", stage1_batch_size,
                "--lr", stage1_lr,
                "--update_freq", 1,
                "--warmup_epochs", 5,
                "--epochs", stage1_epochs,
                "--layer_decay", 0.8,
                "--drop_path", 0.1,
                "--dist_eval",
                "--save_ckpt_freq", 5,
                "--disable_rel_pos_bias",
                "--abs_pos_emb",
                "--disable_qkv_bias",
                "--seed", seed,
                "--sub", subject,
                "--num_workers", stage1_workers,
                "--no_auto_resume",
                "--bench_h5_dir", h5_dir,
                "--bench_profile", profile_id,
                "--bench_features", store,
            ],
            log_path=root / "logs" / "stage1_torchrun.log", timeout=timeout)
    # Upstream writes "best" only when the 200-way validation accuracy improves;
    # Stage 1 additionally always writes the final epoch, so a run in which no
    # epoch beat the initial 0 still yields a usable encoder.
    stage1_checkpoint = stage1_best if stage1_best.is_file() else stage1_final
    if not stage1_checkpoint.is_file():
        raise FileNotFoundError(
            f"AVDE stage 1 produced neither {stage1_best.name} nor "
            f"{stage1_final.name} in {stage1_dir}")

    stage2_dir = root / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    stage2_checkpoint = stage2_dir / "ar-ckpt-best.pth"
    if overwrite or not stage2_checkpoint.is_file():
        for stale in (stage2_dir / "ar-ckpt-best.pth", stage2_dir / "ar-ckpt-last.pth"):
            if stale.is_file():
                stale.unlink()
        _run_stage(
            "AVDE stage 2 (VAR next-scale prediction)",
            bootstrap.AVDE_ROOT / "train_eeg.py",
            [
                "--depth", 16,
                "--bs", stage2_bs,
                "--ac", stage2_ac,
                "--workers", stage2_workers,
                "--ep", stage2_epochs,
                "--fp16", 1,
                "--alng", stage2_alng,
                "--wpe", stage2_wpe,
                "--tblr", stage2_tblr,
                "--labram_ckpt", stage1_checkpoint,
                "--local_out_dir_path", stage2_dir,
                "--sub", subject,
                "--seed", seed,
                "--bench_h5_dir", h5_dir,
                "--bench_profile", profile_id,
                "--bench_features", store,
            ],
            log_path=root / "logs" / "stage2_torchrun.log", timeout=timeout)
    if not stage2_checkpoint.is_file():
        raise FileNotFoundError(
            f"AVDE stage 2 produced no {stage2_checkpoint.name} in {stage2_dir}")

    from eeg2x.baselines.avde.inference import generate_stage

    predictions = generate_stage(
        subject_id=subject, h5_dir=h5_dir, test_h5=test_h5[0],
        profile_id=profile_id, feature_store=store,
        labram_checkpoint=stage1_checkpoint,
        var_checkpoint=stage2_checkpoint, output_dir=output_root / "generation",
        dataset=dataset, cfg_scale=cfg_scale, top_k=top_k, seed=seed,
        device=device, overwrite=overwrite,
        noise=noise, noise_seed=noise_seed,
        distinct_images=distinct_images)

    from eeg2x.evaluation.api import evaluate_predictions

    evaluation = evaluate_predictions(
        task="image_generation", h5_dir=h5_dir, predictions_path=predictions,
        output_path=output_root / "evaluation" / "generation_avde.json",
        metrics=None)
    if noise_control.noise_enabled(noise):
        # 与 ENIGMA / 框架一致，在噪声根目录写一份条件溯源；generate_stage
        # 另外在 generation/ 下写一份更细的（v2 当时也是两份）。
        noise_control.write_noise_meta(
            output_root / "noise_meta.json",
            noise_control.noise_meta(
                noise, noise_seed, model="avde", dataset=dataset,
                subject=subject, profile_id=profile_id,
                stage1_checkpoint=str(stage1_checkpoint),
                stage2_checkpoint=str(stage2_checkpoint),
                clean_root=str(root), evaluation=str(evaluation),
                distinct_images=distinct_images))

    write_json(output_root / "run_config.json", {
        "experiment_id": root.name, "dataset": dataset, "subject_id": subject,
        "task": "image_generation", "encoder": "labram", "model": "avde",
        "preprocessing_profile": profile_id,
        "pipeline": ("EEG -> LaBraM (contrastive fine-tune) -> 1024-d embedding -> "
                     "VAR next-scale prediction over 10 scales -> multi-scale "
                     "VQ-VAE decoder -> 256x256 image"),
        "stage1_epochs": stage1_epochs, "stage1_batch_size": stage1_batch_size,
        "stage1_lr": stage1_lr,
        "stage1_checkpoint_used": stage1_checkpoint.name,
        "stage2_epochs": stage2_epochs, "stage2_bs": stage2_bs,
        "stage2_accumulation": stage2_ac,
        "stage2_effective_batch": stage2_bs,
        "stage2_micro_batch": max(1, round(stage2_bs / max(1, stage2_ac))),
        "stage2_base_lr": stage2_tblr, "stage2_wpe": stage2_wpe,
        "stage2_alng": stage2_alng,
        "cfg": cfg_scale, "top_k": top_k,
        "seed": seed, "pythonhashseed": seed,
        "source_h5": [h5_fingerprint(path) for path in (*train_h5, *test_h5)],
        "derived_inputs_used": False,
    })
    log.info("AVDE %s/%s complete: %s", dataset, subject, evaluation)
    return evaluation


def main(argv: list | None = None) -> int:
    """Allow ``python -m model.avde.pipeline --dataset ...`` for smoke tests."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS)
    parser.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    parser.add_argument("--stage2-ac", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    run_avde(
        dataset=args.dataset, subject=args.subject, profile_id=args.profile_id,
        h5_dir=args.h5_dir, artifact_root=args.artifact_root,
        feature_store=args.feature_store, device=args.device,
        overwrite=args.overwrite, stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs, stage2_ac=args.stage2_ac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
