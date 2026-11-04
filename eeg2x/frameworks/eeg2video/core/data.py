"""Subject-dependent preprocessing and frozen Stable Diffusion target extraction."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import (
    BLOCK_NAMES,
    DATA_ROOT,
    PROJECT_ROOT,
    RUN_ROOT,
    SHARED_RUN_ROOT,
    SD14_ROOT,
    SETTINGS,
)


MANIFEST_PATH = RUN_ROOT / "manifest.jsonl"
RAW_EEG_PATH = RUN_ROOT / "eeg_segments.float32.npy"
WINDOW_EEG_PATH = RUN_ROOT / "eeg_windows.float32.npy"
FEATURE_WINDOW_EEG_PATH = RUN_ROOT / "eeg_feature_windows.float32.npy"
DE_PATH = RUN_ROOT / "de_features.float32.npy"
SEMANTIC_DE_PATH = (
    DE_PATH
    if SETTINGS.semantic_feature_mode == "mean_overlap_3"
    else RUN_ROOT / f"de_features.{SETTINGS.semantic_feature_mode}.float32.npy"
)
VIDEO_FRAMES_PATH = SHARED_RUN_ROOT / "video_frames.uint8.npy"
OPENCV_TEST_VIDEO_FRAMES_PATH = SHARED_RUN_ROOT / "video_frames_test_opencv.uint8.npy"
VIDEO_LATENTS_PATH = SHARED_RUN_ROOT / "video_latents.float16.npy"
TEXT_EMBEDDINGS_PATH = SHARED_RUN_ROOT / "text_embeddings.float16.npy"
UNCONDITIONAL_EMBEDDING_PATH = SHARED_RUN_ROOT / "unconditional_embedding.float16.npy"
PREPARE_MARKER = RUN_ROOT / "prepare.complete.json"
EMBEDDING_MARKER = SHARED_RUN_ROOT / "embeddings.complete.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_output_roots() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    SHARED_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    project_cache = PROJECT_ROOT / ".hf-cache"
    project_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(project_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(project_cache / "transformers"))


def _captions(block_index: int) -> list[str]:
    local = PROJECT_ROOT / "dataset" / "BLIP" / f"{BLOCK_NAMES[block_index]}.txt"
    source = DATA_ROOT / "Video" / "BLIP-caption" / f"{BLOCK_NAMES[block_index]}.txt"
    path = local if local.exists() else source
    captions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    captions = [caption for caption in captions if caption]
    if len(captions) != SETTINGS.samples_per_block:
        raise ValueError(f"Expected 200 captions in {path}, found {len(captions)}")
    return captions


def build_manifest() -> list[dict]:
    """Build the exact 6-train-block/1-test-block sample alignment."""
    ensure_output_roots()
    import decord

    labels = np.load(DATA_ROOT / "Video" / "meta-info" / "All_video_label.npy")
    optical_flow = np.load(
        DATA_ROOT / "Video" / "meta-info" / "All_video_optical_flow_score.npy"
    )
    expected_labels = (7, SETTINGS.concepts)
    expected_flow = (7, SETTINGS.samples_per_block)
    if labels.shape != expected_labels or optical_flow.shape != expected_flow:
        raise ValueError(
            f"Metadata shapes are {labels.shape}/{optical_flow.shape}, expected "
            f"{expected_labels}/{expected_flow}"
        )

    rows: list[dict] = []
    for block_index, block_name in enumerate(BLOCK_NAMES):
        video_path = DATA_ROOT / "Video" / f"{block_name}.mp4"
        reader = decord.VideoReader(str(video_path))
        source_fps = float(reader.get_avg_fps())
        captions = _captions(block_index)
        for class_position in range(SETTINGS.concepts):
            for repeat in range(SETTINGS.repeats_per_concept):
                within_block = class_position * SETTINGS.repeats_per_concept + repeat
                sample_id = block_index * SETTINGS.samples_per_block + within_block
                start_seconds = float(class_position * 13 + 3 + repeat * 2)
                frame_times = start_seconds + (
                    np.arange(SETTINGS.video_frames, dtype=np.float64) / SETTINGS.video_fps
                )
                # The released extraction script indexes the nominal 24-FPS
                # source timeline (8 source frames per 3-FPS output frame),
                # rather than the container's 23.976 average-FPS metadata.
                frame_indices = (
                    start_seconds * 24.0
                    + 8 * np.arange(SETTINGS.video_frames, dtype=np.float64)
                ).astype(np.int64)
                if frame_indices[-1] >= len(reader):
                    raise ValueError(f"Frame index exceeds {video_path}: {frame_indices[-1]}")
                flow = float(optical_flow[block_index, within_block])
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subject": SETTINGS.subject,
                        "block": block_index + 1,
                        "within_block": within_block,
                        "class_position": class_position,
                        "concept_one_based": int(labels[block_index, class_position]),
                        "concept": int(labels[block_index, class_position] - 1),
                        "repeat": repeat,
                        "start_seconds": start_seconds,
                        "source_fps": source_fps,
                        "source_video": str(video_path),
                        "frame_times_seconds": frame_times.tolist(),
                        "frame_indices": frame_indices.tolist(),
                        "caption": captions[within_block],
                        "optical_flow_score": flow,
                        "dynamic_label": int(flow >= SETTINGS.optical_flow_threshold),
                        "dynamic_name": "fast"
                        if flow >= SETTINGS.optical_flow_threshold
                        else "slow",
                        "split": "train"
                        if block_index < SETTINGS.train_blocks
                        else "test",
                    }
                )

    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return rows


def read_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return build_manifest()
    return [
        json.loads(line)
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _extract_eeg(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_path = DATA_ROOT / "EEG" / f"sub{SETTINGS.subject}.npy"
    source = np.load(source_path, mmap_mode="r")
    if source.shape != (7, SETTINGS.eeg_channels, 104000):
        raise ValueError(
            f"Unexpected Subject {SETTINGS.subject:02d} EEG shape from "
            f"{source_path}: {source.shape}"
        )
    raw = np.lib.format.open_memmap(
        RAW_EEG_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(rows),
            SETTINGS.eeg_channels,
            SETTINGS.sampling_rate * SETTINGS.clip_seconds,
        ),
    )
    windows = np.lib.format.open_memmap(
        WINDOW_EEG_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(rows),
            SETTINGS.eeg_tokens,
            SETTINGS.eeg_channels,
            SETTINGS.eeg_window_samples,
        ),
    )
    feature_windows = np.lib.format.open_memmap(
        FEATURE_WINDOW_EEG_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(rows),
            SETTINGS.feature_windows,
            SETTINGS.eeg_channels,
            SETTINGS.feature_window_samples,
        ),
    )
    relative_starts = np.arange(SETTINGS.eeg_tokens) * SETTINGS.eeg_stride_samples
    feature_starts = (
        np.arange(SETTINGS.feature_windows) * SETTINGS.feature_stride_samples
    )
    for row in rows:
        start = int(round(row["start_seconds"] * SETTINGS.sampling_rate))
        segment = np.asarray(
            source[row["block"] - 1, :, start : start + SETTINGS.sampling_rate * 2],
            dtype=np.float32,
        )
        if segment.shape != (SETTINGS.eeg_channels, SETTINGS.sampling_rate * 2):
            raise ValueError(f"Bad EEG segment for sample {row['sample_id']}: {segment.shape}")
        raw[row["sample_id"]] = segment
        windows[row["sample_id"]] = np.stack(
            [
                segment[:, offset : offset + SETTINGS.eeg_window_samples]
                for offset in relative_starts
            ],
            axis=0,
        )
        feature_windows[row["sample_id"]] = np.stack(
            [
                segment[:, offset : offset + SETTINGS.feature_window_samples]
                for offset in feature_starts
            ],
            axis=0,
        )
    raw.flush()
    windows.flush()
    feature_windows.flush()
    return raw, windows, feature_windows


def _extract_de_for_windows(
    feature_windows: np.ndarray,
    window_indices: tuple[int, ...],
    output_path: Path,
) -> np.ndarray:
    """Extract five-band DE and average the selected one-second windows."""
    if not window_indices:
        raise ValueError("At least one DE window must be selected")
    if min(window_indices) < 0 or max(window_indices) >= feature_windows.shape[1]:
        raise ValueError(
            f"DE window indices {window_indices} are invalid for "
            f"shape {feature_windows.shape}"
        )
    samples = SETTINGS.feature_window_samples
    window_index = np.arange(1, samples + 1, dtype=np.float64)
    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * window_index / (samples + 1))
    # Channel count comes from the data, not from SETTINGS: this helper is
    # shared with models running on datasets whose montage is not SEED-DV's 62
    # (ds005662 has 64), and allocating from the constant made the write below
    # fail with a broadcast error that says nothing about montages.  For
    # SEED-DV the two are equal, so this is a no-op there.
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(feature_windows.shape[0], feature_windows.shape[2], 5),
    )
    starts = (1, 4, 8, 14, 31)
    ends = (4, 8, 14, 31, 99)
    for begin in range(0, feature_windows.shape[0], 32):
        batch = np.asarray(
            feature_windows[begin : begin + 32, window_indices], dtype=np.float64
        )
        # Match the released DE_PSD.py index arithmetic exactly: each band
        # starts at fStartNum - 1 and includes the fEndNum - 1 bin.
        spectrum = np.abs(np.fft.fft(batch * hanning, n=200, axis=-1)[..., :100])
        de = np.empty((*batch.shape[:3], 5), dtype=np.float64)
        for band, (start, end) in enumerate(zip(starts, ends)):
            bin_indices = np.arange(start - 1, end, dtype=np.int64)
            bins = spectrum[..., bin_indices]
            energy = np.square(bins).sum(axis=-1) / (end - start + 1)
            de[..., band] = np.log2(100.0 * energy)
        output[begin : begin + batch.shape[0]] = de.mean(axis=1).astype(np.float32)
    output.flush()
    return output


def _extract_de(feature_windows: np.ndarray) -> np.ndarray:
    """Extract five-band DE from the paper's three overlapping windows."""
    return _extract_de_for_windows(feature_windows, (0, 1, 2), DE_PATH)


def semantic_de_features() -> np.ndarray:
    """Return the explicitly selected 310-dimensional semantic DE protocol."""
    expected_shape = (SETTINGS.train_samples + SETTINGS.test_samples, 62, 5)
    if SEMANTIC_DE_PATH.exists():
        existing = np.load(SEMANTIC_DE_PATH, mmap_mode="r")
        if existing.shape != expected_shape or existing.dtype != np.float32:
            raise ValueError(
                f"Semantic DE cache has shape/dtype {existing.shape}/{existing.dtype}, "
                f"expected {expected_shape}/float32: {SEMANTIC_DE_PATH}"
            )
        return existing
    if SETTINGS.semantic_feature_mode != "mean_nonoverlap_2":
        raise ValueError(
            f"Unsupported semantic feature mode: {SETTINGS.semantic_feature_mode}"
        )
    if not FEATURE_WINDOW_EEG_PATH.exists():
        prepare()
    feature_windows = np.load(FEATURE_WINDOW_EEG_PATH, mmap_mode="r")
    return _extract_de_for_windows(feature_windows, (0, 2), SEMANTIC_DE_PATH)


def _extract_video_frames(rows: list[dict]) -> np.ndarray:
    import decord

    expected_shape = (
        len(rows),
        SETTINGS.video_frames,
        SETTINGS.video_height,
        SETTINGS.video_width,
        3,
    )
    if VIDEO_FRAMES_PATH.exists():
        existing = np.load(VIDEO_FRAMES_PATH, mmap_mode="r")
        if existing.shape != expected_shape:
            raise ValueError(
                f"Shared video frame cache has shape {existing.shape}, "
                f"expected {expected_shape}: {VIDEO_FRAMES_PATH}"
            )
        return existing

    decord.bridge.set_bridge("native")
    output = np.lib.format.open_memmap(
        VIDEO_FRAMES_PATH,
        mode="w+",
        dtype=np.uint8,
        shape=expected_shape,
    )
    for block_index, block_name in enumerate(BLOCK_NAMES):
        block_rows = rows[
            block_index
            * SETTINGS.samples_per_block : (block_index + 1)
            * SETTINGS.samples_per_block
        ]
        reader = decord.VideoReader(
            str(DATA_ROOT / "Video" / f"{block_name}.mp4"),
            width=SETTINGS.video_width,
            height=SETTINGS.video_height,
        )
        for local_start in range(0, SETTINGS.samples_per_block, 16):
            batch_rows = block_rows[local_start : local_start + 16]
            indices = np.asarray(
                [index for row in batch_rows for index in row["frame_indices"]],
                dtype=np.int64,
            )
            frames = reader.get_batch(indices)
            frames = frames.asnumpy() if hasattr(frames, "asnumpy") else frames.numpy()
            frames = frames.reshape(
                len(batch_rows),
                SETTINGS.video_frames,
                SETTINGS.video_height,
                SETTINGS.video_width,
                3,
            )
            global_start = block_index * SETTINGS.samples_per_block + local_start
            output[global_start : global_start + len(batch_rows)] = frames
        output.flush()
        print(f"video preprocessing: block {block_index + 1}/7", flush=True)
    return output


def ensure_opencv_test_frames(force: bool = False) -> np.ndarray:
    """Materialize block-7 GT frames using the released OpenCV/GIF path."""
    if OPENCV_TEST_VIDEO_FRAMES_PATH.exists() and not force:
        return np.load(OPENCV_TEST_VIDEO_FRAMES_PATH, mmap_mode="r")
    import cv2

    rows = read_manifest()[SETTINGS.train_samples :]
    wanted = {
        int(frame_index): (sample_id, frame_id)
        for sample_id, row in enumerate(rows)
        for frame_id, frame_index in enumerate(row["frame_indices"])
    }
    output = np.lib.format.open_memmap(
        OPENCV_TEST_VIDEO_FRAMES_PATH,
        mode="w+",
        dtype=np.uint8,
        shape=(
            SETTINGS.test_samples,
            SETTINGS.video_frames,
            SETTINGS.video_height,
            SETTINGS.video_width,
            3,
        ),
    )
    video_path = DATA_ROOT / "Video" / f"{BLOCK_NAMES[SETTINGS.test_block - 1]}.mp4"
    capture = cv2.VideoCapture(str(video_path))
    max_frame = max(wanted)
    for frame_index in range(max_frame + 1):
        success, frame = capture.read()
        if not success:
            capture.release()
            raise RuntimeError(f"OpenCV failed to read frame {frame_index} from {video_path}")
        hit = wanted.get(frame_index)
        if hit is not None:
            sample_id, frame_id = hit
            rgb = frame[..., ::-1]
            output[sample_id, frame_id] = cv2.resize(
                rgb,
                (SETTINGS.video_width, SETTINGS.video_height),
                interpolation=cv2.INTER_LINEAR,
            )
    capture.release()
    output.flush()
    return output


def prepare(force: bool = False) -> None:
    ensure_output_roots()
    required = (
        MANIFEST_PATH,
        RAW_EEG_PATH,
        WINDOW_EEG_PATH,
        FEATURE_WINDOW_EEG_PATH,
        DE_PATH,
        VIDEO_FRAMES_PATH,
    )
    if PREPARE_MARKER.exists() and all(path.exists() for path in required) and not force:
        print(f"paper preprocessing already complete: {PREPARE_MARKER}")
        return
    rows = build_manifest()
    raw, windows, feature_windows = _extract_eeg(rows)
    de = _extract_de(feature_windows)
    frames = _extract_video_frames(rows)
    summary = {
        "paper": "NeurIPS 2024 EEG2Video",
        "table": 2,
        "classes": 40,
        "model": "Full Model",
        "subject": SETTINGS.subject,
        "settings": SETTINGS.as_dict(),
        "shapes": {
            "raw_eeg": list(raw.shape),
            "window_eeg": list(windows.shape),
            "feature_window_eeg": list(feature_windows.shape),
            "de_features": list(de.shape),
            "video_frames": list(frames.shape),
        },
        "train_rule": "blocks 1-6",
        "test_rule": "block 7",
        "paper_pages": [7, 15],
    }
    PREPARE_MARKER.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"paper preprocessing complete: {RUN_ROOT}")


def _torch_generator(device: str, seed: int):
    import torch

    return torch.Generator(device=device).manual_seed(seed)


def extract_embeddings(force: bool = False) -> None:
    """Create the per-video CLIP text targets used by Scheme D.

    ``EEG2Video_clean_d`` also produced VAE video latents and an unconditional
    embedding here.  Both were only consumed by the paper-literal Seq2Seq /
    Tune-A-Video path, which Scheme D does not use and which has been removed
    from this project, so only the text targets are built now.  The text
    targets do not depend on any random seed (CLIP encoding is deterministic),
    but the seed is still fixed so the whole pipeline shares one value.
    """
    ensure_output_roots()
    if not PREPARE_MARKER.exists():
        prepare()
    if EMBEDDING_MARKER.exists() and not force:
        print(f"paper embeddings already complete: {EMBEDDING_MARKER}")
        return
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    seed_everything(SETTINGS.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    text_encoder = CLIPTextModel.from_pretrained(
        str(SD14_ROOT), subfolder="text_encoder", local_files_only=True
    ).to(device=device, dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(
        str(SD14_ROOT), subfolder="tokenizer", local_files_only=True
    )
    text_encoder.eval()

    rows = read_manifest()
    text_embeddings = np.lib.format.open_memmap(
        TEXT_EMBEDDINGS_PATH,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), 77, 768),
    )
    with torch.no_grad():
        for begin in range(0, len(rows), 32):
            captions = [row["caption"] for row in rows[begin : begin + 32]]
            token_ids = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            encoded = text_encoder(token_ids)[0]
            text_embeddings[begin : begin + encoded.shape[0]] = (
                encoded.float().cpu().numpy().astype(np.float16)
            )
    text_embeddings.flush()
    marker = {
        "caption_target": "one released BLIP caption per video clip",
        "text_embeddings_shape": list(text_embeddings.shape),
        "seed": SETTINGS.seed,
        "base_model": str(SD14_ROOT),
        "note": "VAE video latents are not produced in this project; unused by Scheme D.",
    }
    EMBEDDING_MARKER.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print(f"text embedding targets complete: {SHARED_RUN_ROOT}")


class ArrayDataset:
    """Lazy numpy-backed dataset used by the three predictor trainers."""

    def __init__(self, *arrays: np.ndarray) -> None:
        lengths = {len(array) for array in arrays}
        if len(lengths) != 1:
            raise ValueError(f"Dataset arrays have different lengths: {lengths}")
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays[0])

    def __getitem__(self, index: int):
        import torch

        return tuple(torch.from_numpy(np.asarray(array[index])) for array in self.arrays)


class TuneVideoDataset:
    """All 1200 paper training videos, represented by their six 3-FPS frames."""

    def __init__(self, tokenizer) -> None:
        rows = read_manifest()[: SETTINGS.train_samples]
        self.frames = np.load(VIDEO_FRAMES_PATH, mmap_mode="r")
        self.prompt_ids = None
        if tokenizer is not None:
            self.prompt_ids = tokenizer(
                [row["caption"] for row in rows],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

    def __len__(self) -> int:
        return SETTINGS.train_samples

    def __getitem__(self, index: int) -> dict:
        import torch

        frames = torch.from_numpy(np.asarray(self.frames[index], dtype=np.float32))
        result = {
            "pixel_values": frames.permute(0, 3, 1, 2) / 127.5 - 1.0,
        }
        if self.prompt_ids is not None:
            result["prompt_ids"] = self.prompt_ids[index]
        return result


def dynamic_labels() -> np.ndarray:
    return np.asarray([row["dynamic_label"] for row in read_manifest()], dtype=np.int64)


def iter_batches(values: Iterable, size: int):
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
