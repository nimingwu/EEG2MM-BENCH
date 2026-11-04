"""Repository path and data-provenance guards for benchmark runs."""
from __future__ import annotations

from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]   # v3 仓库根
THINGSEEG_LINK = Path(__file__).resolve().parents[1] / "ThingsEEG"
EXPECTED_THINGSEEG_TARGET = Path("/datas/eeg_benchmark_project/ThingsEEG")
EXPECTED_NOD_ROOT = Path("/datas/eeg_benchmark_project/NOD-EEG")
EXPECTED_GROOTSWAGERS_ROOT = Path(
    "/datas/eeg_benchmark_project/openneurodownload/ds004018-download")
EXPECTED_GROOTSWAGERS2025_ROOT = Path(
    "/datas/eeg_benchmark_project/openneurodownload/ds005106-download")
RAW_EEG_BASENAMES = frozenset({"raw_eeg_training.npy", "raw_eeg_test.npy"})


def require_thingseeg_link() -> Path:
    """Return the in-repository ThingsEEG link without resolving it lexically."""
    if not THINGSEEG_LINK.is_symlink():
        raise RuntimeError(
            f"Required dataset link is missing or not a symlink: {THINGSEEG_LINK}")
    target = THINGSEEG_LINK.resolve(strict=True)
    if target != EXPECTED_THINGSEEG_TARGET.resolve(strict=True):
        raise RuntimeError(
            f"ThingsEEG link targets {target}, expected {EXPECTED_THINGSEEG_TARGET}")
    for directory in ("Raw_EEG_data", "Image_set"):
        if not (THINGSEEG_LINK / directory).is_dir():
            raise RuntimeError(f"ThingsEEG link is missing {directory}/")
    return THINGSEEG_LINK.absolute()


#: 允许的 benchmark 根目录。迁移阶段 v3 **读** v2 的 h5/artifacts，所以图像与
#: 视频两个 v2 仓库都认。**写**的路径另由 ``paths.require_v3_output`` 强制限定
#: 在 v3（见各阶段的 output）。
from .paths import V2_TEXT_ROOT, V2_VIDEO_ROOT  # noqa: E402  (paths 不反向依赖 rawdata)

OUTPUT_ROOTS = (BENCHMARK_ROOT,
                BENCHMARK_ROOT.parent / "EEG2X-Benchmark-v2-test",
                V2_VIDEO_ROOT,
                V2_TEXT_ROOT)


def require_benchmark_output(path: str | Path) -> Path:
    """Reject model artifacts that would be written outside a benchmark repository.

    与 v2 的差异：v2 只认自己的根目录；v3 迁移阶段还要**读** v2 的产物，因此
    放行 ``OUTPUT_ROOTS`` 里的任一根本。这条守卫防的是「产物落到仓库外」，
    不是「必须在这个仓库里」——写侧另有限制。
    """
    candidate = Path(path).absolute()
    resolved = candidate.resolve(strict=False)
    for root in OUTPUT_ROOTS:
        root = root.resolve(strict=False)
        if resolved.is_relative_to(root) and resolved != root:
            return candidate
    raise ValueError(
        f"Benchmark outputs must be below one of "
        f"{[str(r) for r in OUTPUT_ROOTS]}, got {candidate}")


def require_raw_thingseeg_eeg(path: str | Path, root: str | Path) -> Path:
    """Validate that an EEG input is a native raw file reached through the link."""
    candidate = Path(path).absolute()
    lexical_root = Path(root).absolute()
    if not candidate.is_relative_to(lexical_root):
        raise ValueError(
            f"ThingsEEG input must be reached through {lexical_root}, got {candidate}")
    if candidate.name not in RAW_EEG_BASENAMES:
        raise ValueError(f"Derived EEG input is forbidden: {candidate}")
    raw_root = (lexical_root / "Raw_EEG_data").resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(raw_root):
        raise ValueError(f"EEG input is outside Raw_EEG_data: {candidate}")
    return candidate


def require_raw_thingseeg_image(path: str | Path, root: str | Path) -> Path:
    """Validate that a stimulus is an original image, never a latent/saliency file."""
    candidate = Path(path).absolute()
    lexical_root = Path(root).absolute()
    # 迁移阶段读 v2 的 HDF5：其中的 stimulus_uri 记录的是 v2 仓库里的 ThingsEEG
    # 链接。两个链接字面不同、解析到同一数据集，因此接受「字面在根下」或「解析后
    # 在根下」两种写法；后面判定原始图像目录时一律用解析后的路径。
    resolved_root = lexical_root.resolve(strict=True)
    if candidate.is_relative_to(lexical_root):
        relative = candidate.relative_to(lexical_root)
    elif candidate.resolve(strict=False).is_relative_to(resolved_root):
        relative = candidate.resolve(strict=True).relative_to(resolved_root)
    else:
        raise ValueError(
            f"ThingsEEG stimulus must be reached through {lexical_root}, got {candidate}")
    forbidden = {"training_images_latents", "test_images_saliency_maps"}
    if forbidden.intersection(relative.parts):
        raise ValueError(f"Derived stimulus input is forbidden: {candidate}")
    image_roots = [
        lexical_root / "Image_set" / "training_images",
        lexical_root / "Image_set" / "test_images",
    ]
    allowed_roots = [item.resolve(strict=True) for item in image_roots if item.is_dir()]
    resolved = candidate.resolve(strict=True)
    if not any(resolved.is_relative_to(allowed) for allowed in allowed_roots):
        raise ValueError(f"Stimulus is outside original image directories: {candidate}")
    return candidate


#: Stimulus directory inside the Grootswagers release (216 PNGs: 200 stimuli +
#: 16 target images).
IMAGE_ROOT_NAME = "stimuli"


def require_grootswagers_root(root: str | Path = EXPECTED_GROOTSWAGERS_ROOT) -> Path:
    """Return the Grootswagers2019 (ds004018) root after validating its layout."""
    lexical = Path(root).absolute()
    if lexical != EXPECTED_GROOTSWAGERS_ROOT.absolute():
        raise ValueError(f"Unexpected Grootswagers root: {lexical}")
    if not lexical.is_dir():
        raise FileNotFoundError(f"Missing Grootswagers dataset root: {lexical}")
    for directory in (IMAGE_ROOT_NAME,):
        if not (lexical / directory).is_dir():
            raise RuntimeError(f"Grootswagers root is missing {directory}/")
    if not (lexical / "dataset_description.json").is_file():
        raise RuntimeError(f"Grootswagers root has no dataset_description.json: {lexical}")
    return lexical


#: Stimulus PNGs of the infant release (200 images, same files as ds004018).
INFANT_STIMULUS_DIR = ("code", "experiment", "stimuli")


def require_grootswagers2025_root(
        root: str | Path = EXPECTED_GROOTSWAGERS2025_ROOT) -> Path:
    """Return the ds005106 (infant) root after validating its layout."""
    lexical = Path(root).absolute()
    if lexical != EXPECTED_GROOTSWAGERS2025_ROOT.absolute():
        raise ValueError(f"Unexpected Grootswagers2025 root: {lexical}")
    if not lexical.is_dir():
        raise FileNotFoundError(f"Missing ds005106 root: {lexical}")
    stimulus_dir = lexical.joinpath(*INFANT_STIMULUS_DIR)
    if not stimulus_dir.is_dir():
        raise RuntimeError(f"ds005106 root is missing {stimulus_dir}")
    if not (lexical / "dataset_description.json").is_file():
        raise RuntimeError(f"ds005106 root has no dataset_description.json: {lexical}")
    return lexical


def require_raw_grootswagers2025_image(
        path: str | Path, root: str | Path = EXPECTED_GROOTSWAGERS2025_ROOT) -> Path:
    """Validate an original ds005106 stimulus PNG (never a derived asset)."""
    candidate = Path(path).absolute()
    lexical_root = Path(root).absolute()
    if lexical_root != EXPECTED_GROOTSWAGERS2025_ROOT.absolute():
        raise ValueError(f"Unexpected Grootswagers2025 root: {lexical_root}")
    if not candidate.is_relative_to(lexical_root):
        raise ValueError(
            f"ds005106 stimulus must be below {lexical_root}, got {candidate}")
    allowed = lexical_root.joinpath(*INFANT_STIMULUS_DIR).resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(allowed) or candidate.suffix.lower() != ".png":
        raise ValueError(f"ds005106 stimulus is outside {allowed}: {candidate}")
    return candidate


def require_raw_grootswagers_image(
        path: str | Path, root: str | Path = EXPECTED_GROOTSWAGERS_ROOT) -> Path:
    """Validate an original ds004018 stimulus PNG (never a derived asset)."""
    candidate = Path(path).absolute()
    lexical_root = Path(root).absolute()
    if lexical_root != EXPECTED_GROOTSWAGERS_ROOT.absolute():
        raise ValueError(f"Unexpected Grootswagers root: {lexical_root}")
    if not candidate.is_relative_to(lexical_root):
        raise ValueError(
            f"Grootswagers stimulus must be below {lexical_root}, got {candidate}")
    allowed = (lexical_root / IMAGE_ROOT_NAME).resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(allowed) or candidate.suffix.lower() != ".png":
        raise ValueError(f"Grootswagers stimulus is outside stimuli/: {candidate}")
    return candidate


def require_raw_nod_image(path: str | Path, root: str | Path = EXPECTED_NOD_ROOT) -> Path:
    """Validate an original NOD ImageNet stimulus (never a derived asset)."""
    candidate = Path(path).absolute()
    lexical_root = Path(root).absolute()
    if lexical_root != EXPECTED_NOD_ROOT.absolute():
        raise ValueError(f"Unexpected NOD root: {lexical_root}")
    if not candidate.is_relative_to(lexical_root):
        raise ValueError(f"NOD stimulus must be below {lexical_root}, got {candidate}")
    allowed = (lexical_root / "stimuli" / "ImageNet").resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(allowed) or candidate.suffix.upper() != ".JPEG":
        raise ValueError(f"NOD stimulus is outside stimuli/ImageNet: {candidate}")
    return candidate
