"""Dataset indexing, splitting, caching and augmentation.

The raw dataset under ``data/natural-images`` is treated as read-only: nothing here writes
to it. Resized arrays are cached separately under ``artifacts/cache``.

The Kaggle archive ships every image twice (``natural_images/`` and
``data/natural_images/``); only ``natural_images/`` is read, so no image is counted twice.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageIndex:
    """File paths (relative to the dataset root), integer labels and class names."""

    paths: list[str]
    labels: np.ndarray
    classes: list[str]
    duplicates_removed: int


@dataclass
class Split:
    x: np.ndarray  # uint8, [N, H, W, 3]
    y: np.ndarray  # int64, [N]


def index_dataset(root: Path, deduplicate: bool = True) -> ImageIndex:
    """List every image under ``root/<class>/``.

    With ``deduplicate`` on, byte-identical files are kept once (the first in sorted
    order). This excludes byte-identical copies across splits; related views and
    visually similar photographs require separate review and grouping.
    """
    root = Path(root)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class folders found under {root}")

    paths: list[str] = []
    labels: list[int] = []
    seen: set[str] = set()
    removed = 0
    for label, name in enumerate(classes):
        for file in sorted((root / name).iterdir()):
            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if deduplicate:
                digest = hashlib.sha1(file.read_bytes()).hexdigest()
                if digest in seen:
                    removed += 1
                    continue
                seen.add(digest)
            paths.append(file.relative_to(root).as_posix())
            labels.append(label)
    return ImageIndex(paths, np.asarray(labels, dtype=np.int64), classes, removed)


def stratified_split(
    labels: np.ndarray, val_size: float, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test index arrays with class proportions preserved in each."""
    all_idx = np.arange(len(labels))
    train_val, test = train_test_split(
        all_idx, test_size=test_size, stratify=labels, random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=labels[train_val], random_state=seed
    )
    return np.sort(train), np.sort(val), np.sort(test)


def load_image(path: Path, size: int) -> np.ndarray:
    """Read one image as RGB uint8 resized to ``size`` x ``size`` (bicubic)."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def load_or_build_cache(root: Path, index: ImageIndex, size: int, cache_dir: Path) -> np.ndarray:
    """Load resized images from cache, rebuilding it if missing or stale."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"images_{size}px.npz"
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as cached:
            if list(cached["paths"]) == index.paths:
                return cached["images"]
    images = np.stack([load_image(Path(root) / p, size) for p in index.paths])
    np.savez(cache_file, images=images, paths=np.asarray(index.paths))
    return images


def channel_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std in [0, 1] units, computed on the training split only."""
    total = np.zeros(3)
    total_sq = np.zeros(3)
    for i in range(0, len(x), 256):  # chunked: never materialise a float copy of the split
        chunk = x[i : i + 256].reshape(-1, 3).astype(np.float64) / 255.0
        total += chunk.sum(axis=0)
        total_sq += np.square(chunk).sum(axis=0)
    n = x.shape[0] * x.shape[1] * x.shape[2]
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean**2, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


class Normalizer:
    """uint8 NHWC numpy batch -> normalized float32 NCHW tensor."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.as_tensor(mean).view(1, 3, 1, 1)
        self.std = torch.as_tensor(std).view(1, 3, 1, 1)

    def to_unit(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255.0)

    def __call__(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self.mean) / self.std


class Augmenter:
    """Batch augmentation on [0, 1] NCHW tensors.

    Random horizontal flip, random translation (reflect-pad then crop), and per-image
    brightness/contrast jitter. No vertical flips: people, cars and airplanes have a
    natural up direction that the row-wise LSTM reads.
    """

    def __init__(self, pad: int = 6, jitter: float = 0.2, seed: int = 0):
        self.pad = pad
        self.jitter = jitter
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        g = self.gen
        flip = torch.rand(b, generator=g) < 0.5
        x = torch.where(flip.view(b, 1, 1, 1), x.flip(-1), x)

        if self.pad:
            padded = torch.nn.functional.pad(x, [self.pad] * 4, mode="reflect")
            offs = torch.randint(0, 2 * self.pad + 1, (b, 2), generator=g)
            x = torch.stack(
                [padded[i, :, oy : oy + h, ox : ox + w] for i, (oy, ox) in enumerate(offs.tolist())]
            )

        if self.jitter:
            bright = 1 + (torch.rand(b, 1, 1, 1, generator=g) * 2 - 1) * self.jitter
            contrast = 1 + (torch.rand(b, 1, 1, 1, generator=g) * 2 - 1) * self.jitter
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            x = ((x - mean) * contrast + mean) * bright
            x = x.clamp_(0.0, 1.0)
        return x
