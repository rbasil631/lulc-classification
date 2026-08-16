"""Configuration loading.

Replaces the hardcoded ``CONFIG`` dicts (and absolute Windows paths) that the
original scripts carried at module level. Every path is resolved relative to
the repository root unless it is already absolute, so the same config file
works on Windows, Linux and Colab.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

_PATH_KEYS = {"image_tif", "mask_dir", "output_dir", "model_path", "input_tif"}


@dataclass
class TrainConfig:
    """Hyperparameters and paths for the training pipeline."""

    image_tif: Path = Path("data/raw/city_s2_6band.tif")
    mask_dir: Path | None = Path("data/patches/masks")
    output_dir: Path = Path("outputs/train")

    patch_size: int = 256
    stride: int = 128           # training grid overlap; val uses a disjoint grid
    batch_size: int = 2
    epochs: int = 20
    lr: float = 1e-3
    early_stop_patience: int = 5
    val_fraction: float = 0.1

    num_bands: int = 6
    num_classes: int = 6

    encoder_weights: str | None = "imagenet"   # None = random init (offline runs)
    use_torchgeo: bool = True                  # SSL4EO Sentinel-2 weights
    freeze_encoder: bool = True
    encoder_warmup_batches: int = 1   # gentle unfreeze after the decoder converges
    label_smoothing: float = 0.1
    grad_clip: float = 1.0

    seed: int = 42
    num_workers: int = 0

    xai_samples: int = 5
    xai_methods: list[str] = field(
        default_factory=lambda: ["gradient", "occlusion", "shap"]
    )
    rgb_indices: list[int] = field(default_factory=lambda: [2, 1, 0])


@dataclass
class InferenceConfig:
    """Paths and knobs for running a trained checkpoint on a new AOI."""

    model_path: Path = Path("outputs/train/best_model.pth")
    input_tif: Path = Path("data/raw/new_aoi_6band.tif")
    output_dir: Path = Path("outputs/inference")

    num_bands: int = 6
    num_classes: int = 6
    patch_size: int = 256
    stride: int = 128
    batch_size: int = 8
    median_size: int = 5        # post-processing filter kernel
    num_workers: int = 0

    rgb_indices: list[int] = field(default_factory=lambda: [2, 1, 0])
    save_debug_rgb: bool = True
    output_dpi: int = 200


def resolve(path: str | os.PathLike | None) -> Path | None:
    """Resolve a config path against the repository root."""
    if path is None or path == "":
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _coerce(cls, raw: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) in config for {cls.__name__}: {sorted(unknown)}"
        )
    for key in _PATH_KEYS & set(raw):
        raw[key] = resolve(raw[key])
    return cls(**raw)


def load_train_config(path: str | os.PathLike) -> TrainConfig:
    """Load a training config from YAML, resolving all paths."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = _coerce(TrainConfig, raw)
    cfg.image_tif = resolve(cfg.image_tif)
    cfg.mask_dir = resolve(cfg.mask_dir)
    cfg.output_dir = resolve(cfg.output_dir)
    return cfg


def load_inference_config(path: str | os.PathLike) -> InferenceConfig:
    """Load an inference config from YAML, resolving all paths."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = _coerce(InferenceConfig, raw)
    cfg.model_path = resolve(cfg.model_path)
    cfg.input_tif = resolve(cfg.input_tif)
    cfg.output_dir = resolve(cfg.output_dir)
    return cfg
