"""Augmentation pipeline.

Radiometric jitter (brightness/contrast, noise) is what makes the model
survive being pointed at a different city; flips and rotations alone let it
memorize the training AOI's exact appearance.

Albumentations renamed several arguments in 2.0. The helpers below try the
2.x signature first and fall back to 1.x so the same code runs on both.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _gauss_noise(p: float = 0.3):
    try:
        return A.GaussNoise(std_range=(0.03, 0.12), p=p)          # albumentations >= 2.0
    except TypeError:
        return A.GaussNoise(var_limit=(5.0, 20.0), p=p)           # albumentations < 2.0


def _coarse_dropout(p: float = 0.2):
    try:
        return A.CoarseDropout(                                   # albumentations >= 2.0
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=p,
        )
    except TypeError:
        return A.CoarseDropout(                                   # albumentations < 2.0
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=0,
            p=p,
        )


def get_transforms():
    """Return ``(train_transform, eval_transform)``."""
    train_tf = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.4
            ),
            _gauss_noise(p=0.3),
            _coarse_dropout(p=0.2),
            A.Sharpen(alpha=(0.1, 0.3), p=0.2),
            ToTensorV2(),
        ]
    )
    eval_tf = A.Compose([ToTensorV2()])
    return train_tf, eval_tf
