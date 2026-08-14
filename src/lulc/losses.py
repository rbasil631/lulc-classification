"""Segmentation loss."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.1,
    ce_weight: float = 0.5,
) -> torch.Tensor:
    """Equal-weight cross-entropy + Dice.

    Cross-entropy alone under-serves the rare classes (water, barren); Dice
    alone is unstable early in training. Label smoothing damps overconfidence
    on the single training AOI - pass ``0.0`` when computing validation loss so
    the number stays comparable across runs.
    """
    target = target.long()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(pred, target)
    dice = smp.losses.DiceLoss(mode="multiclass")(pred, target)
    return ce_weight * ce + (1.0 - ce_weight) * dice
