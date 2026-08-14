"""Lazy patch datasets that slice from an in-memory normalized array."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class S2PatchDataset(Dataset):
    """Training/validation patches with optional masks and augmentation.

    Parameters
    ----------
    data : np.ndarray
        Normalized array of shape ``(bands, H, W)``.
    coords : list[tuple[int, int]]
        Top-left corners of each patch.
    patch_size : int
    masks : np.ndarray | None
        Array of shape ``(len(coords), patch_size, patch_size)``.
    transform : albumentations.Compose | None
    """

    def __init__(self, data, coords, patch_size, masks=None, transform=None):
        self.data = data
        self.coords = coords
        self.patch_size = patch_size
        self.masks = masks
        self.transform = transform

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        p = self.patch_size
        patch = self.data[:, y : y + p, x : x + p].transpose(1, 2, 0)  # HWC

        if self.masks is None:
            if self.transform:
                return self.transform(image=patch)["image"]
            return torch.from_numpy(patch.transpose(2, 0, 1).copy()).float()

        mask = self.masks[idx].astype("int64")
        if self.transform:
            aug = self.transform(image=patch, mask=mask)
            return aug["image"], aug["mask"]
        return (
            torch.from_numpy(patch.transpose(2, 0, 1).copy()).float(),
            torch.from_numpy(mask.copy()).long(),
        )


class PatchDataset(Dataset):
    """Inference-only patches; also returns each patch's origin for stitching."""

    def __init__(self, data: np.ndarray, coords, patch_size: int):
        self.data = data
        self.coords = coords
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        p = self.patch_size
        patch = self.data[:, y : y + p, x : x + p]
        return torch.from_numpy(patch.copy()).float(), torch.tensor([y, x])
