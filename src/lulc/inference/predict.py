"""Sliding-window inference over a full scene."""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

from lulc.data.dataset import PatchDataset
from lulc.data.raster import RasterMeta, normalize, patch_coords, read_raster
from lulc.utils import get_device


@torch.no_grad()
def sliding_window_predict(
    model,
    data: np.ndarray,
    patch_size: int,
    stride: int,
    num_classes: int,
    batch_size: int = 8,
    median_size: int = 5,
    num_workers: int = 0,
    device: torch.device | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Predict a class map for a normalized ``(bands, H, W)`` array.

    Logits - not argmax labels - are accumulated and averaged across
    overlapping windows. Averaging labels would produce visible seams at tile
    boundaries; averaging logits blends the decision smoothly. A median filter
    then clears the residual salt-and-pepper speckle.
    """
    device = device or get_device()
    _, height, width = data.shape
    coords = patch_coords(height, width, patch_size, stride)

    if verbose:
        print(f"  {height}x{width} px | {len(coords)} patches | batch {batch_size}")

    loader = DataLoader(
        PatchDataset(data, coords, patch_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logit_sum = np.zeros((num_classes, height, width), dtype="float32")
    count = np.zeros((height, width), dtype="float32")
    t0 = time.time()

    model.eval()
    use_amp = torch.cuda.is_available()
    for patches, origins in tqdm(loader, desc="  inference", ncols=88, disable=not verbose):
        patches = patches.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(patches)
        logits = logits.float().cpu().numpy()

        for i, (y, x) in enumerate(origins.numpy()):
            y, x = int(y), int(x)
            logit_sum[:, y : y + patch_size, x : x + patch_size] += logits[i]
            count[y : y + patch_size, x : x + patch_size] += 1

    elapsed = time.time() - t0
    if verbose and elapsed > 0:
        print(f"  Elapsed: {elapsed:.1f}s ({len(coords) / elapsed:.0f} patches/s)")

    pred = (logit_sum / np.maximum(count, 1)[np.newaxis]).argmax(0).astype("uint8")
    if median_size and median_size > 1:
        pred = median_filter(pred, size=median_size)
    return pred


def predict_raster(model, tif_path, cfg, device=None, verbose=True) -> tuple[np.ndarray, np.ndarray, RasterMeta]:
    """Read a GeoTIFF, normalize with its own statistics, and predict.

    Normalization stats come from the target scene, never from the training
    AOI - that is the whole reason a checkpoint transfers to a new city.

    Returns ``(pred_map, normalized_data, raster_meta)``.
    """
    data, info = read_raster(tif_path)
    if verbose:
        print(f"  Raw shape: {data.shape} (bands x H x W)")

    data, means, stds = normalize(data)
    if verbose:
        for b, (m, s) in enumerate(zip(means, stds)):
            print(f"    band {b + 1}: mean={m:8.2f} std={s:8.2f}")

    pred = sliding_window_predict(
        model,
        data,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        num_classes=cfg.num_classes,
        batch_size=cfg.batch_size,
        median_size=cfg.median_size,
        num_workers=cfg.num_workers,
        device=device,
        verbose=verbose,
    )
    return pred, data, info
