"""GeoTIFF reading, per-AOI normalization, tiling and reference-mask loading."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


@dataclass
class RasterMeta:
    """Georeferencing carried through the pipeline so outputs stay aligned."""

    meta: dict
    transform: object
    crs: object
    height: int
    width: int


def remap_labels(mask: np.ndarray) -> np.ndarray:
    """Merge the road class (2) into building (1) and shift higher labels down.

    The annotation schema had 7 classes; road pixels are too sparse at 10 m
    resolution to learn separately, so they are folded into ``building``.
    """
    mask = mask.copy()
    mask[mask == 2] = 1
    mask[mask > 2] -= 1
    return mask


def compute_norm_stats(data: np.ndarray) -> tuple[list[float], list[float]]:
    """Per-band mean/std over valid (non-zero) pixels.

    Computing these from the image itself rather than from baked-in constants
    is what lets a checkpoint transfer to a new AOI: different scenes have
    different atmospheric conditions and reflectance ranges.

    Parameters
    ----------
    data : np.ndarray
        Array of shape ``(bands, H, W)``.
    """
    means: list[float] = []
    stds: list[float] = []
    for band in data:
        valid = band[band > 0]
        if valid.size == 0:
            means.append(0.0)
            stds.append(1.0)
        else:
            means.append(float(valid.mean()))
            stds.append(float(valid.std()) + 1e-6)
    return means, stds


def normalize(
    data: np.ndarray,
    means: list[float] | None = None,
    stds: list[float] | None = None,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Band-wise z-score normalization.

    If ``means``/``stds`` are omitted they are computed from ``data`` itself
    and returned so the caller can log or reuse them.
    """
    if means is None or stds is None:
        means, stds = compute_norm_stats(data)

    out = data.copy()
    for b in range(data.shape[0]):
        out[b] = (data[b] - means[b]) / stds[b]
    return out, means, stds


def read_raster(path: str | os.PathLike) -> tuple[np.ndarray, RasterMeta]:
    """Read a multi-band GeoTIFF as float32 plus its georeferencing."""
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
        info = RasterMeta(
            meta=src.meta.copy(),
            transform=src.transform,
            crs=src.crs,
            height=src.height,
            width=src.width,
        )
    return data, info


def patch_coords(height: int, width: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
    """Top-left ``(y, x)`` corners of every full patch on a strided grid."""
    return [
        (y, x)
        for y in range(0, height - patch_size + 1, stride)
        for x in range(0, width - patch_size + 1, stride)
    ]


def tile_geotiff(
    tif_path: str | os.PathLike,
    patch_size: int,
    stride: int,
    verbose: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]], RasterMeta, list[float], list[float]]:
    """Read, normalize and tile the AOI GeoTIFF.

    Returns the normalized array, patch coordinates, georeferencing, and the
    normalization statistics that were used.
    """
    data, info = read_raster(tif_path)
    data, means, stds = normalize(data)
    coords = patch_coords(info.height, info.width, patch_size, stride)

    if verbose:
        print(f"  Image  : {info.height} x {info.width} ({data.shape[0]} bands)")
        print(f"  Patches: {len(coords)} @ {patch_size}px stride {stride}")
        print(f"  Norm   : band0 mean={means[0]:.2f} std={stds[0]:.2f}")

    return data, coords, info, means, stds


def load_masks(
    mask_dir: str | os.PathLike | None,
    coords: list[tuple[int, int]],
    verbose: bool = True,
) -> np.ndarray | None:
    """Load reference mask GeoTIFFs, remapped to the training schema.

    Masks are matched to ``coords`` by sort order, so filenames must sort in
    the same order the coordinate grid is generated (row-major). Returns
    ``None`` when the directory is missing or the count does not match, which
    makes fine-tuning and evaluation gracefully skip.
    """
    if not mask_dir or not os.path.isdir(mask_dir):
        if verbose:
            print("  No mask directory - fine-tuning and evaluation will be skipped.")
        return None

    mask_files = sorted(glob.glob(os.path.join(str(mask_dir), "*.tif")))
    if len(mask_files) != len(coords):
        if verbose:
            print(
                f"  WARNING: {len(mask_files)} mask files vs {len(coords)} patches "
                "- skipping fine-tuning."
            )
        return None

    masks = np.stack(
        [remap_labels(rasterio.open(f).read(1).astype("int64")) for f in mask_files]
    )
    if verbose:
        print(f"  Loaded {len(masks)} reference masks from {Path(mask_dir).name}/")
    return masks
