"""Rendering LULC predictions to GeoTIFF, colourized PNG and charts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from lulc.constants import CLASS_COLORS, CLASS_NAMES
from lulc.utils import ensure_dir


def class_percentages(pred_map: np.ndarray, num_classes: int) -> list[float]:
    """Percentage of pixels assigned to each class."""
    total = pred_map.size
    return [100.0 * np.sum(pred_map == i) / total for i in range(num_classes)]


def save_lulc_geotiff(pred_map: np.ndarray, info, output_dir, filename: str) -> str:
    """Write the class map as a single-band GeoTIFF, CRS and transform intact.

    The rasterio ``meta`` from the source scene is passed through rather than
    rebuilt, so the output overlays the input exactly in QGIS.
    """
    meta = {**info.meta, "count": 1, "dtype": "uint8", "nodata": 255}
    out = ensure_dir(output_dir) / filename
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(pred_map.astype("uint8"), 1)
        dst.update_tags(
            classes=str(CLASS_NAMES),
            colors=str(CLASS_COLORS.tolist()),
            description="LULC prediction - UNet++/ResNet-50 SSL4EO",
        )
    print(f"  GeoTIFF: {out}")
    return str(out)


def save_lulc_png(
    pred_map: np.ndarray,
    output_dir,
    filename: str,
    title: str = "LULC Map - Sentinel-2",
    dpi: int = 200,
) -> str:
    """Colourized class map with a legend and a coverage summary box."""
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(CLASS_COLORS[pred_map])
    ax.set_title(title, fontsize=16, pad=12)
    ax.axis("off")
    ax.legend(
        handles=[
            mpatches.Patch(color=CLASS_COLORS[i] / 255.0, label=CLASS_NAMES[i])
            for i in range(len(CLASS_NAMES))
        ],
        loc="lower right",
        fontsize=11,
        framealpha=0.9,
    )

    unique, counts = np.unique(pred_map, return_counts=True)
    total = pred_map.size
    summary = "\n".join(
        f"{CLASS_NAMES[int(u)]}: {100 * c / total:.1f}%"
        for u, c in zip(unique, counts)
        if int(u) < len(CLASS_NAMES)
    )
    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    out = ensure_dir(output_dir) / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Colour map: {out}")
    return str(out)


def save_class_distribution(
    pred_map: np.ndarray, output_dir, filename: str, num_classes: int = 6
) -> str:
    """Bar chart of per-class coverage."""
    pcts = class_percentages(pred_map, num_classes)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        CLASS_NAMES[:num_classes],
        pcts,
        color=[CLASS_COLORS[i] / 255.0 for i in range(num_classes)],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Coverage (%)", fontsize=12)
    ax.set_title("LULC Class Distribution", fontsize=13)
    ax.set_ylim(0, max(pcts) * 1.15 if pcts else 1)
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    out = ensure_dir(output_dir) / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Distribution: {out}")
    return str(out)


def save_rgb_preview(data: np.ndarray, rgb_indices, output_dir, filename: str) -> str:
    """Normalized true-colour preview of the input, for sanity-checking bands."""
    rgb = data[list(rgb_indices)].transpose(1, 2, 0).astype("float32")
    rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6), 0, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    ax.set_title("Input - normalized RGB (B4/B3/B2)", fontsize=13)
    ax.axis("off")

    out = ensure_dir(output_dir) / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  RGB preview: {out}")
    return str(out)


def print_summary(pred_map: np.ndarray, num_classes: int = 6) -> None:
    """Console table of pixel counts and coverage."""
    total = pred_map.size
    print(f"\n  {'Class':15s} {'Pixels':>12s} {'Coverage':>10s}")
    print("  " + "-" * 39)
    for i, name in enumerate(CLASS_NAMES[:num_classes]):
        px = int(np.sum(pred_map == i))
        print(f"  {name:15s} {px:>12,} {100 * px / total:>9.2f}%")
    print()
