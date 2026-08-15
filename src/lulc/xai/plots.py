"""Shared plotting helpers for the XAI outputs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lulc.utils import ensure_dir


def plot_per_class_importance(
    importance: np.ndarray,
    class_names,
    band_names,
    band_colors,
    title: str,
    ylabel: str,
    output_dir,
    filename: str,
) -> str:
    """Grouped bar chart: one cluster per class, one bar per band."""
    n_classes, n_bands = importance.shape
    x = np.arange(n_classes)
    width = 0.8 / n_bands

    fig, ax = plt.subplots(figsize=(13, 5))
    for b in range(n_bands):
        ax.bar(
            x + b * width,
            importance[:, b],
            width=width,
            label=band_names[b],
            color=band_colors[b],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xticks(x + width * (n_bands - 1) / 2)
    ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(title="Band", fontsize=9, ncol=3, loc="upper right")
    ax.grid(axis="y", color="#333333", alpha=0.2, linestyle="--")
    fig.tight_layout()

    out = ensure_dir(output_dir) / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return str(out)


def plot_aggregate_importance(
    totals: np.ndarray,
    band_names,
    band_colors,
    title: str,
    ylabel: str,
    output_dir,
    filename: str,
) -> str:
    """Single bar chart ranking bands averaged over all classes."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        band_names, totals, color=band_colors, edgecolor="black", linewidth=0.5
    )
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, float(totals.max()) * 1.15 if totals.size else 1)
    for bar, val in zip(bars, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()

    out = ensure_dir(output_dir) / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return str(out)


def to_rgb(patch: np.ndarray, rgb_indices) -> np.ndarray:
    """Min-max stretch three bands of a normalized patch into a 0-1 RGB image."""
    rgb = patch[list(rgb_indices)].transpose(1, 2, 0).astype("float32")
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    return np.clip(rgb, 0.0, 1.0)
