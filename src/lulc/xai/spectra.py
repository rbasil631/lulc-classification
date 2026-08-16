"""Measured class spectral signatures - the model-free ground truth.

Every attribution method in ``attribution.py`` explains *the model*. None of
them can tell you whether the model is reading the imagery correctly, because
a model that has learned the wrong evidence will be explained faithfully and
still be wrong.

This module measures what is actually in the pixels: mean reflectance per band
per class, and how separable the classes are on each band alone. That gives an
independent reference to check attribution against.

If the model's attribution ranks a band highly and this module shows that band
barely separates the class from everything else, the model is keying on
something spurious. If the two agree, the explanation is corroborated.

Everything here operates on the raw (unnormalized) raster, so the numbers are
physical reflectance in Sentinel-2 L2A scaling, comparable to published
spectral libraries.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lulc.constants import BAND_IDS, BAND_NAMES, CLASS_COLORS
from lulc.utils import ensure_dir


def class_spectra(
    raw: np.ndarray,
    label_map: np.ndarray,
    num_classes: int,
    max_pixels: int = 200_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, std and pixel count per class per band.

    Parameters
    ----------
    raw : np.ndarray
        Unnormalized ``(bands, H, W)`` reflectance.
    label_map : np.ndarray
        ``(H, W)`` class labels aligned to ``raw``.
    max_pixels : int
        Per-class sampling cap; agriculture alone runs to tens of millions of
        pixels and the mean is stable long before that.

    Returns
    -------
    (means, stds, counts) with shapes ``(C, B)``, ``(C, B)``, ``(C,)``.
    """
    rng = np.random.default_rng(seed)
    n_bands = raw.shape[0]
    flat = raw.reshape(n_bands, -1)
    labels = label_map.ravel()

    means = np.zeros((num_classes, n_bands))
    stds = np.zeros((num_classes, n_bands))
    counts = np.zeros(num_classes, dtype="int64")

    for cls in range(num_classes):
        idx = np.flatnonzero(labels == cls)
        counts[cls] = idx.size
        if idx.size == 0:
            continue
        if idx.size > max_pixels:
            idx = rng.choice(idx, max_pixels, replace=False)
        sample = flat[:, idx]
        means[cls] = sample.mean(axis=1)
        stds[cls] = sample.std(axis=1)
    return means, stds, counts


def band_separability(means: np.ndarray, stds: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """How well each band separates each class from all other classes.

    Uses a standardized mean difference (Cohen's d) between the class and the
    count-weighted pool of the others, on each band independently. Higher means
    that band alone carries more discriminative signal for that class.

    This is the quantity attribution should broadly track. It is not a perfect
    target - a CNN combines bands and exploits texture, so it can legitimately
    rely on a band with low univariate separability - but a large disagreement
    is worth investigating.
    """
    num_classes, n_bands = means.shape
    sep = np.zeros((num_classes, n_bands))

    for cls in range(num_classes):
        others = [c for c in range(num_classes) if c != cls and counts[c] > 0]
        if counts[cls] == 0 or not others:
            continue
        w = counts[others].astype("float64")
        w /= w.sum()
        other_mean = (means[others] * w[:, None]).sum(axis=0)
        other_var = (stds[others] ** 2 * w[:, None]).sum(axis=0)
        pooled = np.sqrt((stds[cls] ** 2 + other_var) / 2) + 1e-9
        sep[cls] = np.abs(means[cls] - other_mean) / pooled
    return sep


def plot_class_spectra(
    means: np.ndarray, stds: np.ndarray, counts: np.ndarray,
    class_names, output_dir, filename: str = "class_spectral_signatures.png",
) -> str:
    """Spectral response curves, one line per class, with +/- 1 sd bands."""
    n_bands = means.shape[1]
    x = np.arange(n_bands)

    fig, ax = plt.subplots(figsize=(10, 6))
    for cls, name in enumerate(class_names):
        if counts[cls] == 0:
            continue
        colour = CLASS_COLORS[cls] / 255.0
        ax.plot(x, means[cls], "-o", color=colour, label=name, linewidth=2, markersize=5)
        ax.fill_between(
            x, means[cls] - stds[cls], means[cls] + stds[cls], color=colour, alpha=0.12
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{b}\n{i}" for b, i in zip(BAND_NAMES[:n_bands], BAND_IDS[:n_bands])]
    )
    ax.set_ylabel("Surface reflectance (L2A scaling)", fontsize=11)
    ax.set_title(
        "Measured Class Spectral Signatures (from imagery, no model)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()

    out = ensure_dir(output_dir) / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return str(out)


def plot_separability(
    sep: np.ndarray, class_names, output_dir,
    filename: str = "band_separability.png",
) -> str:
    """Heatmap of univariate class-vs-rest separability per band."""
    n_bands = sep.shape[1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(sep, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n_bands))
    ax.set_xticklabels(BAND_NAMES[:n_bands], rotation=20, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_title(
        "Univariate Band Separability (Cohen's d, class vs rest)",
        fontsize=12, fontweight="bold",
    )
    for i in range(sep.shape[0]):
        for j in range(n_bands):
            ax.text(
                j, i, f"{sep[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if sep[i, j] < sep.max() * 0.6 else "black",
            )
    fig.colorbar(im, ax=ax, label="|standardized mean difference|")
    fig.tight_layout()

    out = ensure_dir(output_dir) / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return str(out)


def run_spectral_analysis(
    raw: np.ndarray, label_map: np.ndarray, num_classes: int,
    class_names, output_dir, verbose: bool = True,
) -> dict:
    """Measure and plot class spectra plus per-band separability."""
    if verbose:
        print("  Measuring class spectral signatures from raw imagery ...")

    means, stds, counts = class_spectra(raw, label_map, num_classes)
    sep = band_separability(means, stds, counts)

    plot_class_spectra(means, stds, counts, class_names, output_dir)
    plot_separability(sep, class_names, output_dir)

    if verbose:
        print("\n  Most discriminative band per class (from imagery alone):")
        for cls, name in enumerate(class_names):
            if counts[cls] == 0:
                print(f"    {name:14s} - absent")
                continue
            best = int(np.argmax(sep[cls]))
            print(
                f"    {name:14s} {BAND_NAMES[best]:6s} "
                f"(d={sep[cls, best]:.2f}, n={counts[cls]:,})"
            )

    return {"means": means, "stds": stds, "counts": counts, "separability": sep}
