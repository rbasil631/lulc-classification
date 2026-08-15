"""Per-class spectral band importance.

Three complementary answers to "which Sentinel-2 bands drive each class?":

===========  ======  ==========================================================
Method       Speed   Character
===========  ======  ==========================================================
gradient     fast    Good relative ranking; sensitive to gradient saturation.
occlusion    slow    Directly interpretable ("logit drops X when band is
                     removed"), and needs no gradients at all.
SHAP         medium  Shapley-axiom grounded; requires the ``shap`` package.
===========  ======  ==========================================================

All three L1-normalize per class, so each class's bars sum to 1 and classes are
comparable to one another rather than to their own logit magnitudes.

Sanity check on any run: NIR and Green should dominate for forest and
agriculture. If they do not, suspect the band ordering in the input GeoTIFF
before you suspect the model.

.. note::
   This module reconstructs ``corrected_xai.py``, which was referenced by the
   original ``final.py`` but never committed. Behaviour is reproduced from the
   surviving bytecode; numbers should be regenerated rather than compared
   against the old figures.
"""

from __future__ import annotations

import numpy as np
import torch

from lulc.constants import BAND_COLORS, BAND_NAMES
from lulc.utils import free_cuda
from lulc.xai.plots import plot_aggregate_importance, plot_per_class_importance


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """L1-normalize each row so per-class importances sum to 1."""
    denom = np.abs(arr).sum(axis=1, keepdims=True)
    return arr / np.maximum(denom, 1e-8)


def gradient_band_importance(
    model, patch: np.ndarray, cfg, num_classes: int, class_names, device, verbose: bool = True
) -> np.ndarray:
    """Gradient of each class's mean logit w.r.t. the input bands.

    Two things matter here and both were wrong in the first version of this
    analysis: the class score is the **mean** logit over the patch (``amax``
    made the result depend on a single lucky pixel), and the gradient is
    **not** multiplied by input magnitude (that just re-ranks bands by
    reflectance scale, so SWIR always won).

    Returns
    -------
    np.ndarray
        Shape ``(num_classes, n_bands)``, L1-normalized per class.
    """
    if verbose:
        print("  Gradient band importance ...")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_bands = patch.shape[0]
    raw = np.zeros((num_classes, n_bands), dtype="float32")

    for cls in range(num_classes):
        t = torch.from_numpy(patch.copy()).unsqueeze(0).float().to(device)
        t.requires_grad_(True)
        out = model(t)
        score = out[0, cls].mean()
        score.backward()
        raw[cls] = t.grad.detach().abs().cpu().numpy()[0].sum(axis=(1, 2))
        del t, out, score
        free_cuda()

    for p in model.parameters():
        p.requires_grad_(True)

    norm = _normalize_rows(raw)
    plot_per_class_importance(
        norm,
        class_names,
        BAND_NAMES[:n_bands],
        BAND_COLORS[:n_bands],
        title="Per-Class Band Importance (Gradient, L1-Normalized)",
        ylabel="Normalized Gradient Importance",
        output_dir=cfg.output_dir,
        filename="band_importance_per_class_gradient.png",
    )
    plot_aggregate_importance(
        norm.sum(axis=0) / num_classes,
        BAND_NAMES[:n_bands],
        BAND_COLORS[:n_bands],
        title="Which Bands Matter Most? (Gradient, Aggregate)",
        ylabel="Normalized Total Gradient",
        output_dir=cfg.output_dir,
        filename="band_importance_aggregate_gradient.png",
    )
    return norm


@torch.no_grad()
def occlusion_band_importance(
    model, patch: np.ndarray, cfg, num_classes: int, class_names, device, verbose: bool = True
) -> np.ndarray:
    """Zero out each band in turn and measure the drop in each class logit.

    Slower than gradients (one forward pass per band) but the most honest of
    the three: it answers "how much worse does this class score get without
    this band?" with no linearity assumption.

    Returns
    -------
    np.ndarray
        Shape ``(num_classes, n_bands)``, L1-normalized per class.
    """
    if verbose:
        print("  Occlusion / ablation band importance ...")

    model.eval()
    n_bands = patch.shape[0]

    t_full = torch.from_numpy(patch.copy()).unsqueeze(0).float().to(device)
    baseline = model(t_full).mean(dim=(2, 3))[0].cpu().numpy()

    raw = np.zeros((num_classes, n_bands), dtype="float32")
    for b in range(n_bands):
        masked = patch.copy()
        masked[b] = 0.0
        t_masked = torch.from_numpy(masked).unsqueeze(0).float().to(device)
        logits = model(t_masked).mean(dim=(2, 3))[0].cpu().numpy()
        raw[:, b] = np.maximum(baseline - logits, 0.0)
        del t_masked
        free_cuda()

    del t_full
    free_cuda()

    norm = _normalize_rows(raw)
    plot_per_class_importance(
        norm,
        class_names,
        BAND_NAMES[:n_bands],
        BAND_COLORS[:n_bands],
        title="Per-Class Band Importance (Occlusion / Ablation)",
        ylabel="Normalized Logit Drop",
        output_dir=cfg.output_dir,
        filename="band_importance_per_class_occlusion.png",
    )
    return norm


def shap_band_importance(
    model, patch: np.ndarray, cfg, num_classes: int, class_names, device, verbose: bool = True
) -> np.ndarray | None:
    """SHAP ``GradientExplainer`` attribution, aggregated per band.

    Returns ``None`` if ``shap`` is not installed or the explainer fails - the
    caller should fall back to the gradient result rather than abort the run.
    """
    try:
        import shap
    except ImportError:
        if verbose:
            print("  shap not installed (pip install shap) - skipping SHAP method.")
        return None

    if verbose:
        print("  SHAP GradientExplainer (this takes a moment) ...")

    n_bands = patch.shape[0]
    try:
        tensor = torch.from_numpy(patch.copy()).unsqueeze(0).float().to(device)

        class _MeanLogit(torch.nn.Module):
            """Reduce (B, C, H, W) to (B, C) so SHAP sees one score per class."""

            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x).mean(dim=(2, 3))

        wrapped = _MeanLogit(model).to(device).eval()
        explainer = shap.GradientExplainer(wrapped, tensor)
        shap_values = explainer.shap_values(tensor, nsamples=32)

        if isinstance(shap_values, list):
            per_class = np.stack([np.abs(sv[0]).sum(axis=(1, 2)) for sv in shap_values])
        else:
            arr = np.asarray(shap_values)[0]          # (bands, H, W, classes)
            per_class = np.abs(arr).sum(axis=(1, 2)).T

        per_class = per_class[:num_classes, :n_bands]
        free_cuda()
    except Exception as exc:
        if verbose:
            print(f"  SHAP computation failed ({exc}) - skipping.")
        return None

    norm = _normalize_rows(per_class)
    plot_per_class_importance(
        norm,
        class_names,
        BAND_NAMES[:n_bands],
        BAND_COLORS[:n_bands],
        title="Per-Class Band Importance (SHAP GradientExplainer)",
        ylabel="Normalized SHAP Importance",
        output_dir=cfg.output_dir,
        filename="band_importance_per_class_shap.png",
    )
    return norm


def run_band_importance(
    model,
    data: np.ndarray,
    coords,
    cfg,
    num_classes: int,
    class_names,
    device,
    methods=("gradient", "occlusion", "shap"),
    verbose: bool = True,
) -> dict[str, np.ndarray | None]:
    """Run the selected attribution methods on one representative patch.

    Returns a dict keyed by method name; a value of ``None`` means that method
    was unavailable or failed.
    """
    y, x = coords[len(coords) // 2]
    p = cfg.patch_size
    patch = data[:, y : y + p, x : x + p].copy()

    if verbose:
        print(f"  Patch {patch.shape} at (y={y}, x={x}) | {num_classes} classes")

    runners = {
        "gradient": gradient_band_importance,
        "occlusion": occlusion_band_importance,
        "shap": shap_band_importance,
    }

    results: dict[str, np.ndarray | None] = {}
    for name in methods:
        runner = runners.get(name)
        if runner is None:
            print(f"  Unknown attribution method '{name}' - skipped.")
            continue
        try:
            results[name] = runner(
                model, patch, cfg, num_classes, class_names, device, verbose
            )
        except Exception as exc:
            print(f"  {name} attribution failed: {exc}")
            results[name] = None
    return results
