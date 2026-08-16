"""Per-class spectral band attribution.

Answers "which Sentinel-2 bands drive each class?" three ways.

Why the scoring is class-conditional
------------------------------------
The obvious approach - score a class by its mean logit over a whole patch, on
one arbitrary patch - fails badly here, and it is worth being explicit about
why, because the failure is silent and produces confident-looking plots.

1. **Whole-patch means erase class identity.** ~85% of any patch in this AOI is
   agriculture. If every class score is a mean over the same pixels, all six
   scores move together, and the surviving signal is the magnitude of the input
   convolution weights - identical for every class. The result is six bar
   groups of near-identical height.

2. **One patch does not contain every class.** Scoring forest on a patch with
   no forest measures noise. Rare classes (water 0.9%, barren 1.1%) are absent
   from most patches entirely.

3. **Clipping negative contributions destroys the row.** If removing a band
   *raises* a logit, that is real evidence about the model, not an error to
   floor at zero. Clipping leaves rows that are mostly zero, and L1
   normalisation then inflates whichever band survived to 1.0.

So every method below restricts the class score to pixels whose reference label
is that class, selects patches where the class is actually present, and keeps
the sign of the contribution.

Methods
-------
=========  ======  ============================================================
Method     Speed   Character
=========  ======  ============================================================
occlusion  slow    Zero a band, measure the change in the class score. No
                   linearity assumption; the most directly interpretable.
                   Blind to redundancy between correlated bands.
gradient   fast    Local sensitivity of the class score to each band.
                   Susceptible to gradient saturation.
integrated medium  Integrated Gradients. Satisfies completeness - attributions
                   sum to the score difference from the baseline - which the
                   code verifies numerically and reports.
=========  ======  ============================================================

Because inputs are z-scored per scene, zero **is** the band mean, so both the
occlusion and Integrated Gradients baselines are mean-imputation: "what if this
band carried no information beyond its average?"

Sanity check on any run: NIR should dominate forest and agriculture (the
vegetation red-edge response) and matter for water, which absorbs NIR almost
completely. If it does not, suspect band ordering in the input GeoTIFF before
suspecting the model. Compare against ``spectra.py``, which measures the same
thing directly from the imagery with no model involved.

.. note::
   Reconstructs ``corrected_xai.py``, referenced by the original ``final.py``
   but never committed.
"""

from __future__ import annotations

import numpy as np
import torch

from lulc.constants import BAND_COLORS, BAND_NAMES
from lulc.utils import free_cuda
from lulc.xai.plots import (
    plot_aggregate_importance,
    plot_importance_heatmap,
    plot_per_class_importance,
)

# A class must cover at least this fraction of a patch to be scored on it.
MIN_CLASS_FRACTION = 0.02


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Scale each row so its absolute values sum to 1, preserving sign."""
    denom = np.abs(arr).sum(axis=1, keepdims=True)
    return arr / np.maximum(denom, 1e-12)


def select_class_patches(
    masks: np.ndarray,
    num_classes: int,
    n_per_class: int = 6,
    min_fraction: float = MIN_CLASS_FRACTION,
    verbose: bool = True,
) -> dict[int, list[int]]:
    """For each class, the patch indices where it covers the most area.

    Ranking by coverage rather than sampling at random is what makes rare
    classes measurable: water is under 1% of pixels overall, but the handful of
    patches containing a reservoir carry plenty of it.

    Returns a dict mapping class index to patch indices, possibly empty for a
    class that never reaches ``min_fraction`` anywhere.
    """
    counts = np.stack(
        [np.bincount(m.ravel(), minlength=num_classes) for m in masks]
    ).astype("float64")
    fractions = counts / masks[0].size

    selection: dict[int, list[int]] = {}
    for cls in range(num_classes):
        order = np.argsort(-fractions[:, cls])
        chosen = [int(i) for i in order[:n_per_class] if fractions[i, cls] >= min_fraction]
        selection[cls] = chosen
        if verbose:
            cover = fractions[chosen, cls].mean() if chosen else 0.0
            note = "" if chosen else "  (never reaches threshold - skipped)"
            print(
                f"    class {cls}: {len(chosen)} patches, "
                f"mean coverage {100 * cover:5.1f}%{note}"
            )
    return selection


def _patch_tensor(data: np.ndarray, coord, patch_size: int, device) -> torch.Tensor:
    y, x = coord
    patch = data[:, y : y + patch_size, x : x + patch_size].copy()
    return torch.from_numpy(patch).unsqueeze(0).float().to(device)


def _region_mean(logits: torch.Tensor, region: torch.Tensor, cls: int):
    """Mean logit for ``cls`` over the pixels that truly belong to ``cls``.

    Returns ``None`` when the region is empty, so the caller can skip rather
    than average over nothing.
    """
    if not bool(region.any()):
        return None
    return logits[0, cls][region].mean()


def occlusion_band_importance(
    model, data, coords, masks, cfg, num_classes, class_names, device,
    selection=None, verbose=True,
) -> np.ndarray:
    """Zero each band and measure the signed change in each class score.

    Positive means removing the band *lowers* the class score, i.e. the band
    supports that class. Negative means removal raises it.
    """
    if verbose:
        print("  Occlusion / ablation (class-conditional, multi-patch) ...")

    model.eval()
    n_bands = cfg.num_bands
    raw = np.zeros((num_classes, n_bands), dtype="float64")

    with torch.no_grad():
        for cls, idxs in selection.items():
            acc, used = np.zeros(n_bands), 0
            for i in idxs:
                t = _patch_tensor(data, coords[i], cfg.patch_size, device)
                region = torch.from_numpy(masks[i].astype("int64")).to(device) == cls

                base = _region_mean(model(t), region, cls)
                if base is None:
                    del t
                    continue
                base = float(base)

                for b in range(n_bands):
                    occluded = t.clone()
                    occluded[0, b] = 0.0            # z-scored, so 0 == band mean
                    score = _region_mean(model(occluded), region, cls)
                    acc[b] += base - float(score)
                    del occluded
                used += 1
                del t
                free_cuda()

            if used:
                raw[cls] = acc / used
            elif verbose:
                print(f"    {class_names[cls]}: no usable patch - row left at zero")

    norm = _normalize_rows(raw)
    plot_per_class_importance(
        norm, class_names, BAND_NAMES[:n_bands], BAND_COLORS[:n_bands],
        title="Per-Class Band Importance (Occlusion, class-conditional)",
        ylabel="Normalized logit drop when band removed",
        output_dir=cfg.output_dir,
        filename="band_importance_per_class_occlusion.png",
    )
    return norm


def gradient_band_importance(
    model, data, coords, masks, cfg, num_classes, class_names, device,
    selection=None, verbose=True,
) -> np.ndarray:
    """Gradient magnitude of the class-region score w.r.t. each input band."""
    if verbose:
        print("  Gradient sensitivity (class-conditional, multi-patch) ...")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_bands = cfg.num_bands
    raw = np.zeros((num_classes, n_bands), dtype="float64")

    for cls, idxs in selection.items():
        acc, used = np.zeros(n_bands), 0
        for i in idxs:
            t = _patch_tensor(data, coords[i], cfg.patch_size, device)
            t.requires_grad_(True)
            region = torch.from_numpy(masks[i].astype("int64")).to(device) == cls

            score = _region_mean(model(t), region, cls)
            if score is None:
                del t
                continue
            grad = torch.autograd.grad(score, t)[0]
            acc += grad.abs()[0].sum(dim=(1, 2)).cpu().numpy()
            used += 1
            del t, grad, score
            free_cuda()

        if used:
            raw[cls] = acc / used

    for p in model.parameters():
        p.requires_grad_(True)

    norm = _normalize_rows(raw)
    plot_per_class_importance(
        norm, class_names, BAND_NAMES[:n_bands], BAND_COLORS[:n_bands],
        title="Per-Class Band Importance (Gradient, class-conditional)",
        ylabel="Normalized gradient magnitude",
        output_dir=cfg.output_dir,
        filename="band_importance_per_class_gradient.png",
    )
    plot_aggregate_importance(
        np.abs(norm).mean(axis=0), BAND_NAMES[:n_bands], BAND_COLORS[:n_bands],
        title="Which Bands Matter Most? (Gradient, averaged over classes)",
        ylabel="Mean normalized importance",
        output_dir=cfg.output_dir,
        filename="band_importance_aggregate_gradient.png",
    )
    return norm


def integrated_gradients_band_importance(
    model, data, coords, masks, cfg, num_classes, class_names, device,
    selection=None, steps: int = 32, verbose=True,
) -> np.ndarray:
    """Integrated Gradients from the band-mean baseline.

    Replaces the SHAP ``GradientExplainer`` call, which returned identically
    zero because it was handed the explicand as its own background
    distribution.

    IG satisfies *completeness*: attributions sum to
    ``score(input) - score(baseline)``. The mean relative error of that
    identity is printed, so the numbers can be trusted or discarded on evidence
    rather than faith.
    """
    if verbose:
        print(f"  Integrated Gradients ({steps} steps, baseline = band mean) ...")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_bands = cfg.num_bands
    raw = np.zeros((num_classes, n_bands), dtype="float64")
    completeness: list[float] = []

    for cls, idxs in selection.items():
        acc, used = np.zeros(n_bands), 0
        for i in idxs:
            x = _patch_tensor(data, coords[i], cfg.patch_size, device)
            region = torch.from_numpy(masks[i].astype("int64")).to(device) == cls
            baseline = torch.zeros_like(x)          # z-scored: zero == band mean

            with torch.no_grad():
                s_in = _region_mean(model(x), region, cls)
                s_base = _region_mean(model(baseline), region, cls)
            if s_in is None:
                del x, baseline
                continue

            total = torch.zeros_like(x)
            for step in range(1, steps + 1):
                alpha = step / steps
                point = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
                score = _region_mean(model(point), region, cls)
                total += torch.autograd.grad(score, point)[0]
                del point, score

            ig = (x - baseline) * total / steps
            acc += ig[0].sum(dim=(1, 2)).detach().cpu().numpy()

            # Completeness: sum(IG) should equal the score difference. The
            # score is a mean over the region, so divide by the region size.
            delta = float(s_in) - float(s_base)
            if abs(delta) > 1e-6:
                approx = float(ig.sum()) / max(int(region.sum().item()), 1)
                completeness.append(abs(approx - delta) / abs(delta))

            used += 1
            del x, baseline, total, ig
            free_cuda()

        if used:
            raw[cls] = acc / used

    for p in model.parameters():
        p.requires_grad_(True)

    if verbose and completeness:
        print(f"    completeness check: mean relative error {np.mean(completeness):.3f}")

    norm = _normalize_rows(raw)
    plot_per_class_importance(
        norm, class_names, BAND_NAMES[:n_bands], BAND_COLORS[:n_bands],
        title="Per-Class Band Attribution (Integrated Gradients, signed)",
        ylabel="Normalized attribution (+ supports class)",
        output_dir=cfg.output_dir,
        filename="band_importance_per_class_integrated.png",
    )
    return norm


def _report_agreement(results: dict[str, np.ndarray], class_names, verbose=True) -> None:
    """Rank-correlate the methods per class.

    Methods that agree are evidence. Methods that disagree mean the attribution
    is not stable enough to draw conclusions from, and reporting that is more
    useful than picking whichever plot looks tidiest.
    """
    if not verbose:
        return
    from itertools import combinations

    from scipy.stats import spearmanr

    print("\n  Method agreement (Spearman rank correlation, per class):")
    names = list(results)
    for cls, cname in enumerate(class_names):
        rhos = []
        for a, b in combinations(names, 2):
            ra, rb = np.abs(results[a][cls]), np.abs(results[b][cls])
            if ra.std() < 1e-12 or rb.std() < 1e-12:
                continue
            rho = spearmanr(ra, rb).statistic
            if not np.isnan(rho):
                rhos.append(rho)
        print(f"    {cname:14s} {'n/a' if not rhos else f'{np.mean(rhos):+.2f}'}")


def run_band_importance(
    model, data, coords, cfg, num_classes, class_names, device,
    masks=None, methods=None, verbose=True,
) -> dict[str, np.ndarray | None]:
    """Run the selected attribution methods.

    ``masks`` is required: class-conditional scoring needs to know which pixels
    belong to which class. Without them the methods are skipped rather than
    silently falling back to whole-patch means, which is what produced the
    uninformative flat bar charts in the first place.
    """
    if masks is None:
        print("  Band attribution needs reference masks - skipping.")
        return {}

    if methods is None:
        methods = tuple(
            getattr(cfg, "xai_methods", ("occlusion", "gradient", "integrated"))
        )

    n_per_class = int(getattr(cfg, "xai_patches_per_class", 6))
    if verbose:
        print(f"  Selecting up to {n_per_class} patches per class:")
    selection = select_class_patches(masks, num_classes, n_per_class, verbose=verbose)

    runners = {
        "occlusion": occlusion_band_importance,
        "gradient": gradient_band_importance,
        "integrated": integrated_gradients_band_importance,
    }

    results: dict[str, np.ndarray | None] = {}
    for name in methods:
        runner = runners.get(name)
        if runner is None:
            print(f"  Unknown attribution method '{name}' - skipped.")
            continue
        try:
            results[name] = runner(
                model, data, coords, masks, cfg, num_classes, class_names,
                device, selection=selection, verbose=verbose,
            )
        except Exception as exc:
            print(f"  {name} attribution failed: {exc}")
            results[name] = None
        finally:
            free_cuda()

    available = {k: v for k, v in results.items() if v is not None}
    if len(available) > 1:
        plot_importance_heatmap(
            available, class_names, BAND_NAMES[: cfg.num_bands], cfg.output_dir,
            "band_importance_heatmap.png",
        )
        _report_agreement(available, class_names, verbose)
    return results
