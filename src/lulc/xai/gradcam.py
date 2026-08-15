"""SegGradCAM: per-class spatial attention maps over sampled patches."""

from __future__ import annotations

import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from lulc.constants import CLASS_COLORS
from lulc.utils import ensure_dir, free_cuda
from lulc.xai.plots import to_rgb


def resolve_target_layer(model: nn.Module) -> nn.Module | None:
    """Find the last convolutional stage of the decoder.

    Grad-CAM needs a layer whose activations are still spatially arranged but
    already class-discriminative. In a UNet++ that is the final decoder block,
    not the segmentation head (too shallow) and not the encoder (too generic).
    The fallbacks below cover the layout differences between ``smp`` versions.
    """
    try:
        blocks = model.decoder.blocks
        key = list(blocks._modules)[-1]
        if blocks._modules[key] is not None:
            return blocks._modules[key]
    except (AttributeError, IndexError):
        pass

    try:
        for child in reversed(list(model.decoder.children())):
            if any(True for _ in child.parameters()):
                return child
    except AttributeError:
        pass

    last_conv = None
    for _, module in model.decoder.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def _identity_reshape(tensor, height=None, width=None):
    """Grad-CAM's reshape hook is for ViTs; CNN activations pass through."""
    return tensor


def run_gradcam(
    model,
    data: np.ndarray,
    coords,
    cfg,
    class_names,
    device,
    seed: int = 42,
    verbose: bool = True,
) -> list[str]:
    """Write one figure per sampled patch: RGB plus a CAM per class.

    Returns the list of saved file paths. Missing ``grad-cam`` is a soft
    failure - the rest of the pipeline still runs.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
    except ImportError:
        print("  grad-cam not installed (pip install grad-cam) - GradCAM skipped.")
        return []

    target_layer = resolve_target_layer(model)
    if target_layer is None:
        print("  Could not resolve a decoder target layer - GradCAM skipped.")
        return []

    rng = random.Random(seed)
    n_samples = min(cfg.xai_samples, len(coords))
    sample_coords = rng.sample(coords, n_samples)
    p = cfg.patch_size
    num_classes = len(class_names)
    out_dir = ensure_dir(cfg.output_dir)
    saved: list[str] = []

    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)

    n_cols = 4
    n_rows = int(np.ceil((num_classes + 1) / n_cols))

    for idx, (y, x) in enumerate(sample_coords, start=1):
        patch = data[:, y : y + p, x : x + p].copy()
        rgb = to_rgb(patch, cfg.rgb_indices)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.5 * n_rows))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")

        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("Sentinel-2 RGB", fontsize=12)

        for cls_idx, cls_name in enumerate(class_names):
            slot = cls_idx + 1
            ax = axes[slot // n_cols, slot % n_cols]

            tensor = torch.from_numpy(patch.copy()).unsqueeze(0).float().to(device)
            tensor.requires_grad_(True)
            targets = [
                SemanticSegmentationTarget(cls_idx, np.ones((p, p), dtype=np.float32))
            ]

            try:
                with GradCAM(
                    model=model,
                    target_layers=[target_layer],
                    reshape_transform=_identity_reshape,
                ) as cam:
                    heatmap = cam(input_tensor=tensor, targets=targets)[0]
                ax.imshow(show_cam_on_image(rgb, heatmap, use_rgb=True))
                # Darken toward the class colour: the raw palette (yellow in
                # particular) is unreadable as text on a white figure.
                title_rgb = (CLASS_COLORS[cls_idx].astype("float32") * 0.62).clip(0, 165)
                ax.set_title(
                    f"GradCAM: {cls_name}",
                    color=tuple((title_rgb / 255).tolist()),
                    fontsize=11,
                )
            except Exception as exc:
                print(f"  GradCAM failed for {cls_name}: {exc}")
                ax.set_title(f"{cls_name} - failed", fontsize=9)
            finally:
                del tensor
                free_cuda()

        fig.suptitle(f"SegGradCAM - patch {idx} at (y={y}, x={x})", fontsize=14)
        fig.tight_layout(h_pad=2.5)
        out = out_dir / f"gradcam_patch_{idx}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(out))
        if verbose:
            print(f"  Saved: {out}")

    free_cuda()
    return saved
