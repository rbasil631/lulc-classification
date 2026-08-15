"""Per-class IoU / F1 and the confusion matrix."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from lulc.constants import CLASS_NAMES
from lulc.data.dataset import S2PatchDataset
from lulc.data.transforms import get_transforms
from lulc.utils import ensure_dir, get_device


@torch.no_grad()
def evaluate(model, data, coords, masks, cfg, device=None, indices=None) -> dict | None:
    """Score the model and write ``confusion_matrix.png``.

    Parameters
    ----------
    indices : list[int] | None
        Restrict evaluation to these positions in ``coords`` - pass the held-out
        validation indices for an honest number. Defaults to all patches, which
        includes data the model trained on.

    Returns
    -------
    dict | None
        ``{"iou": ndarray, "f1": ndarray, "miou": float, "mean_f1": float,
        "confusion_matrix": ndarray}``, or ``None`` when masks are unavailable.
    """
    if masks is None:
        print("  No masks - skipping evaluation.")
        return None

    device = device or get_device()
    if indices is not None:
        coords = [coords[i] for i in indices]
        masks = masks[indices]

    _, eval_tf = get_transforms()
    loader = DataLoader(
        S2PatchDataset(data, coords, cfg.patch_size, masks, eval_tf),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    jaccard = MulticlassJaccardIndex(num_classes=cfg.num_classes, average="none").to(device)
    f1_metric = MulticlassF1Score(num_classes=cfg.num_classes, average="none").to(device)

    preds, labels = [], []
    model.eval()
    for imgs, msks in loader:
        preds.append(model(imgs.to(device)).argmax(dim=1))
        labels.append(msks.to(device))

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    iou = jaccard(preds, labels).cpu().numpy()
    f1 = f1_metric(preds, labels).cpu().numpy()

    print(f"\n  {'Class':15s} {'IoU':>8s} {'F1':>8s}")
    print("  " + "-" * 33)
    for name, i, f in zip(CLASS_NAMES, iou, f1):
        print(f"  {name:15s} {i:8.4f} {f:8.4f}")
    print(f"\n  mIoU {iou.mean():.4f} | mean F1 {f1.mean():.4f}")

    cm = confusion_matrix(
        labels.cpu().ravel().numpy(),
        preds.cpu().ravel().numpy(),
        labels=list(range(cfg.num_classes)),
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    out = ensure_dir(cfg.output_dir) / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    return {
        "iou": iou,
        "f1": f1,
        "miou": float(iou.mean()),
        "mean_f1": float(f1.mean()),
        "confusion_matrix": cm,
    }
