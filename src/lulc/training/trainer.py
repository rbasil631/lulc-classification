"""Fine-tuning loop with a leak-free validation split and a frozen encoder."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from lulc.data.dataset import S2PatchDataset
from lulc.data.raster import patch_coords
from lulc.data.transforms import get_transforms
from lulc.losses import combined_loss
from lulc.utils import ensure_dir, get_device


def build_disjoint_split(
    coords: list[tuple[int, int]],
    height: int,
    width: int,
    patch_size: int,
    val_fraction: float = 0.1,
    rng: random.Random | None = None,
) -> tuple[list[int], list[int]]:
    """Split patch indices so no training patch overlaps a validation patch.

    The training grid is strided (patches overlap by design), so a naive
    index split leaks: a held-out patch shares pixels with several training
    patches, and validation loss then measures memorisation rather than
    generalisation.

    Validation patches are drawn from a non-overlapping grid, and any training
    patch whose window intersects a chosen validation window is dropped.

    Returns
    -------
    (train_idx, val_idx)
        Indices into ``coords``.
    """
    rng = rng or random.Random(0)

    grid = patch_coords(height, width, patch_size, patch_size)
    lookup = {c: i for i, c in enumerate(coords)}
    grid = [c for c in grid if c in lookup]
    if not grid:
        return list(range(len(coords))), []

    rng.shuffle(grid)
    n_val = max(1, int(val_fraction * len(grid)))
    val_corners = grid[:n_val]
    val_idx = [lookup[c] for c in val_corners]

    def overlaps(c):
        y, x = c
        return any(
            abs(y - vy) < patch_size and abs(x - vx) < patch_size
            for vy, vx in val_corners
        )

    train_idx = [i for i, c in enumerate(coords) if not overlaps(c)]
    return train_idx, val_idx


def make_weighted_sampler(masks: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """Sample patches inversely to how common their dominant class is.

    Without this, agriculture and background swamp the batches and the rare
    classes (water, barren) never get a gradient signal worth the name.
    """
    dominant = np.array(
        [
            int(np.argmax(np.bincount(m.ravel().astype(int), minlength=num_classes)))
            for m in masks
        ]
    )
    counts = np.maximum(np.bincount(dominant, minlength=num_classes), 1).astype("float32")
    weights = (1.0 / counts)[dominant]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.float),
        num_samples=len(weights),
        replacement=True,
    )


def _freeze_encoder(model: nn.Module, verbose: bool = True) -> None:
    """Train the decoder only.

    The SSL4EO encoder saw millions of Sentinel-2 scenes; a few hundred local
    patches will overwrite that with AOI-specific noise if left unfrozen.
    """
    for p in model.encoder.parameters():
        p.requires_grad = False
    for module in (model.decoder, model.segmentation_head):
        for p in module.parameters():
            p.requires_grad = True
    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Encoder frozen | trainable: {trainable / 1e6:.1f}M params")


def _train_step(model, imgs, msks, optimizer, device, label_smoothing, grad_clip) -> float:
    imgs, msks = imgs.to(device), msks.long().to(device)
    loss = combined_loss(model(imgs), msks, label_smoothing)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def _val_loss(model, loader, device) -> float:
    model.eval()
    total = 0.0
    for imgs, msks in loader:
        total += combined_loss(
            model(imgs.to(device)), msks.long().to(device), label_smoothing=0.0
        ).item()
    return total / max(len(loader), 1)


def fine_tune(model, data, coords, masks, cfg, height, width, device=None):
    """Fine-tune the decoder on reference masks.

    Returns the model with the best-validation-loss weights reloaded. If
    ``masks`` is ``None`` the model is returned untouched.
    """
    if masks is None:
        print("  No reference masks - skipping fine-tuning.")
        return model

    device = device or get_device()
    rng = random.Random(cfg.seed)

    if cfg.freeze_encoder:
        _freeze_encoder(model)
    else:
        for p in model.parameters():
            p.requires_grad = True

    train_idx, val_idx = build_disjoint_split(
        coords, height, width, cfg.patch_size, cfg.val_fraction, rng
    )
    if not train_idx:
        print("  WARNING: split left no training patches - skipping fine-tuning.")
        return model

    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val (disjoint windows)")

    train_coords = [coords[i] for i in train_idx]
    train_masks = masks[train_idx]
    train_tf, eval_tf = get_transforms()

    train_dl = DataLoader(
        S2PatchDataset(data, train_coords, cfg.patch_size, train_masks, train_tf),
        batch_size=cfg.batch_size,
        sampler=make_weighted_sampler(train_masks, cfg.num_classes),
        num_workers=cfg.num_workers,
    )

    val_dl = None
    if val_idx:
        val_dl = DataLoader(
            S2PatchDataset(
                data, [coords[i] for i in val_idx], cfg.patch_size, masks[val_idx], eval_tf
            ),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    ckpt = ensure_dir(cfg.output_dir) / "best_model.pth"
    best_val, patience = float("inf"), 0

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = sum(
            _train_step(
                model, imgs, msks, optimizer, device, cfg.label_smoothing, cfg.grad_clip
            )
            for imgs, msks in train_dl
        ) / max(len(train_dl), 1)

        val_loss = _val_loss(model, val_dl, device) if val_dl else train_loss
        scheduler.step()
        print(
            f"  Epoch {epoch + 1:02d}/{cfg.epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("  Early stopping.")
                break

    if Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    if cfg.freeze_encoder and cfg.encoder_warmup_batches > 0:
        _gentle_encoder_update(model, train_dl, cfg, device)

    print(f"  Fine-tuning complete | best val loss {best_val:.4f}")
    return model


def _gentle_encoder_update(model, loader, cfg, device) -> None:
    """Unfreeze the encoder for a handful of low-LR batches.

    Just enough to let the encoder's batch-norm statistics adapt to the local
    reflectance distribution without unpicking the pretrained features.
    """
    print(f"  Gentle encoder update ({cfg.encoder_warmup_batches} batches @ lr/10) ...")
    for p in model.encoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr * 0.1)
    model.train()
    for i, (imgs, msks) in enumerate(loader):
        if i >= cfg.encoder_warmup_batches:
            break
        _train_step(model, imgs, msks, optimizer, device, cfg.label_smoothing, cfg.grad_clip)
