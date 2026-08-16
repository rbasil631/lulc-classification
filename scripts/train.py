#!/usr/bin/env python3
"""Train (fine-tune) the LULC model and run the full XAI + mapping pipeline.

Usage
-----
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --skip-xai
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from lulc.config import load_train_config
from lulc.constants import CLASS_NAMES, NUM_CLASSES
from lulc.data.raster import load_masks, tile_geotiff
from lulc.models import build_model
from lulc.training import evaluate, fine_tune
from lulc.training.trainer import build_disjoint_split
from lulc.utils import ensure_dir, free_cuda, get_device, set_seed
from lulc.viz.maps import (
    print_summary,
    save_class_distribution,
    save_lulc_geotiff,
    save_lulc_png,
)
from lulc.xai import run_band_importance, run_gradcam


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train.yaml", help="path to YAML config")
    ap.add_argument("--skip-xai", action="store_true", help="skip GradCAM and attribution")
    ap.add_argument("--skip-map", action="store_true", help="skip full-scene inference")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_train_config(args.config)
    device = get_device()
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    print("=" * 68)
    print("  LULC PIPELINE - UNet++/ResNet-50 (SSL4EO) + XAI")
    print("=" * 68)
    print(f"  Device : {device}")
    print(f"  Classes: {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")
    print(f"  Output : {cfg.output_dir}")

    if not Path(cfg.image_tif).exists():
        print(f"\nERROR: input scene not found: {cfg.image_tif}")
        print("Export a 6-band Sentinel-2 composite first (scripts/export_s2_gee.js).")
        return 1

    print("\n[1/6] Tiling scene")
    data, coords, info, _means, _stds = tile_geotiff(cfg.image_tif, cfg.patch_size, cfg.stride)

    print("\n[2/6] Loading reference masks")
    masks = load_masks(cfg.mask_dir, coords)

    print("\n[3/6] Building model")
    model = build_model(
        cfg.num_bands,
        cfg.num_classes,
        encoder_weights=cfg.encoder_weights,
        use_torchgeo=cfg.use_torchgeo,
        device=device,
    )

    print("\n[4/6] Fine-tuning")
    model = fine_tune(model, data, coords, masks, cfg, info.height, info.width, device)

    print("\n[5/6] Evaluating")
    # Recompute the same split - deterministic given cfg.seed - so metrics come
    # from held-out patches only, not from data the decoder just trained on.
    val_idx = None
    if masks is not None:
        _, val_idx = build_disjoint_split(
            coords,
            info.height,
            info.width,
            cfg.patch_size,
            cfg.val_fraction,
            random.Random(cfg.seed),
        )
        print(f"  Scoring {len(val_idx)} held-out patches")
    evaluate(model, data, coords, masks, cfg, device, indices=val_idx)

    if not args.skip_xai:
        print("\n[6/6] Explainability")
        run_gradcam(model, data, coords, cfg, CLASS_NAMES, device, seed=cfg.seed)
        run_band_importance(model, data, coords, cfg, cfg.num_classes, CLASS_NAMES, device)
    else:
        print("\n[6/6] Explainability - skipped (--skip-xai)")

    if not args.skip_map:
        from lulc.inference import sliding_window_predict

        # XAI holds large autograd graphs; release them before inference so a
        # heavy attribution pass cannot starve the full-scene pass of VRAM.
        free_cuda()
        print("\n[+] Full-scene inference")
        pred = sliding_window_predict(
            model,
            data,
            cfg.patch_size,
            cfg.stride,
            cfg.num_classes,
            batch_size=cfg.batch_size,
            device=device,
        )
        save_lulc_geotiff(pred, info, cfg.output_dir, "lulc_map.tif")
        save_lulc_png(pred, cfg.output_dir, "lulc_map.png")
        save_class_distribution(pred, cfg.output_dir, "class_distribution.png", cfg.num_classes)
        print_summary(pred, cfg.num_classes)

    final = Path(cfg.output_dir) / "lulc_model_final.pth"
    torch.save(model.state_dict(), final)
    print(f"\n  Final weights: {final}")
    print("=" * 68)
    print(f"  DONE - outputs in {cfg.output_dir}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
