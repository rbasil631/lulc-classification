#!/usr/bin/env python3
"""Run a trained checkpoint on a new area of interest.

Usage
-----
    python scripts/infer.py --config configs/inference.yaml
    python scripts/infer.py --config configs/inference.yaml \
        --input data/raw/delhi_6band.tif --output-dir outputs/delhi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lulc.config import load_inference_config, resolve
from lulc.inference import predict_raster
from lulc.models import load_checkpoint
from lulc.utils import ensure_dir, get_device
from lulc.viz.maps import (
    print_summary,
    save_class_distribution,
    save_lulc_geotiff,
    save_lulc_png,
    save_rgb_preview,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/inference.yaml", help="path to YAML config")
    ap.add_argument("--input", help="override input_tif")
    ap.add_argument("--model", help="override model_path")
    ap.add_argument("--output-dir", help="override output_dir")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_inference_config(args.config)
    if args.input:
        cfg.input_tif = resolve(args.input)
    if args.model:
        cfg.model_path = resolve(args.model)
    if args.output_dir:
        cfg.output_dir = resolve(args.output_dir)

    for label, path in (("model_path", cfg.model_path), ("input_tif", cfg.input_tif)):
        if not Path(path).exists():
            print(f"ERROR: {label} not found: {path}")
            return 1

    device = get_device()
    ensure_dir(cfg.output_dir)
    stem = Path(cfg.input_tif).stem

    print("=" * 68)
    print("  LULC INFERENCE - new AOI")
    print("=" * 68)

    print("\n[1/3] Loading checkpoint")
    model = load_checkpoint(cfg.model_path, cfg.num_bands, cfg.num_classes, device)

    print(f"\n[2/3] Predicting {Path(cfg.input_tif).name}")
    pred, data, info = predict_raster(model, cfg.input_tif, cfg, device)

    print("\n[3/3] Writing outputs")
    save_lulc_geotiff(pred, info, cfg.output_dir, f"{stem}_lulc.tif")
    save_lulc_png(pred, cfg.output_dir, f"{stem}_lulc.png", dpi=cfg.output_dpi)
    save_class_distribution(pred, cfg.output_dir, f"{stem}_distribution.png", cfg.num_classes)
    if cfg.save_debug_rgb:
        save_rgb_preview(data, cfg.rgb_indices, cfg.output_dir, f"{stem}_rgb.png")

    print_summary(pred, cfg.num_classes)
    print("=" * 68)
    print(f"  DONE - outputs in {cfg.output_dir}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
