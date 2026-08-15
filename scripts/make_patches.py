#!/usr/bin/env python3
"""Cut a large GeoTIFF into fixed-size patches for annotation.

Used to slice both the Sentinel-2 composite and the hand-drawn reference
raster into matching 256 px tiles. Run it twice with the same
``--patch-size``/``--stride`` so image and mask indices line up - the mask
loader pairs them by sort order, not by georeferencing.

Usage
-----
    python scripts/make_patches.py --input data/raw/city_s2_6band.tif \
        --output data/patches/images --prefix image
    python scripts/make_patches.py --input data/raw/city_labels.tif \
        --output data/patches/masks --prefix mask --single-band
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rasterio


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="source GeoTIFF")
    ap.add_argument("--output", required=True, help="destination directory")
    ap.add_argument("--prefix", default="patch", help="output filename prefix")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument(
        "--single-band",
        action="store_true",
        help="write band 1 only as uint8 (use for label rasters)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    p, stride = args.patch_size, args.stride

    with rasterio.open(args.input) as src:
        data = src.read(1) if args.single_band else src.read()
        base_meta = src.meta.copy()
        transform = src.transform

    height, width = data.shape[-2:]
    count = 1 if args.single_band else data.shape[0]
    dtype = "uint8" if args.single_band else base_meta["dtype"]
    idx = 0

    for y in range(0, height - p + 1, stride):
        for x in range(0, width - p + 1, stride):
            patch = data[y : y + p, x : x + p] if args.single_band else data[:, y : y + p, x : x + p]
            meta = {
                **base_meta,
                "height": p,
                "width": p,
                "count": count,
                "dtype": dtype,
                # keep each tile georeferenced to its true position
                "transform": rasterio.windows.transform(
                    rasterio.windows.Window(x, y, p, p), transform
                ),
            }
            out = out_dir / f"{args.prefix}_{idx:04d}.tif"
            with rasterio.open(out, "w", **meta) as dst:
                if args.single_band:
                    dst.write(patch.astype("uint8"), 1)
                else:
                    dst.write(patch)
            idx += 1

    print(f"Wrote {idx} patches ({p}px, stride {stride}) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
