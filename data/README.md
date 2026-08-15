# Data

Rasters are deliberately **not** tracked in git. A single Sentinel-2 composite
runs to hundreds of megabytes and the patch tiles multiply that; keeping them
in version control makes every clone slow and every history rewrite painful.

```
data/
├── raw/         # source GeoTIFFs from GEE            (gitignored)
└── patches/     # tiles produced by make_patches.py   (gitignored)
    ├── images/
    └── masks/
```

## Getting the source imagery

1. Open [Google Earth Engine Code Editor](https://code.earthengine.google.com).
2. Paste `scripts/export_s2_gee.js`, edit the AOI rectangle, run.
3. Download the exported GeoTIFF from Drive to `data/raw/`.
4. Point `image_tif` in `configs/train.yaml` at it.

Any 6-band composite works as long as the band order is
`[B2, B3, B4, B8, B11, B12]` — Blue, Green, Red, NIR, SWIR1, SWIR2.

## Producing patches

```bash
python scripts/make_patches.py --input data/raw/city_s2_6band.tif \
    --output data/patches/images --prefix image
python scripts/make_patches.py --input data/raw/city_labels.tif \
    --output data/patches/masks  --prefix mask --single-band
```

Use identical `--patch-size` and `--stride` for both. The mask loader pairs
images to masks **by sorted filename**, so the two directories must contain the
same number of files in the same grid order.

## Label schema

Annotations were drawn with 7 classes; `remap_labels()` folds `road` into
`building` at load time, because road pixels are too thin to learn reliably at
10 m resolution.

| Trained ID | Class       | Colour             | Annotation source        |
|-----------:|-------------|--------------------|--------------------------|
| 0          | background  | black (0,0,0)      | 0 — background / no-data |
| 1          | building    | red (255,0,0)      | 1 — building, 2 — road   |
| 2          | water       | blue (0,0,255)     | 3                        |
| 3          | barren      | brown (165,42,42)  | 4                        |
| 4          | forest      | green (0,128,0)    | 5                        |
| 5          | agriculture | yellow (255,255,0) | 6                        |

## About the old `dataset/` directory

The previous repo tracked ~7,900 tiles under `dataset/`. Two things are worth
knowing before you reuse them.

**`dataset/images/` is not valid model input.** Those tiles are **single-band
`uint8`**, not 6-band. The old `patch.py` opened the 6-band composite, read
band 1 only (`src.read(1)`), and cast it to `uint8` — which collapses
Sentinel-2 reflectance (roughly 0–10000) into 0–255 and discards Green, Red,
NIR, SWIR1 and SWIR2 entirely. Feeding them to a 6-band model is impossible,
and the quantisation is lossy regardless. Regenerate them from the source
composite with `make_patches.py`, which preserves band count and dtype.

**`dataset/reference_masks/` is fine and worth keeping.** Those are valid
single-band `uint8` label rasters with values 0–6, matching the 7-class
annotation schema above. They represent real annotation effort that cannot be
regenerated from the imagery — back them up before any history rewrite.
