# LULC Classification — Sentinel-2 Semantic Segmentation with Explainable AI

Pixel-wise land use / land cover mapping from Sentinel-2 imagery, using a
UNet++ decoder over a ResNet-50 encoder pretrained on Sentinel-2 (TorchGeo
SSL4EO), with three complementary explainability methods layered on top.

The point of the XAI half is not decoration. A segmentation model that reports
92% accuracy can still be reading the wrong spectral evidence — keying on SWIR
haze instead of NIR vegetation response, say — and per-class band attribution
is how you catch that before publishing a map someone might plan around.

---

## What it does

- Reads a 6-band Sentinel-2 L2A composite (B2, B3, B4, B8, B11, B12 @ 10 m)
- Normalizes **per scene**, from the scene's own valid pixels — the single
  change that lets one checkpoint transfer to a city it never trained on
- Fine-tunes a frozen-encoder UNet++ on local reference masks, with a
  validation split guaranteed not to overlap any training patch
- Scores per-class IoU / F1 and writes a confusion matrix
- Explains predictions three ways: SegGradCAM (where), plus gradient,
  occlusion and SHAP band attribution (which wavelengths)
- Runs sliding-window inference over the full scene, averaging **logits**
  across overlaps so tile boundaries don't show
- Writes a georeferenced GeoTIFF that opens in QGIS directly over the input

## Classes

| ID | Class | Colour |
|---:|-------|--------|
| 0 | background | ⬛ black |
| 1 | building | 🟥 red |
| 2 | water | 🟦 blue |
| 3 | barren | 🟫 brown |
| 4 | forest | 🟩 green |
| 5 | agriculture | 🟨 yellow |

Road is merged into `building` at load time — see [`data/README.md`](data/README.md).

---

> **Migrating from the old flat layout?** Read
> [`MIGRATION.md`](MIGRATION.md) first — it covers moving your existing
> `dataset/reference_masks/` across and why `dataset/images/` must be
> regenerated.

## Quick start

```bash
git clone https://github.com/rbasil631/lulc-classification.git
cd lulc-classification

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[pretrained,xai]"
```

Get imagery (see [`data/README.md`](data/README.md) for the GEE export), point
the config at it, then:

```bash
python scripts/train.py --config configs/train.yaml
python scripts/infer.py --config configs/inference.yaml --input data/raw/new_city.tif
```

No GPU? Everything runs on CPU — slowly. Start with `--skip-xai` and a small
`epochs` value to confirm the plumbing works before committing to a full run.

---

## Repository layout

```
lulc-classification/
├── configs/
│   ├── train.yaml                  # all training hyperparameters and paths
│   └── inference.yaml              # inference paths and knobs
├── scripts/
│   ├── train.py                    # fine-tune + evaluate + XAI + map
│   ├── infer.py                    # run a checkpoint on a new AOI
│   ├── make_patches.py             # tile a GeoTIFF for annotation
│   └── export_s2_gee.js            # Google Earth Engine export
├── src/lulc/
│   ├── config.py                   # YAML → dataclass, path resolution
│   ├── constants.py                # class schema, band definitions
│   ├── losses.py                   # cross-entropy + Dice
│   ├── utils.py                    # device, seeding, dir helpers
│   ├── data/
│   │   ├── raster.py               # read, normalize, tile, load masks
│   │   ├── dataset.py              # lazy patch datasets
│   │   └── transforms.py           # augmentation (albumentations 1.x/2.x)
│   ├── models/unetpp.py            # UNet++/ResNet-50, SSL4EO weight loading
│   ├── training/
│   │   ├── trainer.py              # fine-tuning, disjoint split, sampler
│   │   └── evaluate.py             # IoU, F1, confusion matrix
│   ├── inference/predict.py        # sliding-window inference
│   ├── xai/
│   │   ├── gradcam.py              # SegGradCAM
│   │   ├── attribution.py          # gradient / occlusion / SHAP
│   │   └── plots.py                # shared chart helpers
│   └── viz/maps.py                 # GeoTIFF, colour map, distribution chart
├── data/README.md                  # how to get imagery; rasters gitignored
├── docs/methodology.md             # design decisions and known limitations
├── tools/cleanup_old_repo.sh       # retire the old flat layout
└── outputs/                        # run artifacts (gitignored)
```

---

## Configuration

Everything lives in YAML. Paths resolve against the repo root unless absolute,
so the same file works on Windows, Linux and Colab.

```yaml
image_tif: data/raw/city_s2_6band.tif
mask_dir: data/patches/masks
output_dir: outputs/train

patch_size: 256
stride: 128
batch_size: 2          # 8–16 on a 12 GB+ GPU
epochs: 20
lr: 0.001
freeze_encoder: true
```

The two knobs worth understanding before you change anything else:

**`freeze_encoder`** — the SSL4EO encoder was trained on millions of Sentinel-2
scenes. A few hundred local patches will overwrite that with AOI-specific noise
if you unfreeze it. Leave it `true` unless you have thousands of labelled
patches.

**`stride`** — controls training-grid overlap and inference smoothness. Lower
means more patches, slower runs, softer tile seams. Validation always uses a
non-overlapping grid regardless.

---

## Outputs

| File | Contents |
|------|----------|
| `lulc_map.tif` | Georeferenced class map, CRS and transform preserved |
| `lulc_map.png` | Colourized map with legend and coverage box |
| `class_distribution.png` | Per-class coverage bar chart |
| `confusion_matrix.png` | Reference vs. predicted |
| `gradcam_patch_*.png` | Per-class spatial attention |
| `band_importance_per_class_*.png` | Band attribution, one per method |
| `best_model.pth` | Best-validation-loss checkpoint |

---

## Method

| Component | Choice |
|-----------|--------|
| Encoder | ResNet-50, SSL4EO Sentinel-2 MoCo weights (ImageNet fallback) |
| Decoder | UNet++ |
| Loss | 0.5 × cross-entropy (label smoothing 0.1) + 0.5 × Dice |
| Optimizer | Adam + cosine annealing, gradient clipping at 1.0 |
| Sampling | Weighted by inverse dominant-class frequency |
| Augmentation | Flips, rot90, brightness/contrast, Gaussian noise, coarse dropout, sharpen |
| Normalization | Per-scene z-score over valid pixels |
| Inference | Sliding window, logit averaging, median filter |

Rationale and known limitations: [`docs/methodology.md`](docs/methodology.md).

---

## Notes on this version

This is a restructured rewrite of the original single-file pipeline
(`final.py` / `inference.py`). Behavioural changes worth knowing about:

- **`corrected_xai.py` was reconstructed.** The original `final.py` imported
  it, but it was never committed — only a `.pyc` survived. `xai/attribution.py`
  reproduces its three methods from the surviving bytecode. Regenerate figures
  rather than comparing against old ones.
- **Validation split fixed.** The previous split re-added overlapping training
  coordinates after holding out validation patches, so validation loss was
  partly measuring memorization. Training patches that intersect a validation
  window are now dropped.
- **Albumentations 1.x/2.x.** The old code called the 1.x API against a pinned
  2.0.8, which crashed on `GaussNoise(var_limit=…)`. Both are now handled.
- **Checkpoints load with `weights_only=True`.** A `.pth` is untrusted input.
- **Metrics now score held-out patches only.** `train.py` recomputes the
  split deterministically from `seed` and passes it to `evaluate()`, so the
  printed IoU/F1 no longer include patches the decoder just trained on.
- **`dataset/images/` from the old repo is unusable** — single-band `uint8`,
  not 6-band. See [`data/README.md`](data/README.md). The reference masks are
  fine and worth preserving.

---

## Requirements

Python ≥ 3.10, PyTorch ≥ 2.0. CUDA optional but strongly recommended —
`grad-cam` and `shap` are optional extras and degrade gracefully when absent.

## Author

Syed Basil Rizvi — B.Tech Artificial Intelligence, minor project.
