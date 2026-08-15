# Migrating from the old flat layout

The old repo kept everything at the root: `finalist.py`, `inference.py`,
`patch.py`, a 250 MB `dataset/`, and several `lulc_results*` output folders.
This branch replaces that with a `src/lulc` package, YAML configs and
entry-point scripts.

Nothing here deletes your local data. Work through it in order.

---

## 1. What carries over

| Old | New | Notes |
|-----|-----|-------|
| `dataset/reference_masks/` | `data/patches/masks/` | **Keep.** Hand-annotated, not regenerable. |
| `dataset/images/` | — | **Discard and regenerate.** See below. |
| `finalist.py` | `scripts/train.py` + `src/lulc/` | Split into modules. |
| `inference.py` | `scripts/infer.py` + `src/lulc/inference/` | |
| `patch.py` | `scripts/make_patches.py` | Now multi-band and dtype-preserving. |
| `corrected_xai.py` | `src/lulc/xai/attribution.py` | Reconstructed — was never committed. |
| `lulc_results*/` | `outputs/` | Gitignored. |
| `requirements.txt` | `requirements.txt` | Rewritten; the old one was a UTF-16 `pip freeze` of 266 packages. |

### Why `dataset/images/` has to be regenerated

Those tiles are **single-band `uint8`**. The old `patch.py` opened the 6-band
composite, read band 1 only (`src.read(1)`), and cast the result to `uint8`.
That drops Green, Red, NIR, SWIR1 and SWIR2 entirely, and crushes Sentinel-2
reflectance (roughly 0–10000) into 0–255. A 6-band model cannot consume them.

The masks are unaffected — they were always single-band `uint8` label rasters
with values 0–6, which is correct.

---

## 2. Set up

```bash
git clone -b restructure https://github.com/rbasil631/lulc-classification.git lulc-new
cd lulc-new

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[pretrained,xai]"
```

## 3. Wire in your existing data

```bash
mkdir -p data/raw data/patches/masks data/patches/images

# your 6-band Sentinel-2 composite
cp "/path/to/old/6 bands.tif" data/raw/city_s2_6band.tif

# your annotations - these are the irreplaceable part
cp /path/to/old/dataset/reference_masks/*.tif data/patches/masks/
ls data/patches/masks/*.tif | wc -l          # expect 3971
```

Regenerate the image patches with the **same** geometry the masks were cut
with, so the two directories pair up index-for-index:

```bash
python scripts/make_patches.py \
    --input data/raw/city_s2_6band.tif \
    --output data/patches/images \
    --prefix image --patch-size 256 --stride 128

ls data/patches/images/*.tif | wc -l         # must match the mask count
```

If the counts differ, the source raster is not the one the masks were cut from,
or the patch size/stride differ. `load_masks()` pairs masks to patches **by
sorted filename**, not by georeferencing, so a mismatched count aborts
fine-tuning — but a mismatched *order* would silently train on misaligned
labels. Verify the counts match before proceeding.

Sanity-check a patch:

```bash
python -c "import rasterio; s=rasterio.open('data/patches/images/image_0000.tif'); print(s.count, s.dtypes)"
# expect: 6 ('uint16',)   -- NOT 1 ('uint8',)
```

## 4. Point the config at it

Edit `configs/train.yaml`:

```yaml
image_tif: data/raw/city_s2_6band.tif
mask_dir: data/patches/masks
output_dir: outputs/train
epochs: 2                    # raise once the plumbing is confirmed
```

## 5. Run

```bash
python scripts/train.py --config configs/train.yaml --skip-xai
```

Confirm it tiles, loads masks, and completes two epochs. Then run the real
thing:

```bash
python scripts/train.py --config configs/train.yaml
python scripts/infer.py --config configs/inference.yaml --input data/raw/other_city.tif
```

---

## 6. Expect different numbers

**IoU and F1 will drop.** The old validation split held out non-overlapping
patches and then re-added overlapping ones via `extra_train`, so a held-out
patch shared pixels with several training patches. Validation loss was partly
measuring memorisation. `train.py` now recomputes a disjoint split from `seed`
and scores only held-out patches. The lower numbers are the honest ones.

**Band importance rankings will change.** The old gradient method multiplied
gradients by input magnitude, which effectively ranks bands by reflectance
scale — SWIR wins automatically. Attribution is now on gradient magnitude
alone, with occlusion and SHAP as cross-checks. Regenerate the figures rather
than comparing against the old ones.

Sanity check on any XAI run: NIR (B8) and Green (B3) should dominate for
forest and agriculture. That is the vegetation red-edge response and it is not
subtle. If they do not, check the band order in your GeoTIFF before you suspect
the model.

---

## 7. Retiring the old layout

Once you have confirmed the new branch trains and infers correctly, merge it
and clear out the old files:

```bash
git checkout main
git merge restructure
bash tools/cleanup_old_repo.sh
git push origin main
```

That removes `finalist.py`, `inference.py`, `patch.py`, `dataset/`,
`__pycache__/`, `presentable/` and `lulc_results*/` from the repo. It does not
shrink clone size — the blobs remain in the pack file.

To actually reclaim the ~250 MB, history has to be rewritten:

```bash
pip install git-filter-repo
bash tools/cleanup_old_repo.sh --purge
git remote add origin https://github.com/rbasil631/lulc-classification.git
git push --force origin main
```

This rewrites every commit SHA and requires a force-push. **Back up
`dataset/reference_masks/` somewhere off your machine first** — after the
force-push, GitHub's copy is gone and your local backup is the only one. The
script prompts for confirmation and makes its own backup, but do not rely on
that alone.
