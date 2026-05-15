"""
=============================================================================
  LULC INFERENCE PIPELINE v2 — New AOI Inference
=============================================================================
  Author  : Syed Basil Rizvi
  Project : Land Use Land Cover Semantic Segmentation
  Model   : ResNet50 (TorchGeo SSL4EO) + UNet++ decoder
  Usage   : Edit CONFIG, then: python lulc_inference_v2.py

  KEY DESIGN DECISIONS:
    [1] Per-AOI normalization  — stats computed from the NEW city's own pixels,
                                  exactly matching the training pipeline's FIX 1/8
    [2] weights_only=True      — safe model loading (PyTorch best practice)
    [3] GeoTIFF integrity      — CRS + transform preserved in output
    [4] Overlap averaging      — logits averaged across overlapping tiles
    [5] Median filter          — removes salt-and-pepper noise post-inference
    [6] Batch inference        — configurable batch size for GPU throughput
    [7] Debug guard            — RGB preview is opt-in, never blocks pipeline

  INSTALL:
    pip install torch torchvision segmentation-models-pytorch
    pip install rasterio matplotlib tqdm scipy
=============================================================================
"""

import os
import time
import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import median_filter
from tqdm import tqdm
import segmentation_models_pytorch as smp

# =============================================================================
#  CONFIG  — edit these paths before running
# =============================================================================

CONFIG = {
    # ── required paths ──────────────────────────────────────────────────────
    "model_path"   : r"C:\Users\rbasi\Desktop\BCODERS\lulc_results14\best_model.pth",
    "input_tif"    : r"C:\Users\rbasi\Downloads\Aligarh_LULC.tif",
    "output_dir"   : "inference_results_test2/",

    # ── model / data ─────────────────────────────────────────────────────────
    "num_bands"    : 6,
    "num_classes"  : 6,
    "patch_size"   : 256,
    "stride"       : 128,   # overlap = patch_size - stride
    "batch_size"   : 8,     # increase if VRAM allows
    "median_size"  : 5,     # post-processing filter kernel

    # ── visualization ────────────────────────────────────────────────────────
    "rgb_indices"  : [2, 1, 0],   # band indices for RGB preview (0-based)
    "show_debug"   : False,       # set True to open a matplotlib window
    "save_debug"   : True,        # always saves debug PNG regardless
    "output_dpi"   : 200,
}

# =============================================================================
#  CONSTANTS
# =============================================================================

CLASS_NAMES  = ['background', 'building', 'water', 'barren', 'forest', 'agriculture']
CLASS_COLORS = np.array([
    [0,   0,   0  ],   # background  — black
    [255, 0,   0  ],   # building    — red
    [0,   0,   255],   # water       — blue
    [165, 42,  42 ],   # barren      — brown
    [0,   128, 0  ],   # forest      — green
    [255, 255, 0  ],   # agriculture — yellow
], dtype='uint8')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
#  [1] PER-AOI NORMALIZATION
#  Computes mean/std from THIS image's valid (non-zero) pixels — same logic
#  as compute_norm_stats() + normalize_data() in the training pipeline.
#  Never uses hardcoded stats so any new city normalizes consistently.
# =============================================================================

def normalize_per_aoi(data):
    """
    Band-wise z-score normalization using the image's own statistics.

    Parameters
    ----------
    data : np.ndarray  shape (bands, H, W), float32

    Returns
    -------
    data_norm : np.ndarray  same shape, normalized in-place copy
    means     : list[float]  per-band means  (logged for reproducibility)
    stds      : list[float]  per-band stds
    """
    data_norm = data.copy()
    means, stds = [], []

    for b in range(data.shape[0]):
        band  = data[b]
        valid = band[band > 0]

        if len(valid) == 0:          # entirely masked / no-data band
            means.append(0.0)
            stds.append(1.0)
        else:
            m = float(valid.mean())
            s = float(valid.std()) + 1e-6
            means.append(m)
            stds.append(s)

        data_norm[b] = (data[b] - means[b]) / stds[b]

    return data_norm, means, stds

# =============================================================================
#  MODEL
# =============================================================================

def build_model(cfg):
    """
    Reconstruct the exact same UNet++ architecture used in training,
    then load weights with weights_only=True for security. [2]
    """
    print("\n[1/5] Loading model ...")

    model = smp.UnetPlusPlus(
        encoder_name    = 'resnet50',
        encoder_weights = None,          # weights come from the checkpoint
        in_channels     = cfg['num_bands'],
        classes         = cfg['num_classes'],
        activation      = None,
    )

    state = torch.load(
        cfg['model_path'],
        map_location = DEVICE,
        weights_only = True,             # [2] safe loading
    )
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Checkpoint : {cfg['model_path']}")
    print(f"  Parameters : {n_params:.1f}M")
    print(f"  Device     : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")

    return model

# =============================================================================
#  PATCH DATASET
# =============================================================================

class PatchDataset(Dataset):
    """Lightweight dataset that slices patches on-the-fly from a numpy array."""

    def __init__(self, data, coords, patch_size):
        self.data       = data
        self.coords     = coords
        self.patch_size = patch_size

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        p    = self.patch_size
        patch = self.data[:, y:y+p, x:x+p]
        return torch.tensor(patch).float(), torch.tensor([y, x])

# =============================================================================
#  INFERENCE
# =============================================================================

def load_and_normalize(tif_path):
    """Read GeoTIFF, normalize per-AOI, return data + rasterio metadata."""
    print(f"\n[2/5] Reading: {tif_path}")
    with rasterio.open(tif_path) as src:
        data      = src.read().astype('float32')
        meta      = src.meta.copy()
        transform = src.transform
        crs       = src.crs

    print(f"  Raw shape  : {data.shape}  (bands × H × W)")

    # [1] per-AOI normalization
    data_norm, means, stds = normalize_per_aoi(data)

    print(f"  Norm stats (band 0) : mean={means[0]:.2f}  std={stds[0]:.2f}")
    for b in range(len(means)):
        print(f"    band {b+1}: mean={means[b]:.2f}  std={stds[b]:.2f}")

    return data_norm, meta, transform, crs


def get_patch_coords(H, W, patch_size, stride):
    return [
        (y, x)
        for y in range(0, H - patch_size + 1, stride)
        for x in range(0, W - patch_size + 1, stride)
    ]


def run_inference(model, data, cfg):
    """
    Sliding-window inference with overlap averaging. [4]
    Returns the raw argmax prediction map before post-processing.
    """
    _, H, W    = data.shape
    patch_size = cfg['patch_size']
    stride     = cfg['stride']

    coords = get_patch_coords(H, W, patch_size, stride)
    print(f"\n[3/5] Inference  ({H}×{W} px | {len(coords)} patches | "
          f"batch={cfg['batch_size']})")

    dataset = PatchDataset(data, coords, patch_size)
    loader  = DataLoader(
        dataset,
        batch_size = cfg['batch_size'],
        shuffle    = False,
        num_workers= 0,
        pin_memory = torch.cuda.is_available(),
    )

    pred_sum  = np.zeros((cfg['num_classes'], H, W), dtype='float32')
    count_map = np.zeros((H, W), dtype='float32')
    t0        = time.time()

    with torch.no_grad():
        for patches, coord_batch in tqdm(loader, desc="  Running", ncols=90):
            patches = patches.to(DEVICE)

            # mixed-precision on GPU [6]
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(patches)

            logits = logits.cpu().numpy()
            ys     = coord_batch[:, 0].numpy()
            xs     = coord_batch[:, 1].numpy()

            for i in range(len(ys)):
                y, x = int(ys[i]), int(xs[i])
                pred_sum[:, y:y+patch_size, x:x+patch_size] += logits[i]
                count_map[y:y+patch_size, x:x+patch_size]   += 1

    elapsed = time.time() - t0
    print(f"  Elapsed    : {elapsed:.1f}s  ({len(coords)/elapsed:.0f} patches/s)")

    # average overlapping predictions, then argmax
    pred_avg = pred_sum / np.maximum(count_map, 1)[np.newaxis]
    pred_map = pred_avg.argmax(0).astype('uint8')

    # [5] remove salt-and-pepper noise
    print(f"  Median filter (k={cfg['median_size']}) ...")
    pred_map = median_filter(pred_map, size=cfg['median_size'])

    return pred_map

# =============================================================================
#  SAVE RESULTS
# =============================================================================

def save_results(pred_map, data, meta, transform, crs, cfg):
    """
    Saves:
      • <name>_lulc.tif          — georeferenced prediction (CRS preserved) [3]
      • <name>_lulc_color.png    — colorized LULC map with legend
      • <name>_distribution.png  — class coverage bar chart
      • <name>_debug_rgb.png     — normalized RGB preview of the input image [7]
    """
    print("\n[4/5] Saving results ...")

    os.makedirs(cfg['output_dir'], exist_ok=True)
    stem = os.path.splitext(os.path.basename(cfg['input_tif']))[0]

    # ── 1. GeoTIFF — keep CRS + transform [3] ───────────────────────────────
    tif_meta = meta.copy()
    tif_meta.update({'count': 1, 'dtype': 'uint8', 'nodata': 255})
    # meta already carries 'crs' and 'transform' from rasterio.open — do NOT
    # overwrite them; just pass meta through to preserve georeferencing
    tif_out = os.path.join(cfg['output_dir'], f'{stem}_lulc.tif')
    with rasterio.open(tif_out, 'w', **tif_meta) as dst:
        dst.write(pred_map.astype('uint8'), 1)
        dst.update_tags(
            classes     = str(CLASS_NAMES),
            colors      = str(CLASS_COLORS.tolist()),
            description = 'LULC inference v2 — per-AOI norm')
    print(f"  GeoTIFF    : {tif_out}")

    # ── 2. Colorized PNG ─────────────────────────────────────────────────────
    color_map = CLASS_COLORS[pred_map]
    fig, ax   = plt.subplots(figsize=(14, 14))
    ax.imshow(color_map)
    ax.set_title('LULC Map — Sentinel-2 (inference v2)', fontsize=16, pad=12)
    ax.axis('off')
    ax.legend(
        handles=[
            mpatches.Patch(color=CLASS_COLORS[i] / 255.0, label=CLASS_NAMES[i])
            for i in range(cfg['num_classes'])
        ],
        loc='lower right', fontsize=11, framealpha=0.9)

    unique, counts = np.unique(pred_map, return_counts=True)
    total_px = pred_map.size
    stats_txt = "\n".join(
        f"{CLASS_NAMES[int(u)]}: {100 * c / total_px:.1f}%"
        for u, c in zip(unique, counts) if int(u) < cfg['num_classes'])
    ax.text(0.01, 0.99, stats_txt, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    png_out = os.path.join(cfg['output_dir'], f'{stem}_lulc_color.png')
    plt.savefig(png_out, dpi=cfg['output_dpi'], bbox_inches='tight')
    plt.close()
    print(f"  Color map  : {png_out}")

    # ── 3. Class distribution bar chart ──────────────────────────────────────
    class_pcts = [
        100 * np.sum(pred_map == i) / total_px
        for i in range(cfg['num_classes'])
    ]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bars = ax2.bar(
        CLASS_NAMES, class_pcts,
        color=[CLASS_COLORS[i] / 255.0 for i in range(cfg['num_classes'])],
        edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Coverage (%)', fontsize=12)
    ax2.set_title('LULC Class Distribution', fontsize=13)
    ax2.set_ylim(0, max(class_pcts) * 1.15 if class_pcts else 1)
    for bar, pct in zip(bars, class_pcts):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    dist_out = os.path.join(cfg['output_dir'], f'{stem}_distribution.png')
    plt.savefig(dist_out, dpi=150)
    plt.close()
    print(f"  Distribution: {dist_out}")

    # ── 4. Debug RGB preview [7] — saved always, shown only if requested ─────
    rgb_idx = cfg['rgb_indices']
    rgb     = data[rgb_idx].transpose(1, 2, 0).astype('float32')
    rgb     = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    rgb     = np.clip(rgb, 0.0, 1.0)

    fig3, ax3 = plt.subplots(figsize=(10, 10))
    ax3.imshow(rgb)
    ax3.set_title('Input — Normalized RGB (B4/B3/B2)', fontsize=13)
    ax3.axis('off')
    dbg_out = os.path.join(cfg['output_dir'], f'{stem}_debug_rgb.png')
    plt.savefig(dbg_out, dpi=150, bbox_inches='tight')
    if cfg.get('show_debug', False):
        plt.show()
    plt.close()
    print(f"  Debug RGB  : {dbg_out}")

    return tif_out, png_out

# =============================================================================
#  SUMMARY
# =============================================================================

def print_summary(pred_map, cfg):
    print("\n[5/5] Prediction summary")
    print(f"  {'Class':15s}  {'Pixels':>10s}  {'Coverage':>10s}")
    print("  " + "─" * 40)
    total = pred_map.size
    for i, name in enumerate(CLASS_NAMES):
        px  = int(np.sum(pred_map == i))
        pct = 100 * px / total
        print(f"  {name:15s}  {px:>10,}  {pct:>9.2f}%")
    print()

# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  LULC INFERENCE v2 — New AOI")
    print("=" * 65)

    cfg = CONFIG

    # validate inputs
    for key in ('model_path', 'input_tif'):
        if not os.path.exists(cfg[key]):
            raise FileNotFoundError(
                f"[ERROR] '{key}' not found:\n  {cfg[key]}\n"
                "  Update CONFIG at the top of this script.")

    # ── pipeline ─────────────────────────────────────────────────────────────
    model                        = build_model(cfg)
    data, meta, transform, crs   = load_and_normalize(cfg['input_tif'])
    pred_map                     = run_inference(model, data, cfg)
    tif_out, png_out             = save_results(
                                       pred_map, data, meta,
                                       transform, crs, cfg)
    print_summary(pred_map, cfg)

    print("=" * 65)
    print("  INFERENCE COMPLETE")
    print(f"  Outputs → {cfg['output_dir']}")
    print(f"    GeoTIFF    : {os.path.basename(tif_out)}")
    print(f"    Color map  : {os.path.basename(png_out)}")
    print("=" * 65)


if __name__ == '__main__':
    main()