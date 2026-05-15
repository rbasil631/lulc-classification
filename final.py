"""
=============================================================================
  LULC MAPPING PIPELINE — TorchGeo Pretrained Sentinel-2 + XAI
=============================================================================
  Author  : Syed Basil Rizvi
  Project : Land Use Land Cover Semantic Segmentation + Explainable AI
  Dataset : Sentinel-2 (your city via Google Earth Engine)
  Model   : ResNet50 pretrained on S2 (TorchGeo SSL4EO weights) + UNet++ decoder
  XAI     : SegGradCAM + SHAP / Gradient fallback
  Output  : Georeferenced LULC GeoTIFF + XAI visualizations

  USAGE:
    1. Edit CONFIG section below (city coordinates, paths, etc.)
    2. Run GEE script (see comment block) to download Sentinel-2
    3. python lulc_torchgeo_pipeline.py

  INSTALL:
    pip install torch torchvision torchgeo
    pip install segmentation-models-pytorch albumentations
    pip install rasterio matplotlib seaborn torchmetrics
    pip install grad-cam shap scipy scikit-learn
=============================================================================
"""

import os
import glob
import warnings
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import torch.nn as nn
import rasterio
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import median_filter
from sklearn.metrics import confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score

warnings.filterwarnings('ignore')

# =============================================================================
#  CONFIG
# =============================================================================

CONFIG = {
    "city_tif"           : r"C:\Users\rbasi\Desktop\LULC\6 bands.tif",
    "output_dir"         : "lulc_results/",
    "patch_size"         : 256,
    "stride"             : 128,
    "batch_size"         : 2,
    "epochs"             : 5,
    "lr"                 : 1e-4,
    "early_stop_patience": 2,
    "num_bands"          : 6,
    "rgb_indices"        : [2, 1, 0],
    "mask_dir"           : r"c:\Users\rbasi\Desktop\LULC\aligarh annotated\reference_masks",
    "xai_samples"        : 5,
}

# =============================================================================
#  CONSTANTS
# =============================================================================

CLASS_NAMES = ['background', 'building', 'water', 'barren', 'forest', 'agriculture']

CLASS_COLORS = np.array([
    [0,   0,   0  ],   # background  — black
    [255, 0,   0  ],   # building    — red
    [0,   0,   255],   # water       — blue
    [165, 42,  42 ],   # barren      — brown
    [0,   128, 0  ],   # forest      — green
    [255, 255, 0  ],   # agriculture — yellow
], dtype='uint8')

S2_MEAN = [752.40, 884.29, 1140.79, 1336.05, 1102.15, 761.42]
S2_STD  = [1108.02, 1155.18, 1183.34, 1368.11, 1076.19, 918.66]

NUM_CLASSES = len(CLASS_NAMES)   # 6
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
#  GEE SCRIPT (paste into https://code.earthengine.google.com)
# =============================================================================
"""
var city = ee.Geometry.Rectangle([78.0, 27.8, 78.2, 28.0]);
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(city).filterDate('2024-01-01', '2024-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
  .median().select(['B2','B3','B4','B8','B11','B12']);
Export.image.toDrive({image:s2, description:'my_city_s2', scale:10,
  region:city, fileFormat:'GeoTIFF', maxPixels:1e10});
"""

# =============================================================================
#  HELPER — label remapping
# =============================================================================

def remap_labels(mask):
    """Merge road (label 2) into building (1); shift higher labels down by 1."""
    mask = mask.copy()
    mask[mask == 2] = 1
    mask[mask > 2] -= 1
    return mask

# =============================================================================
#  DATASET
# =============================================================================

class S2PatchDataset(Dataset):
    """Lazily extracts patches from a (bands, H, W) normalized array."""

    def __init__(self, data, coords, patch_size, masks=None, transform=None):
        self.data       = data
        self.coords     = coords
        self.patch_size = patch_size
        self.masks      = masks      # (N, patch_H, patch_W) int64 or None
        self.transform  = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        p    = self.patch_size
        patch = self.data[:, y:y+p, x:x+p].transpose(1, 2, 0)  # (H, W, bands)

        if self.masks is not None:
            mask = self.masks[idx].astype('int64')
            if self.transform:
                aug   = self.transform(image=patch, mask=mask)
                patch = aug['image']
                mask  = aug['mask']
            else:
                patch = torch.tensor(patch.transpose(2, 0, 1)).float()
                mask  = torch.tensor(mask).long()
            return patch, mask
        else:
            if self.transform:
                patch = self.transform(image=patch)['image']
            else:
                patch = torch.tensor(patch.transpose(2, 0, 1)).float()
            return patch


def get_transforms():
    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ])
    val_tf = A.Compose([ToTensorV2()])
    return train_tf, val_tf

# =============================================================================
#  NORMALIZATION + TILING
# =============================================================================

def normalize_data(data):
    """Band-wise z-score normalization using S2 statistics."""
    out = data.copy()
    for b in range(data.shape[0]):
        out[b] = (data[b] - S2_MEAN[b]) / (S2_STD[b] + 1e-6)
    return out


def tile_geotiff(tif_path, patch_size, stride):
    """
    Read and normalize the city GeoTIFF; generate patch coordinates.

    Returns
    -------
    data      : (bands, H, W) float32 — normalized
    coords    : list of (y, x) top-left corners
    (H, W)    : image shape
    meta      : rasterio metadata
    transform : affine transform
    crs       : coordinate reference system
    """
    print(f"\n[STEP 3] Tiling {tif_path} ...")
    with rasterio.open(tif_path) as src:
        data      = src.read().astype('float32')
        meta      = src.meta.copy()
        transform = src.transform
        crs       = src.crs

    data = normalize_data(data)
    _, H, W = data.shape

    coords = [
        (y, x)
        for y in range(0, H - patch_size + 1, stride)
        for x in range(0, W - patch_size + 1, stride)
    ]
    print(f"  Image: {H}x{W} | Patches: {len(coords)}")
    return data, coords, (H, W), meta, transform, crs

# =============================================================================
#  MODEL
# =============================================================================

def build_model(num_bands=6, num_classes=6):
    """
    UNet++ with ResNet50 encoder.
    Loads TorchGeo SSL4EO weights where shapes match; falls back to ImageNet.
    conv1 weight is intentionally skipped (6-band vs 3-band shape mismatch).
    """
    print("\n[STEP 4] Building model ...")
    model = smp.UnetPlusPlus(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=num_bands,
        classes=num_classes,
        activation=None,
    )

    try:
        from torchgeo.models import ResNet50_Weights, resnet50 as tg_resnet50
        tg_state  = tg_resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO).state_dict()
        enc_state = model.encoder.state_dict()
        matched   = {k: v for k, v in tg_state.items()
                     if k in enc_state and v.shape == enc_state[k].shape}
        enc_state.update(matched)
        model.encoder.load_state_dict(enc_state, strict=False)
        print(f"  TorchGeo weights: {len(matched)}/{len(enc_state)} layers matched")
    except Exception as e:
        print(f"  TorchGeo weights unavailable ({e}). Using ImageNet init.")

    model.to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model

# =============================================================================
#  LOSS
# =============================================================================

def combined_loss(pred, target):
    target = target.long()
    ce   = nn.CrossEntropyLoss()(pred, target)
    dice = smp.losses.DiceLoss(mode='multiclass')(pred, target)
    return 0.5 * ce + 0.5 * dice

# =============================================================================
#  MASK LOADING
# =============================================================================

def load_masks(mask_dir, coords, patch_size):
    """
    Load and remap reference mask GeoTIFFs.
    Returns (N, patch_H, patch_W) int64 array or None.
    Expects exactly len(coords) *.tif files in mask_dir.
    """
    if not (mask_dir and os.path.isdir(mask_dir)):
        return None

    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.tif')))
    if len(mask_files) != len(coords):
        print(f"  WARNING: {len(mask_files)} mask files vs {len(coords)} patches "
              "— skipping fine-tuning.")
        return None

    masks = np.stack([
        remap_labels(rasterio.open(f).read(1).astype('int64'))
        for f in mask_files
    ])
    print(f"  Loaded {len(masks)} reference masks.")
    return masks

# =============================================================================
#  FINE-TUNING
# =============================================================================

def _train_step(model, imgs, msks, optimizer):
    imgs, msks = imgs.to(DEVICE), msks.long().to(DEVICE)
    loss = combined_loss(model(imgs), msks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def fine_tune(model, data, coords, masks, cfg):
    """
    90/10 train/val fine-tune with early stopping.
    Skipped when masks is None.

    Parameters
    ----------
    model  : nn.Module
    data   : (bands, H, W) normalized float32 array
    coords : list of (y, x) — must align with masks axis-0
    masks  : (N, patch_H, patch_W) int64 array, or None
    cfg    : CONFIG dict
    """
    if masks is None:
        print("\n[STEP 6] No reference masks — skipping fine-tuning.")
        return model

    print("\n[STEP 6] Fine-tuning ...")
    for p in model.parameters():
        p.requires_grad = True

    n_train  = int(0.9 * len(coords))
    train_tf, val_tf = get_transforms()

    train_dl = DataLoader(
        S2PatchDataset(data, coords[:n_train], cfg['patch_size'],
                       masks[:n_train], train_tf),
        batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    val_dl = DataLoader(
        S2PatchDataset(data, coords[n_train:], cfg['patch_size'],
                       masks[n_train:], val_tf),
        batch_size=cfg['batch_size'], num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'])

    best_val, patience = float('inf'), 0
    ckpt = os.path.join(cfg['output_dir'], 'best_model.pth')

    for epoch in range(cfg['epochs']):
        model.train()
        train_loss = (
            sum(_train_step(model, imgs, msks, optimizer)
                for imgs, msks in train_dl)
            / max(len(train_dl), 1)
        )

        model.eval()
        with torch.no_grad():
            val_loss = (
                sum(combined_loss(model(imgs.to(DEVICE)),
                                  msks.long().to(DEVICE)).item()
                    for imgs, msks in val_dl)
                / max(len(val_dl), 1)
            )

        scheduler.step()
        print(f"  Epoch {epoch+1:02d}/{cfg['epochs']} "
              f"| Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= cfg['early_stop_patience']:
                print("  Early stopping.")
                break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    print("  Fine-tuning complete.")
    return model

# =============================================================================
#  EVALUATION
# =============================================================================

def evaluate(model, data, coords, masks, cfg):
    """Per-class IoU, F1, and confusion matrix. Skipped when masks is None."""
    if masks is None:
        print("\n[STEP 7] No masks — skipping evaluation.")
        return

    print("\n[STEP 7] Evaluating ...")
    _, val_tf = get_transforms()
    dl = DataLoader(
        S2PatchDataset(data, coords, cfg['patch_size'], masks, val_tf),
        batch_size=cfg['batch_size'], num_workers=0)

    jaccard   = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average='none').to(DEVICE)
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES,      average='none').to(DEVICE)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, msks in dl:
            preds = model(imgs.to(DEVICE)).argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(msks.to(DEVICE))

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    iou = jaccard(all_preds, all_labels)
    f1  = f1_metric(all_preds, all_labels)

    print(f"\n  {'Class':15s} {'IoU':>8s} {'F1':>8s}")
    print("  " + "-" * 35)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:15s} {iou[i]:.4f}   {f1[i]:.4f}")
    print(f"\n  mIoU: {iou.mean():.4f}  |  Mean F1: {f1.mean():.4f}")

    cm = confusion_matrix(
        all_labels.cpu().flatten().numpy(),
        all_preds.cpu().flatten().numpy(),
        labels=list(range(NUM_CLASSES)))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    out = os.path.join(cfg['output_dir'], 'confusion_matrix.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

# =============================================================================
#  XAI — shared helpers
# =============================================================================

def _get_target_layer(model):
    """
    Resolve a GradCAM target layer from the UNet++ decoder.
    Priority: decoder.blocks (SMP >= 0.3) -> last decoder child -> last Conv2d.
    """
    # 1. SMP UNet++ standard layout: decoder.blocks is a ModuleDict
    try:
        blocks = model.decoder.blocks
        key    = list(blocks._modules.keys())[-1]
        layer  = blocks._modules[key]
        if layer is not None:
            return layer
    except (AttributeError, IndexError):
        pass

    # 2. Last direct decoder child with parameters
    try:
        for child in reversed(list(model.decoder.children())):
            if any(True for _ in child.parameters()):
                return child
    except AttributeError:
        pass

    # 3. Last Conv2d anywhere in the decoder
    last_conv = None
    for _, m in model.decoder.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv   # may be None — caller handles that


def _seg_reshape_transform(tensor, height=None, width=None):
    """
    No-op reshape transform for pytorch-grad-cam >= 1.4.
    UNet++ decoder output is already (B, C, H, W).
    The height/width kwargs satisfy the updated call signature.
    """
    return tensor


def _plot_spatial_importance(importance, title, cfg,
                             filename='band_importance.png',
                             cbar_label='Importance'):
    """
    Plot up to the first 3 bands of a (bands, H, W) importance array
    as spatial heatmaps and save to disk.
    Uses explicit figure references — never closes the wrong figure.
    """
    band_names = ['Blue (B2)', 'Green (B3)', 'Red (B4)']
    n_plot = min(3, importance.shape[0])

    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 5))
    if n_plot == 1:
        axes = [axes]

    for i in range(n_plot):
        imp  = importance[i]
        vmax = float(imp.max()) if imp.max() > 0 else 1.0
        im   = axes[i].imshow(imp, cmap='hot', vmin=0.0, vmax=vmax)
        axes[i].set_title(band_names[i], fontsize=12)
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i], fraction=0.046, label=cbar_label)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    out = os.path.join(cfg['output_dir'], filename)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

# =============================================================================
#  XAI — gradient x input fallback
# =============================================================================

def _run_gradient_importance(model, patch, cfg):
    """
    Gradient x Input attribution — universal XAI fallback.

    Outputs
    -------
    shap_attribution.png            spatial heatmap, first 3 bands
    band_importance.png             aggregate per-band bar chart
    band_importance_per_class.png   per-class breakdown bar chart

    Design notes
    ------------
    - Model weights are frozen (requires_grad=False) so only the input
      gradient is computed — cleaner and faster than full backprop.
    - Each class uses a fresh input tensor to avoid freed-graph errors.
    - All figures use explicit handles; plt.close(fig) is always called
      on the owning figure object.
    - Weights are restored to requires_grad=True before returning.
    """
    print("  Running gradient x input band importance ...")
    model.eval()

    # Freeze model — differentiate w.r.t. INPUT only
    for param in model.parameters():
        param.requires_grad_(False)

    band_labels = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    band_colors = ['#4472C4', '#70AD47', '#FF0000', '#7030A0', '#C55A11', '#843C0C']
    n_bands     = patch.shape[0]

    # ── 1. Per-class band importance ─────────────────────────────────────────
    per_class_imp = np.zeros((NUM_CLASSES, n_bands), dtype='float32')

    for cls in range(NUM_CLASSES):
        t = torch.tensor(patch).unsqueeze(0).float().to(DEVICE)
        t.requires_grad_(True)
        out   = model(t)
        score = out[:, cls, :, :].mean()
        score.backward()                              # fresh graph per class
        grads = t.grad.data.abs().cpu().numpy()[0]   # (bands, H, W)
        per_class_imp[cls] = (grads * np.abs(patch)).sum(axis=(1, 2))
        del t, out, score
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Per-class bar chart (classes on x-axis, bands in legend)
    x     = np.arange(NUM_CLASSES)
    width = 0.8 / n_bands
    fig3, ax3 = plt.subplots(figsize=(13, 5))
    for b in range(n_bands):
        ax3.bar(x + b * width, per_class_imp[:, b],
                width=width, label=band_labels[b],
                color=band_colors[b], edgecolor='black', linewidth=0.4)
    ax3.set_xticks(x + width * (n_bands - 1) / 2)
    ax3.set_xticklabels(CLASS_NAMES, rotation=20, ha='right', fontsize=10)
    ax3.set_ylabel('|grad x input| (summed)', fontsize=12)
    ax3.set_title('Per-Class Band Importance', fontsize=13)
    ax3.legend(title='Band', fontsize=9, ncol=3)
    fig3.tight_layout()
    out3 = os.path.join(cfg['output_dir'], 'band_importance_per_class.png')
    fig3.savefig(out3, dpi=150)
    plt.close(fig3)
    print(f"  Saved: {out3}")

    # ── 2. Aggregate importance across all classes ────────────────────────────
    t_agg = torch.tensor(patch).unsqueeze(0).float().to(DEVICE)
    t_agg.requires_grad_(True)
    out_agg = model(t_agg)
    out_agg.sum().backward()
    grads_agg   = t_agg.grad.data.abs().cpu().numpy()[0]  # (bands, H, W)
    importance  = grads_agg * np.abs(patch)               # (bands, H, W)
    band_totals = importance.sum(axis=(1, 2))              # (bands,)
    del t_agg, out_agg
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Spatial heatmap (first 3 bands)
    _plot_spatial_importance(
        importance, 'Band Importance (Gradient x Input)', cfg,
        filename='shap_attribution.png', cbar_label='|grad x input|')

    # Aggregate bar chart
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(band_labels[:n_bands], band_totals,
            color=band_colors[:n_bands], edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Total |grad x input|', fontsize=12)
    ax2.set_title('Which Bands Matter Most?', fontsize=13)
    for idx, val in enumerate(band_totals):
        ax2.text(idx, val * 1.01, f'{val:.1f}', ha='center', fontsize=9)
    fig2.tight_layout()
    bar_out = os.path.join(cfg['output_dir'], 'band_importance.png')
    fig2.savefig(bar_out, dpi=150)
    plt.close(fig2)
    print(f"  Saved: {bar_out}")

    # Restore model gradients for downstream steps
    for param in model.parameters():
        param.requires_grad_(True)

# =============================================================================
#  XAI — SHAP (with gradient fallback)
# =============================================================================

def _run_shap(model, patch, cfg):
    """
    SHAP GradientExplainer band-importance analysis.
    Falls back to _run_gradient_importance if SHAP is missing or fails.

    Correctly handles both SHAP return formats:
      shap >= 0.40 : list of num_classes arrays, each shape (1, bands, H, W)
      older shap   : single ndarray of shape (num_classes, 1, bands, H, W)
    """
    try:
        import shap
    except ImportError:
        print("  SHAP not installed (pip install shap) — using gradient fallback.")
        _run_gradient_importance(model, patch, cfg)
        return

    print("  Running SHAP GradientExplainer ...")
    model.eval()

    background = torch.tensor(patch).unsqueeze(0).float().to(DEVICE)
    test_input = torch.tensor(patch).unsqueeze(0).float().to(DEVICE)

    try:
        explainer = shap.GradientExplainer(model, background)
        shap_vals = explainer.shap_values(test_input, nsamples=10)

        # Normalize to list of (bands, H, W) — one per class
        if isinstance(shap_vals, list):
            # shap >= 0.40
            per_class = [np.abs(sv[0]) for sv in shap_vals]
        else:
            # older shap: ndarray (num_classes, 1, bands, H, W)
            per_class = [np.abs(shap_vals[c, 0])
                         for c in range(shap_vals.shape[0])]

        mean_shap = np.mean(per_class, axis=0)   # (bands, H, W)
        _plot_spatial_importance(
            mean_shap,
            'SHAP Attribution — Mean |value| per Band',
            cfg,
            filename='shap_attribution.png',
            cbar_label='SHAP |value|')

    except Exception as exc:
        print(f"  SHAP failed ({exc}) — using gradient fallback.")
        _run_gradient_importance(model, patch, cfg)

# =============================================================================
#  XAI — main entry point
# =============================================================================

def run_xai(model, data, coords, cfg):
    """
    Lazily extract sample patches, then run SegGradCAM + SHAP.

    Parameters
    ----------
    model  : nn.Module
    data   : (bands, H, W) normalized float32 array
    coords : list of (y, x) patch origins
    cfg    : CONFIG dict
    """
    print("\n[STEP 8] Running XAI analysis ...")

    n_samples = min(cfg['xai_samples'], len(coords))
    p         = cfg['patch_size']
    sample_coords = random.sample(coords, min(n_samples, len(coords)))
    patches = [data[:, y:y+p, x:x+p].copy() for y, x in sample_coords]

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        has_gradcam = True
    except ImportError:
        print("  grad-cam not installed (pip install grad-cam) — GradCAM skipped.")
        has_gradcam = False

    target_layer = _get_target_layer(model) if has_gradcam else None
    if has_gradcam and target_layer is None:
        print("  WARNING: Could not resolve decoder target layer — GradCAM skipped.")

    rgb_idx = cfg['rgb_indices']

    for sample_idx, patch in enumerate(patches):
        # RGB thumbnail
        rgb = patch[rgb_idx].transpose(1, 2, 0).astype('float32')
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
        rgb = np.clip(rgb, 0.0, 1.0)

        # 2x4 grid: cell 0 = RGB, cells 1-6 = one per class, cell 7 hidden
        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('Sentinel-2 RGB', fontsize=12)
        axes[0, 0].axis('off')
        axes[1, 3].axis('off')   # 8th cell unused (1 RGB + 6 classes = 7)

        model.eval()
        for param in model.parameters():
            param.requires_grad_(True)

        if has_gradcam and target_layer is not None:
            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                row = (cls_idx + 1) // 4
                col = (cls_idx + 1) %  4

                # Fresh tensor per class — avoids freed-graph errors
                tensor_cls = (torch.tensor(patch)
                              .unsqueeze(0).float().to(DEVICE))
                tensor_cls.requires_grad_(True)

                dummy_mask = np.ones((p, p), dtype=np.float32)
                targets    = [SemanticSegmentationTarget(cls_idx, dummy_mask)]

                try:
                    with GradCAM(
                        model=model,
                        target_layers=[target_layer],
                        reshape_transform=_seg_reshape_transform,
                    ) as cam:
                        heatmap = cam(input_tensor=tensor_cls,
                                      targets=targets)[0]    # (H, W) float

                    vis = show_cam_on_image(rgb, heatmap, use_rgb=True)
                    axes[row, col].imshow(vis)
                    axes[row, col].set_title(
                        f'GradCAM: {cls_name}',
                        # must be a Python tuple, not a numpy array
                        color=tuple((np.clip(CLASS_COLORS[cls_idx], 30, 255) / 255.0).tolist()),
                        fontsize=11)
                    axes[row, col].axis('off')

                except Exception as exc:
                    print(f"  GradCAM failed for {cls_name}: {exc}")
                    axes[row, col].set_title(f'{cls_name} — failed', fontsize=9)
                    axes[row, col].axis('off')

                finally:
                    del tensor_cls
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            for cls_idx in range(NUM_CLASSES):
                r, c = (cls_idx + 1) // 4, (cls_idx + 1) % 4
                axes[r, c].set_title(CLASS_NAMES[cls_idx], fontsize=9)
                axes[r, c].axis('off')

        fig.suptitle(
            f'SegGradCAM — Patch {sample_idx + 1} ({NUM_CLASSES} classes)',
            fontsize=14)
        plt.tight_layout()
        out = os.path.join(cfg['output_dir'],
                           f'gradcam_patch_{sample_idx + 1}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  GradCAM saved: {out}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # SHAP / gradient importance on first sample patch
    _run_shap(model, patches[0], cfg)

# =============================================================================
#  CITY INFERENCE
# =============================================================================

def city_inference(model, tif_path, cfg):
    """
    Sliding-window inference with overlap-averaged logits
    and median-filter post-processing.
    """
    print("\n[STEP 9] Running city inference ...")

    with rasterio.open(tif_path) as src:
        data      = src.read().astype('float32')
        meta      = src.meta.copy()
        transform = src.transform
        crs       = src.crs

    data = normalize_data(data)
    _, H, W    = data.shape
    patch_size = cfg['patch_size']
    stride     = cfg['stride']
    pred_sum   = np.zeros((NUM_CLASSES, H, W), dtype='float32')
    count_map  = np.zeros((H, W), dtype='float32')
    n_tiles    = 0

    model.eval()
    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch  = data[:, y:y+patch_size, x:x+patch_size]
                tensor = torch.tensor(patch).unsqueeze(0).float().to(DEVICE)
                logits = model(tensor).squeeze(0).cpu().numpy()
                pred_sum[:, y:y+patch_size, x:x+patch_size] += logits
                count_map[y:y+patch_size, x:x+patch_size]   += 1
                n_tiles += 1

    pred_map = (pred_sum / np.maximum(count_map, 1)[np.newaxis]
                ).argmax(0).astype('uint8')
    pred_map = median_filter(pred_map, size=5)

    print(f"  Tiles: {n_tiles} | Output: {H}x{W}")
    return pred_map, meta, transform, crs

# =============================================================================
#  SAVE RESULTS
# =============================================================================

def save_results(pred_map, meta, transform, crs, cfg):
    """Save colorized PNG, georeferenced GeoTIFF, and class distribution chart."""
    print("\n[STEP 10] Saving results ...")

    # Colorized LULC map
    color_map = CLASS_COLORS[pred_map]
    fig, ax   = plt.subplots(figsize=(14, 14))
    ax.imshow(color_map)
    ax.set_title('LULC Map — Sentinel-2 (TorchGeo)', fontsize=16, pad=12)
    ax.axis('off')
    ax.legend(
        handles=[mpatches.Patch(color=CLASS_COLORS[i] / 255.0,
                                label=CLASS_NAMES[i])
                 for i in range(NUM_CLASSES)],
        loc='lower right', fontsize=11, framealpha=0.9)

    unique, counts = np.unique(pred_map, return_counts=True)
    total_px = pred_map.size
    stats = "\n".join(
        f"{CLASS_NAMES[int(u)]}: {100 * c / total_px:.1f}%"
        for u, c in zip(unique, counts) if int(u) < NUM_CLASSES)
    ax.text(0.01, 0.99, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    png_out = os.path.join(cfg['output_dir'], 'city_lulc_map.png')
    plt.savefig(png_out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  PNG saved: {png_out}")

    # Georeferenced GeoTIFF
    tif_meta = {**meta, 'count': 1, 'dtype': 'uint8', 'nodata': 255}
    tif_out  = os.path.join(cfg['output_dir'], 'city_lulc_final.tif')
    with rasterio.open(tif_out, 'w', **tif_meta) as dst:
        dst.write(pred_map.astype('uint8'), 1)
        dst.update_tags(classes=str(CLASS_NAMES),
                        colors=str(CLASS_COLORS.tolist()),
                        description='TorchGeo S2 LULC pipeline')
    print(f"  GeoTIFF saved: {tif_out}")
    print("  Open city_lulc_final.tif in QGIS with a 6-class colour palette.")

    # Class distribution bar chart
    class_pcts = [100 * np.sum(pred_map == i) / pred_map.size
                  for i in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CLASS_NAMES, class_pcts,
                  color=[CLASS_COLORS[i] / 255.0 for i in range(NUM_CLASSES)],
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('LULC Class Distribution', fontsize=13)
    ax.set_ylim(0, max(class_pcts) * 1.15 if class_pcts else 1)
    for bar, pct in zip(bars, class_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    bar_out = os.path.join(cfg['output_dir'], 'class_distribution.png')
    plt.savefig(bar_out, dpi=150)
    plt.close()
    print(f"  Distribution chart saved: {bar_out}")

# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  LULC PIPELINE — TorchGeo Pretrained Sentinel-2 + XAI")
    print("=" * 65)
    print(f"  Device : {DEVICE}")
    print(f"  Classes: {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    if not os.path.exists(CONFIG['city_tif']):
        print(f"\nERROR: {CONFIG['city_tif']} not found.")
        print("  Download Sentinel-2 from GEE first (see GEE script above).")
        return

    # STEP 3 — tile
    data, coords, (H, W), meta, transform, crs = tile_geotiff(
        CONFIG['city_tif'], CONFIG['patch_size'], CONFIG['stride'])

    # Load reference masks (optional)
    masks = load_masks(CONFIG['mask_dir'], coords, CONFIG['patch_size'])

    # STEP 4 — build model
    model = build_model(num_bands=CONFIG['num_bands'], num_classes=NUM_CLASSES)

    # STEP 6 — fine-tune (skipped if no masks)
    model = fine_tune(model, data, coords, masks, CONFIG)

    # STEP 7 — evaluate (skipped if no masks)
    evaluate(model, data, coords, masks, CONFIG)

    # STEP 8 — XAI
    run_xai(model, data, coords, CONFIG)

    # STEP 9 — full city inference
    pred_map, meta, transform, crs = city_inference(
        model, CONFIG['city_tif'], CONFIG)

    # STEP 10 — save outputs
    save_results(pred_map, meta, transform, crs, CONFIG)

    model_path = os.path.join(CONFIG['output_dir'], 'lulc_model_final.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n  Final model: {model_path}")

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Outputs -> {CONFIG['output_dir']}")
    print("    city_lulc_map.png              colorized LULC map")
    print("    city_lulc_final.tif            georeferenced GeoTIFF")
    print("    class_distribution.png         coverage per class")
    print("    gradcam_patch_*.png            GradCAM per class")
    print("    shap_attribution.png           SHAP / gradient spatial map")
    print("    band_importance.png            aggregate band ranking")
    print("    band_importance_per_class.png  per-class band ranking")
    print("    confusion_matrix.png           evaluation (if masks provided)")
    print("=" * 65)


if __name__ == '__main__':
    main()