# LULC Mapping Pipeline — Land Use Land Cover Semantic Segmentation with Explainable AI

A comprehensive deep learning pipeline for **Land Use Land Cover (LULC) semantic segmentation** using Sentinel-2 satellite imagery, with integrated explainable AI (XAI) analysis. This project combines state-of-the-art computer vision models with interpretability techniques to understand which spectral bands and spatial regions drive predictions.

---

## 🌍 Project Overview

**Land Use Land Cover (LULC) mapping** is a fundamental geospatial task that classifies every pixel in satellite imagery into semantic categories (e.g., buildings, water, forests). This project:

- **Leverages Sentinel-2 multispectral data** (6 bands: B2, B3, B4, B8, B11, B12 at 10m resolution)
- **Uses transfer learning** with TorchGeo SSL4EO pretrained ResNet50 encoder
- **Employs UNet++ decoder** for high-quality segmentation
- **Includes fine-tuning** on local reference masks with early stopping
- **Provides interpretability** via SegGradCAM, SHAP, and gradient-based attribution
- **Outputs georeferenced GeoTIFF maps** preserving CRS and spatial alignment

### Use Cases
- Urban planning and building footprint extraction
- Water resource monitoring
- Forest conservation and land degradation assessment
- Agricultural land classification
- Environmental impact analysis

---

## 📊 Dataset

### Input Data
- **Source**: Copernicus Sentinel-2 Level-2A (Surface Reflectance)
- **Bands**: 6 selected (Blue, Green, Red, NIR, SWIR1, SWIR2)
- **Resolution**: 10m/pixel
- **Temporal**: Configurable date range (e.g., 2024-01-01 to 2024-12-31)
- **Preprocessing**: Median composite to reduce cloud cover

### Reference Masks (Optional)
- Ground-truth annotations in GeoTIFF format
- Expected to be in the `mask_dir` folder, one file per patch
- Used for fine-tuning and model evaluation
- Automatically remapped to align with training classes

### Classes (6-class schema)
| Class | RGB Color | Meaning |
|-------|-----------|---------|
| 0 | Black (0,0,0) | Background / No Data |
| 1 | Red (255,0,0) | Building / Urban |
| 2 | Blue (0,0,255) | Water / Aquatic |
| 3 | Brown (165,42,42) | Barren / Exposed Soil |
| 4 | Green (0,128,0) | Forest / Dense Vegetation |
| 5 | Yellow (255,255,0) | Agriculture / Cropland |

---

## 🏗️ Project Structure

```
minor project/
├── final.py                    # Main training pipeline
├── inference.py                # Inference for new AOIs
├── final.ipynb                 # Jupyter notebook version (interactive)
├── lulc_torchgeo_pipeline.ipynb # Alternative pipeline notebook
├── README.md                   # This file
├── lulc_results/               # Output directory (training run 1)
│   └── best_model.pth          # Trained model checkpoint
├── lulc_results12/             # Output directory (training run 2)
│   ├── best_model.pth
│   └── lulc_model_final.pth
├── lulc_results16/             # Output directory (training run 3)
└── presentable/
    └── show/                   # Presentation-ready outputs
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Core dependencies
pip install torch torchvision torchgeo segmentation-models-pytorch

# Data & visualization
pip install rasterio matplotlib seaborn tqdm scipy scikit-learn albumentations

# Explainable AI
pip install grad-cam shap

# Metrics
pip install torchmetrics
```

**Note**: Tested with PyTorch 2.0+, CUDA 11.8+ (optional but recommended for GPU acceleration)

### 2. Prepare Your Data

#### Option A: Download Sentinel-2 from Google Earth Engine
1. Visit [Google Earth Engine Code Editor](https://code.earthengine.google.com)
2. Copy the GEE script from the **GEE SCRIPT** section in `final.py`
3. Replace coordinates with your area of interest (AOI)
4. Export as GeoTIFF to Google Drive
5. Download to your local machine

#### Option B: Use Pre-existing GeoTIFF
If you already have a 6-band Sentinel-2 composite, place it at the path specified in CONFIG.

### 3. Configure the Pipeline

Edit the `CONFIG` dictionary at the top of `final.py`:

```python
CONFIG = {
    "city_tif"           : r"C:\path\to\your\6bands.tif",      # Input Sentinel-2 GeoTIFF
    "output_dir"         : "lulc_results/",                     # Output directory
    "patch_size"         : 256,                                  # Patch size (pixels)
    "stride"             : 128,                                  # Overlap stride
    "batch_size"         : 2,                                    # Training batch size
    "epochs"             : 5,                                    # Number of epochs
    "lr"                 : 1e-4,                                 # Learning rate
    "early_stop_patience": 2,                                    # Early stopping patience
    "num_bands"          : 6,                                    # Always 6 for Sentinel-2
    "rgb_indices"        : [2, 1, 0],                            # Band indices for RGB viz
    "mask_dir"           : r"c:\path\to\reference_masks",       # Reference masks (optional)
    "xai_samples"        : 5,                                    # Number of patches for XAI
}
```

### 4. Run Training

```bash
python final.py
```

**Pipeline steps**:
1. ✅ Load & normalize Sentinel-2 data
2. ✅ Generate patches (tiling)
3. ✅ Build model (UNet++ with pretrained ResNet50)
4. ✅ Fine-tune on reference masks (if provided)
5. ✅ Evaluate (IoU, F1 per class)
6. ✅ Explainability analysis (SegGradCAM, SHAP, gradient attribution)
7. ✅ City-wide inference (sliding window with overlap averaging)
8. ✅ Post-processing (median filter for noise removal)
9. ✅ Save georeferenced outputs

### 5. Run Inference on New AOI

For a new area of interest without retraining:

```bash
# Edit CONFIG in inference.py
CONFIG = {
    "model_path"   : r"C:\path\to\trained\best_model.pth",
    "input_tif"    : r"C:\path\to\new_city.tif",
    "output_dir"   : "inference_results/",
    ...
}

python inference.py
```

**Key difference**: Inference performs **per-AOI normalization**—statistics are computed from the new image's own pixels, ensuring consistent results regardless of absolute reflectance values.

---

## 📈 Model Architecture

### Encoder: ResNet50 (TorchGeo SSL4EO)
- Pretrained on 1M Sentinel-2 patches via self-supervised learning
- Frozen batch norm for stable fine-tuning
- 6-band input → 3-band conv1 adapted via interpolation or weight averaging
- Progressive skip connections (C1, C2, C3, C4, C5)

### Decoder: UNet++
- Multi-scale dense skip connections for feature refinement
- Reduces checkerboard artifacts common in standard U-Net
- Output: (B, 6, H, W) logits for 6 LULC classes

### Loss Function
```
Combined Loss = 0.5 × Cross-Entropy + 0.5 × Dice Loss
```
Balanced weighting ensures both pixel-level accuracy and class-level stability.

---

## 📊 Evaluation Metrics

The pipeline reports per-class metrics:

- **Intersection-over-Union (IoU)**: $\text{IoU} = \frac{\text{TP}}{\text{TP + FP + FN}}$
- **F1-Score**: $F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}}$
- **Mean IoU (mIoU)**: Average across all classes
- **Confusion Matrix**: Pixel-level classification breakdown

Example output:
```
  Class           IoU        F1
  ───────────────────────────────
  background     0.8934     0.9412
  building       0.7823     0.8675
  water          0.9201     0.9543
  barren         0.6543     0.7234
  forest         0.8876     0.9123
  agriculture    0.8234     0.8901

  mIoU: 0.8301  |  Mean F1: 0.8481
```

---

## 🧠 Explainable AI (XAI)

Understanding *why* the model makes predictions is critical for geospatial applications. This pipeline provides three XAI methods:

### 1. SegGradCAM (Spatial Attribution)
- Visualizes which spatial regions are important for each class
- Generates heatmap overlays on RGB imagery
- Helps identify false predictions

### 2. SHAP (Gradient Explainer)
- Per-band importance via gradient-based SHAP
- Quantifies which spectral bands drive predictions
- Fallback to gradient×input if SHAP fails

### 3. Gradient × Input (Universal Fallback)
- Perturb-based attribution without external libraries
- Band importance histogram + spatial heatmap
- Per-class breakdown of band importance

**Output visualizations**:
- `shap_attribution.png` — Spatial importance heatmap
- `band_importance.png` — Bar chart: which bands matter most
- `band_importance_per_class.png` — Per-class band importance
- `gradcam_<sample>.png` — Class activation maps

---

## 💾 Output Files

### Training Pipeline Outputs
| File | Description |
|------|-------------|
| `best_model.pth` | Best model checkpoint (lowest validation loss) |
| `lulc_map.tif` | Georeferenced prediction map (GeoTIFF) |
| `lulc_map_color.png` | Colorized LULC map with legend |
| `class_distribution.png` | Bar chart: % coverage per class |
| `confusion_matrix.png` | Pixel-level confusion matrix |
| `shap_attribution.png` | SHAP spatial attribution heatmap |
| `band_importance.png` | Per-band importance bar chart |
| `band_importance_per_class.png` | Per-class band importance breakdown |
| `gradcam_sample_*.png` | Grad-CAM visualizations (one per XAI sample) |

### Inference Pipeline Outputs
| File | Description |
|------|-------------|
| `{stem}_lulc.tif` | Georeferenced prediction (preserves CRS + transform) |
| `{stem}_lulc_color.png` | Colorized LULC with coverage statistics |
| `{stem}_distribution.png` | Class distribution bar chart |
| `{stem}_debug_rgb.png` | Normalized RGB preview of input |

All files are saved to `output_dir` specified in CONFIG.

---

## 🔧 Advanced Configuration

### Hyperparameters
```python
"patch_size"        : 256         # Larger = more context but more memory
"stride"            : 128         # Overlap = patch_size - stride
"batch_size"        : 2           # Increase if VRAM allows
"epochs"            : 5           # More epochs = longer training
"lr"                : 1e-4        # Lower for fine-tuning
"early_stop_patience": 2          # Stop if val loss doesn't improve
```

### Model Selection
To use a different encoder:
```python
model = smp.UnetPlusPlus(
    encoder_name='resnet101',      # Try: resnet18, resnet50, resnet101, etc.
    encoder_weights='imagenet',    # Or 'ssl4eo' if available
    in_channels=6,
    classes=6,
)
```

### Custom Data Normalization
Edit `normalize_data()` to use custom statistics:
```python
S2_MEAN = [your_band_means]        # Per-band statistics
S2_STD  = [your_band_stds]
```

---

## 📋 Requirements

### Hardware
- **GPU** (recommended): NVIDIA GPU with 8GB+ VRAM (tested on RTX 3060, 4060)
- **CPU** (possible but slow): Multi-core processor for data loading

### Software
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional)
- See [Installation](#1-installation) for full dependency list

### Data
- Min 1 Sentinel-2 GeoTIFF (6 bands, any size)
- Optional: Reference mask GeoTIFFs for fine-tuning
- 2-3 GB disk space for a typical project

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA out of memory** | Reduce `batch_size`, `patch_size`, or enable CPU-only mode |
| **Model not improving** | Reduce `lr` or increase `epochs`; check reference masks for quality |
| **XAI fails / imports missing** | Fall back to gradient×input; install `pip install grad-cam shap` |
| **GeoTIFF missing CRS** | Ensure input GeoTIFF has valid CRS; rasterio will preserve it |
| **Inference results look wrong** | Verify per-AOI normalization—check `normalize_per_aoi()` statistics |
| **SHAP taking too long** | Reduce `xai_samples`; gradient×input fallback is faster |

---

## 📚 Key Design Decisions

1. **Per-AOI Normalization** (Inference)
   - Statistics computed from the *new* image's own pixels, not training data
   - Ensures consistent results across cities with different reflectance ranges

2. **Overlap Averaging**
   - Predictions averaged across overlapping patches during inference
   - Reduces edge artifacts and improves spatial smoothness

3. **Median Filter Post-Processing**
   - Removes salt-and-pepper noise (isolated misclassified pixels)
   - Kernel size: 5×5 (adjustable)

4. **Lazy Patch Loading**
   - Patches extracted on-the-fly during training/inference
   - Reduces memory footprint; enables large-scale processing

5. **Early Stopping**
   - Monitor validation loss with configurable patience
   - Prevents overfitting to reference masks

6. **Weights-Only Loading** (`weights_only=True`)
   - PyTorch best practice for secure model deserialization
   - Prevents arbitrary code execution from untrusted checkpoints

---

## 📖 References

- **TorchGeo**: [pytorch.org/vision/main/io.html](https://pytorch.org/vision/main/io.html)
- **SSL4EO-S12**: Self-supervised learning on Sentinel-2
- **segmentation-models-pytorch**: [smp.readthedocs.io](https://smp.readthedocs.io)
- **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- **SHAP**: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions"

---

## 📝 Usage Examples

### Example 1: Train on Aligarh with Reference Masks
```python
CONFIG = {
    "city_tif"     : r"C:\LULC\6bands.tif",
    "mask_dir"     : r"C:\LULC\aligarh_masks",
    "output_dir"   : "lulc_results_aligarh/",
    "epochs"       : 10,
    "batch_size"   : 4,
}
```

### Example 2: Inference on New City (No Fine-tuning)
```python
# inference.py CONFIG
"model_path"  : r"C:\lulc_results\best_model.pth",
"input_tif"   : r"C:\Downloads\NewCity_S2.tif",
"output_dir"  : "inference_results_newcity/",
```

### Example 3: Export Results for QGIS
```bash
# All outputs (.tif, .png) can be directly imported into QGIS
# GeoTIFF preserves CRS → automatic spatial alignment
```

---

## 🤝 Contributing

Issues, improvements, and alternative architectures welcome! 
- Test on your own AOI and share results
- Propose XAI enhancements
- Optimize for different Sentinel-2 bands

---

## 👤 Author

**Syed Basil Rizvi**  
*Land Use Land Cover Semantic Segmentation + Explainable AI*

---

## 📄 License

This project is provided as-is for educational and research purposes. Sentinel-2 data is publicly available under the [Copernicus Program](https://www.copernicus.eu/).

---

## ✨ Highlights

- ✅ Transfer learning with TorchGeo SSL4EO weights
- ✅ Fine-tuning on local reference masks
- ✅ Multi-scale UNet++ decoder
- ✅ 3 interpretability methods (SegGradCAM, SHAP, Gradient×Input)
- ✅ Georeferenced outputs (CRS + transform preserved)
- ✅ Per-AOI normalization for robust cross-city inference
- ✅ Overlap-averaged sliding window inference
- ✅ Comprehensive visualization suite

---

**Happy mapping! 🗺️**
