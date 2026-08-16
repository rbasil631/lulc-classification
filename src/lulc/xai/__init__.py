"""Explainability: SegGradCAM, band attribution and measured spectra."""

from lulc.xai.attribution import (
    gradient_band_importance,
    integrated_gradients_band_importance,
    occlusion_band_importance,
    run_band_importance,
    select_class_patches,
)
from lulc.xai.gradcam import run_gradcam
from lulc.xai.spectra import band_separability, class_spectra, run_spectral_analysis

__all__ = [
    "band_separability",
    "class_spectra",
    "gradient_band_importance",
    "integrated_gradients_band_importance",
    "occlusion_band_importance",
    "run_band_importance",
    "run_gradcam",
    "run_spectral_analysis",
    "select_class_patches",
]
