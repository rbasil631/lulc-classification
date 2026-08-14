"""LULC semantic segmentation on Sentinel-2 imagery with explainable AI."""

from lulc.constants import (
    BAND_IDS,
    BAND_NAMES,
    CLASS_COLORS,
    CLASS_NAMES,
    NUM_BANDS,
    NUM_CLASSES,
)

__version__ = "2.0.0"

__all__ = [
    "BAND_IDS",
    "BAND_NAMES",
    "CLASS_COLORS",
    "CLASS_NAMES",
    "NUM_BANDS",
    "NUM_CLASSES",
    "__version__",
]
