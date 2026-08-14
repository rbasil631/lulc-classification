"""Class schema and band definitions shared across the pipeline."""

import numpy as np

CLASS_NAMES = [
    "background",
    "building",
    "water",
    "barren",
    "forest",
    "agriculture",
]

CLASS_COLORS = np.array(
    [
        [0, 0, 0],        # background   - black
        [255, 0, 0],      # building     - red
        [0, 0, 255],      # water        - blue
        [165, 42, 42],    # barren       - brown
        [0, 128, 0],      # forest       - green
        [255, 255, 0],    # agriculture  - yellow
    ],
    dtype="uint8",
)

NUM_CLASSES = len(CLASS_NAMES)

# Sentinel-2 L2A bands used, in the order they appear in the input GeoTIFF.
BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
BAND_IDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
BAND_COLORS = ["#4472C4", "#70AD47", "#FF0000", "#7030A0", "#C55A11", "#843C0C"]

NUM_BANDS = len(BAND_NAMES)

# 0-based indices into the band axis that form a true-colour composite.
RGB_INDICES = [2, 1, 0]
