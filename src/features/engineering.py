"""Feature engineering utilities based on notebook logic."""

import numpy as np
from typing import Dict, Any


def calculate_volume(x: float, y: float, z: float) -> float:
    """Calculate the volume of a diamond."""
    return x * y * z


def calculate_dimension_ratio(x: float, y: float, z: float) -> float:
    """Calculate the dimension ratio."""
    if z <= 0:
        return 0.0
    return (x + y) / (2 * z)


def get_carat_category_encoded(carat: float) -> int:
    """Encode carat category.
    Light (<0.5) -> 0
    Medium (0.5-1.5) -> 1
    Heavy (>1.5) -> 2
    """
    if carat < 0.5:
        return 0
    elif carat <= 1.5:
        return 1
    else:
        return 2


def prepare_features(inputs: Dict[str, Any], mappings: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """Prepare all features for inference based on user inputs.

    Args:
        inputs (Dict[str, Any]): Raw user inputs (carat, cut, color, clarity, depth, table, x, y, z)
        mappings (Dict[str, Dict[str, int]]): Mappings for ordinal encoding.

    Returns:
        Dict[str, Any]: Extracted and engineered features.
    """
    # 1. Base inputs
    carat = inputs["carat"]
    cut_encoded = mappings["cut"][inputs["cut"]]
    color_encoded = mappings["color"][inputs["color"]]
    clarity_encoded = mappings["clarity"][inputs["clarity"]]
    depth = inputs["depth"]
    table = inputs["table"]
    x, y, z = inputs["x"], inputs["y"], inputs["z"]

    # 2. Engineered features
    volume = calculate_volume(x, y, z)
    volume_log = np.log1p(volume)
    carat_log = np.log1p(carat)
    dimension_ratio = calculate_dimension_ratio(x, y, z)
    carat_category_encoded = get_carat_category_encoded(carat)

    return {
        "carat": carat,
        "carat_log": carat_log,
        "cut_encoded": cut_encoded,
        "color_encoded": color_encoded,
        "clarity_encoded": clarity_encoded,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z,
        "volume": volume,
        "volume_log": volume_log,
        "dimension_ratio": dimension_ratio,
        "carat_category_encoded": carat_category_encoded,
    }
