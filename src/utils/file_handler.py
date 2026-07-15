"""File handling utilities."""

import pickle
import os
import sys
from pathlib import Path
from typing import Any
from src.utils.exceptions import ModelLoadError

# --- MOCK CUML FOR GPU-TRAINED MODELS ---
# The models were trained using RAPIDS (cuml) on Google Colab.
# Since cuml requires specific NVIDIA environments, we mock it 
# to fallback to the equivalent scikit-learn classes which share the same API.
try:
    import cuml
except ImportError:
    import sklearn.preprocessing
    import sklearn.cluster
    import sklearn.ensemble
    
    class CumlMock:
        pass
        
    cuml_mock = CumlMock()
    sys.modules['cuml'] = cuml_mock
    sys.modules['cuml.preprocessing'] = sklearn.preprocessing
    sys.modules['cuml.cluster'] = sklearn.cluster
    sys.modules['cuml.ensemble'] = sklearn.ensemble
# ----------------------------------------


def load_pickle(file_path: str) -> Any:
    """Load a pickle file safely.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        Any: Unpickled data.

    Raises:
        ModelLoadError: If the file cannot be found or loaded.
    """
    if not os.path.exists(file_path):
        raise ModelLoadError(f"File not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise ModelLoadError(f"Failed to load pickle file {file_path}: {e}")


def save_pickle(data: Any, file_path: str) -> None:
    """Save data to a pickle file safely.

    Args:
        data (Any): Data to pickle.
        file_path (str): Path to save the pickle file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f)
