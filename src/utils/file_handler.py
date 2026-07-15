"""File handling utilities."""

import pickle
import os
import sys
from pathlib import Path
from typing import Any
import sklearn.preprocessing
import sklearn.cluster
import sklearn.ensemble
from src.utils.exceptions import ModelLoadError

class CumlUnpickler(pickle.Unpickler):
    """Custom unpickler to map cuml classes to scikit-learn equivalents."""
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith('cuml'):
            if 'StandardScaler' in name or 'Scaler' in name:
                return sklearn.preprocessing.StandardScaler
            if 'KMeans' in name:
                return sklearn.cluster.KMeans
            if 'RandomForest' in name:
                return sklearn.ensemble.RandomForestRegressor
            
            # For any other cuml internal classes, provide a dummy class
            class CumlDummy:
                pass
            return CumlDummy
            
        if module.startswith('cupy'):
            import numpy
            if 'multiarray' in module or 'core' in module:
                try:
                    import numpy.core.multiarray as multiarray
                    return getattr(multiarray, name, numpy.ndarray)
                except ImportError:
                    return numpy.ndarray
            return getattr(numpy, name, numpy.ndarray)
            
        return super().find_class(module, name)

def load_pickle(file_path: str) -> Any:
    """Load a pickle file safely, handling cuml models via fallback.
    
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
            return CumlUnpickler(f).load()
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
