"""File handling utilities."""

import pickle
import os
import sys
from pathlib import Path
from typing import Any
from src.utils.exceptions import ModelLoadError

class CumlDummy:
    @classmethod
    def host_deserialize(cls, header, frames):
        return frames[0] if frames else None

class CumlUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('cuml'):
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

class CumlKMeansWrapper:
    def __init__(self, n_clusters, cluster_centers_):
        self.n_clusters = n_clusters
        self.cluster_centers_ = cluster_centers_
    def predict(self, X):
        import numpy as np
        # Calculate euclidean distance to each center and argmin
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)

def load_pickle(file_path: str) -> Any:
    """Load a pickle file safely, parsing RAPIDS cuml objects back to scikit-learn.
    
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
            model = CumlUnpickler(f).load()
            
            # Post-process clustering model
            if hasattr(model, 'cluster_centers_'):
                n_clust = getattr(model, 'n_clusters', 3)
                centers = getattr(model.cluster_centers_, 'input_value', model.cluster_centers_)
                return CumlKMeansWrapper(n_clust, centers)
                
            # Post-process scaler models
            if hasattr(model, 'mean_') and hasattr(model, 'scale_'):
                import sklearn.preprocessing
                import numpy as np
                sk_model = sklearn.preprocessing.StandardScaler()
                mean_val = getattr(model.mean_, 'input_value', model.mean_)
                scale_val = getattr(model.scale_, 'input_value', model.scale_)
                sk_model.mean_ = np.array(mean_val)
                sk_model.scale_ = np.array(scale_val)
                sk_model.var_ = sk_model.scale_ ** 2
                return sk_model
                
            return model
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
