"""Inference logic."""

import os
import numpy as np
from typing import Dict, Any, Tuple
from src.utils.logger import get_logger
from src.utils.file_handler import load_pickle
from src.utils.config_loader import ConfigManager
from src.features.engineering import prepare_features

logger = get_logger(__name__)


class Predictor:
    """Predictor for diamond price and segments."""

    def __init__(self, config_manager: ConfigManager):
        """Initialize the predictor and load models."""
        self.config = config_manager
        self._load_models()

    def _load_models(self) -> None:
        """Load ML models and scalers from disk."""
        models_dir = self.config.get_path("models_dir")

        reg_model_file = os.path.join(models_dir, self.config.get_model_config("regression_model"))
        reg_scaler_file = os.path.join(
            models_dir, self.config.get_model_config("regression_scaler")
        )
        cls_model_file = os.path.join(models_dir, self.config.get_model_config("clustering_model"))
        cls_scaler_file = os.path.join(
            models_dir, self.config.get_model_config("clustering_scaler")
        )
        cls_names_file = os.path.join(models_dir, self.config.get_model_config("cluster_names"))

        logger.info("Loading regression model...")
        self.reg_model = load_pickle(reg_model_file)
        self.reg_scaler = load_pickle(reg_scaler_file)

        logger.info("Loading clustering model...")
        self.cls_model = load_pickle(cls_model_file)
        self.cls_scaler = load_pickle(cls_scaler_file)
        self.cls_names = load_pickle(cls_names_file)

        self.features_reg = self.config.get_model_config("features")["regression"]
        self.features_cls = self.config.get_model_config("features")["clustering"]
        self.mappings = self.config.get_model_config("mappings")

    def predict(self, inputs: Dict[str, Any]) -> Tuple[float, float, str, int]:
        """Run prediction.

        Args:
            inputs: Dictionary containing raw user inputs.

        Returns:
            Tuple of (price_inr, price_usd, cluster_name, cluster_id)
        """
        logger.info(f"Running prediction for inputs: {inputs}")
        features = prepare_features(inputs, self.mappings)

        # 1. Regression
        reg_features_array = np.array([[features[f] for f in self.features_reg]])
        reg_scaled = self.reg_scaler.transform(reg_features_array)
        pred_price_log = self.reg_model.predict(reg_scaled)[0]

        price_inr = float(np.expm1(pred_price_log))
        conversion_rate = self.config.config.get("conversion_rate", 83.0)
        price_usd = price_inr / conversion_rate

        # 2. Clustering
        cls_features_array = np.array([[features[f] for f in self.features_cls]])
        cls_scaled = self.cls_scaler.transform(cls_features_array)
        cluster_id = int(self.cls_model.predict(cls_scaled)[0])
        cluster_name = self.cls_names.get(cluster_id, f"Cluster {cluster_id}")

        logger.info(f"Prediction successful: {price_inr} INR, {cluster_name}")
        return price_inr, price_usd, cluster_name, cluster_id
