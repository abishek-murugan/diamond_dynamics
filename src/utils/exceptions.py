"""Custom exceptions for the application."""


class DiamondDynamicsError(Exception):
    """Base exception for Diamond Dynamics."""

    pass


class ConfigurationError(DiamondDynamicsError):
    """Exception raised for configuration errors."""

    pass


class ModelLoadError(DiamondDynamicsError):
    """Exception raised when a model fails to load."""

    pass


class ValidationError(DiamondDynamicsError):
    """Exception raised for data validation errors."""

    pass


class PredictionError(DiamondDynamicsError):
    """Exception raised during prediction."""

    pass
