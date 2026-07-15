"""Tests for feature engineering."""

from src.features.engineering import (
    calculate_volume,
    calculate_dimension_ratio,
    get_carat_category_encoded,
)


def test_calculate_volume():
    """Test volume calculation."""
    assert calculate_volume(2, 3, 4) == 24.0
    assert calculate_volume(0, 0, 0) == 0.0


def test_calculate_dimension_ratio():
    """Test dimension ratio calculation."""
    assert calculate_dimension_ratio(5.0, 5.0, 2.5) == 2.0
    assert calculate_dimension_ratio(5.0, 5.0, 0.0) == 0.0


def test_get_carat_category_encoded():
    """Test carat category encoding."""
    assert get_carat_category_encoded(0.4) == 0
    assert get_carat_category_encoded(1.0) == 1
    assert get_carat_category_encoded(2.0) == 2
