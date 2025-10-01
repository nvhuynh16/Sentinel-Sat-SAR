"""
Tests for SAR preprocessing functions.
"""
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing import (
    convert_to_db, convert_from_db, apply_speckle_filter,
    lee_filter, enhance_contrast, normalize_intensity
)


class TestPreprocessing:
    """Test suite for SAR preprocessing functions."""

    @pytest.fixture
    def sample_intensity_image(self):
        """Create a sample SAR intensity image."""
        np.random.seed(42)
        # Create synthetic SAR data with speckle
        img = np.random.gamma(2.0, 100, size=(100, 100))
        return img

    @pytest.fixture
    def sample_db_image(self):
        """Create a sample SAR image in dB scale."""
        np.random.seed(42)
        img = np.random.normal(10, 5, size=(100, 100))
        return img

    def test_convert_to_db(self, sample_intensity_image):
        """Test intensity to dB conversion."""
        db_image = convert_to_db(sample_intensity_image)

        # Check that conversion produces reasonable dB values
        assert isinstance(db_image, np.ndarray)
        assert db_image.shape == sample_intensity_image.shape

        # dB values should typically be between -50 and 50 for SAR
        assert db_image.min() > -100, "dB values seem too low"
        assert db_image.max() < 100, "dB values seem too high"

    def test_convert_to_db_handles_zeros(self):
        """Test that conversion handles zero values without error."""
        img_with_zeros = np.array([[0, 1, 100], [0, 0, 50]])
        db_image = convert_to_db(img_with_zeros)

        # Should not have NaN or Inf values
        assert not np.any(np.isnan(db_image))
        assert not np.any(np.isinf(db_image))

    def test_convert_from_db(self, sample_db_image):
        """Test dB to intensity conversion."""
        intensity = convert_from_db(sample_db_image)

        assert isinstance(intensity, np.ndarray)
        assert intensity.shape == sample_db_image.shape
        assert np.all(intensity >= 0), "Intensity should be non-negative"

    def test_db_conversion_roundtrip(self, sample_intensity_image):
        """Test that converting to dB and back preserves values."""
        db = convert_to_db(sample_intensity_image)
        recovered = convert_from_db(db)

        # Should be very close (within floating point precision)
        np.testing.assert_allclose(recovered, sample_intensity_image, rtol=1e-5)

    def test_median_filter(self, sample_intensity_image):
        """Test median speckle filter."""
        filtered = apply_speckle_filter(sample_intensity_image, method='median', size=3)

        assert filtered.shape == sample_intensity_image.shape
        # Median filter should reduce variance (despeckle)
        assert np.var(filtered) < np.var(sample_intensity_image)

    def test_mean_filter(self, sample_intensity_image):
        """Test mean speckle filter."""
        filtered = apply_speckle_filter(sample_intensity_image, method='mean', size=3)

        assert filtered.shape == sample_intensity_image.shape
        # Mean filter should also reduce variance
        assert np.var(filtered) < np.var(sample_intensity_image)

    def test_speckle_filter_invalid_method(self, sample_intensity_image):
        """Test error handling for invalid filter method."""
        with pytest.raises(ValueError):
            apply_speckle_filter(sample_intensity_image, method='invalid')

    def test_lee_filter(self, sample_intensity_image):
        """Test Lee filter for speckle reduction."""
        filtered = lee_filter(sample_intensity_image, window_size=5)

        assert filtered.shape == sample_intensity_image.shape
        # Lee filter should reduce variance
        assert np.var(filtered) < np.var(sample_intensity_image)
        # Should not introduce NaN values
        assert not np.any(np.isnan(filtered))

    def test_enhance_contrast_equalize(self, sample_db_image):
        """Test histogram equalization."""
        enhanced = enhance_contrast(sample_db_image, method='equalize')

        assert enhanced.shape == sample_db_image.shape
        # Enhanced image should have better distributed values
        assert np.ptp(enhanced) > 0  # Range should be non-zero

    def test_enhance_contrast_stretch(self, sample_db_image):
        """Test contrast stretching."""
        enhanced = enhance_contrast(sample_db_image, method='stretch')

        assert enhanced.shape == sample_db_image.shape

    def test_normalize_intensity(self, sample_intensity_image):
        """Test intensity normalization."""
        normalized = normalize_intensity(sample_intensity_image)

        assert normalized.shape == sample_intensity_image.shape
        assert normalized.min() >= 0, "Should be >= 0"
        assert normalized.max() <= 1, "Should be <= 1"
        assert normalized.dtype == np.float32

    def test_normalize_constant_image(self):
        """Test normalization of constant-valued image."""
        constant_img = np.ones((50, 50)) * 42
        normalized = normalize_intensity(constant_img)

        # Should return zeros for constant image
        assert np.all(normalized == 0)
