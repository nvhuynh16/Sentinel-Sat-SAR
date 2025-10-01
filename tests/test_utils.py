"""
Tests for SAR data utility functions.
"""
import pytest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_sar_band, get_safe_info, extract_sentinel_product


class TestSARDataLoading:
    """Test suite for SAR data loading utilities."""

    @pytest.fixture
    def safe_dir(self):
        """Fixture providing path to extracted SAFE directory."""
        return Path("data/sentinel1/S1A_IW_GRDH_1SDV_20250816T025212_20250816T025237_060555_078800_1619.SAFE")

    def test_load_sar_band_vv(self, safe_dir):
        """Test loading VV polarization band."""
        if not safe_dir.exists():
            pytest.skip("Test data not available")

        data, metadata = load_sar_band(safe_dir, polarization='VV')

        # Check data type and shape
        assert isinstance(data, np.ndarray), "Data should be numpy array"
        assert data.ndim == 2, "SAR data should be 2D"
        assert data.shape[0] > 1000, "Image height should be reasonable"
        assert data.shape[1] > 1000, "Image width should be reasonable"

        # Check metadata
        assert 'shape' in metadata
        assert 'polarization' in metadata
        assert metadata['polarization'] == 'VV'
        assert 'filename' in metadata

    def test_load_sar_band_vh(self, safe_dir):
        """Test loading VH polarization band."""
        if not safe_dir.exists():
            pytest.skip("Test data not available")

        data, metadata = load_sar_band(safe_dir, polarization='VH')

        assert isinstance(data, np.ndarray)
        assert metadata['polarization'] == 'VH'

    def test_load_sar_band_invalid_polarization(self, safe_dir):
        """Test error handling for invalid polarization."""
        if not safe_dir.exists():
            pytest.skip("Test data not available")

        with pytest.raises(FileNotFoundError):
            load_sar_band(safe_dir, polarization='HH')

    def test_load_sar_band_nonexistent_dir(self):
        """Test error handling for non-existent SAFE directory."""
        fake_dir = Path("nonexistent/directory.SAFE")

        with pytest.raises(FileNotFoundError):
            load_sar_band(fake_dir, polarization='VV')

    def test_get_safe_info(self, safe_dir):
        """Test extracting information from SAFE directory."""
        if not safe_dir.exists():
            pytest.skip("Test data not available")

        info = get_safe_info(safe_dir)

        assert 'satellite' in info
        assert info['satellite'] in ['S1A', 'S1B', 'S1C']
        assert 'product_type' in info
        assert info['product_type'] == 'GRDH'
        assert 'polarizations' in info
        assert 'VV' in info['polarizations']

    def test_data_range(self, safe_dir):
        """Test that SAR data has expected value range."""
        if not safe_dir.exists():
            pytest.skip("Test data not available")

        data, _ = load_sar_band(safe_dir, polarization='VV')

        # SAR intensity data should be non-negative
        assert data.min() >= 0, "SAR intensity should be non-negative"
        assert data.max() > 0, "SAR data should have some non-zero values"

        # Check for reasonable dynamic range
        assert data.max() > 100, "Maximum intensity seems too low"
