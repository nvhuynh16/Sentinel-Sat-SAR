"""
Tests for change detection functions.
"""
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.change_detection import (
    compute_difference, compute_log_ratio, detect_changes,
    compute_change_magnitude, refine_change_map, detect_bright_spots,
    filter_detections_by_size, categorize_changes
)


class TestChangeDetection:
    """Test suite for change detection functions."""

    @pytest.fixture
    def sample_images(self):
        """Create sample before/after images."""
        np.random.seed(42)
        # Create two similar images with some changes
        before = np.random.rand(100, 100) * 50 + 100
        after = before.copy()
        # Add some changes
        after[20:30, 20:30] += 20  # Increase
        after[60:70, 60:70] -= 15  # Decrease
        return before, after

    def test_compute_difference(self, sample_images):
        """Test simple difference calculation."""
        before, after = sample_images
        diff = compute_difference(before, after)

        assert diff.shape == before.shape
        assert isinstance(diff, np.ndarray)

        # Check that the changed region has positive difference
        assert np.mean(diff[20:30, 20:30]) > 0

    def test_compute_difference_shape_mismatch(self):
        """Test error handling for mismatched shapes."""
        img1 = np.random.rand(100, 100)
        img2 = np.random.rand(50, 50)

        with pytest.raises(ValueError):
            compute_difference(img1, img2)

    def test_compute_log_ratio(self, sample_images):
        """Test log-ratio change detection."""
        before, after = sample_images
        log_ratio = compute_log_ratio(before, after)

        assert log_ratio.shape == before.shape
        assert not np.any(np.isnan(log_ratio))
        assert not np.any(np.isinf(log_ratio))

    def test_compute_log_ratio_handles_zeros(self):
        """Test log-ratio with zero values."""
        before = np.array([[0, 1, 10], [0, 5, 20]])
        after = np.array([[1, 2, 15], [0, 0, 25]])

        log_ratio = compute_log_ratio(before, after)

        # Should not have NaN or Inf
        assert not np.any(np.isnan(log_ratio))
        assert not np.any(np.isinf(log_ratio))

    def test_detect_changes_difference(self, sample_images):
        """Test change detection using difference method."""
        before, after = sample_images
        changes = detect_changes(before, after, threshold=10, method='difference')

        assert changes.shape == before.shape
        assert changes.dtype == np.uint8
        assert set(np.unique(changes)) <= {0, 1}

        # Should detect changes in the modified regions
        assert np.any(changes[20:30, 20:30] == 1)

    def test_detect_changes_log_ratio(self, sample_images):
        """Test change detection using log-ratio method."""
        before, after = sample_images
        changes = detect_changes(before, after, threshold=2, method='log_ratio')

        assert changes.shape == before.shape
        assert changes.dtype == np.uint8

    def test_detect_changes_invalid_method(self, sample_images):
        """Test error handling for invalid method."""
        before, after = sample_images

        with pytest.raises(ValueError):
            detect_changes(before, after, method='invalid')

    def test_compute_change_magnitude(self, sample_images):
        """Test change magnitude calculation."""
        before, after = sample_images
        magnitude = compute_change_magnitude(before, after)

        assert magnitude.shape == before.shape
        assert np.all(magnitude >= 0), "Magnitude should be non-negative"

        # Magnitude in changed region should be higher
        changed_mag = np.mean(magnitude[20:30, 20:30])
        unchanged_mag = np.mean(magnitude[40:50, 40:50])
        assert changed_mag > unchanged_mag

    def test_refine_change_map(self):
        """Test change map refinement."""
        # Create a change map with small isolated regions
        change_map = np.zeros((100, 100), dtype=np.uint8)
        change_map[10:20, 10:20] = 1  # Large region (100 pixels)
        change_map[50, 50] = 1  # Single pixel
        change_map[80:82, 80:82] = 1  # Small region (4 pixels)

        refined = refine_change_map(change_map, min_size=10)

        # Large region should remain
        assert np.any(refined[10:20, 10:20] == 1)
        # Small regions should be removed
        assert refined[50, 50] == 0
        assert np.sum(refined[80:82, 80:82]) == 0

    def test_detect_bright_spots(self):
        """Test bright spot detection."""
        # Create image with bright spots
        image = np.random.rand(100, 100) * 10
        image[30:35, 30:35] = 50  # Bright spot 1
        image[70:73, 70:73] = 45  # Bright spot 2

        detections = detect_bright_spots(image, percentile=98)

        assert isinstance(detections, list)
        assert len(detections) >= 2

        # Check detection properties
        for det in detections:
            assert 'centroid' in det
            assert 'area' in det
            assert 'intensity_mean' in det
            assert 'bbox' in det

    def test_detect_bright_spots_with_bbox(self):
        """Test bright spot detection within bounding box."""
        image = np.random.rand(100, 100) * 10
        image[30:35, 30:35] = 50

        # Search only in the region containing the bright spot
        bbox = (20, 20, 50, 50)
        detections = detect_bright_spots(image, bbox=bbox, percentile=95)

        assert len(detections) >= 1

        # Centroid should be within the bbox
        centroid = detections[0]['centroid']
        assert bbox[0] <= centroid[0] <= bbox[2]
        assert bbox[1] <= centroid[1] <= bbox[3]

    def test_filter_detections_by_size(self):
        """Test filtering detections by area."""
        detections = [
            {'area': 5, 'intensity_max': 100},
            {'area': 50, 'intensity_max': 90},
            {'area': 500, 'intensity_max': 80},
            {'area': 2000, 'intensity_max': 70},
        ]

        filtered = filter_detections_by_size(detections, min_area=10, max_area=1000)

        assert len(filtered) == 2
        assert all(10 <= d['area'] <= 1000 for d in filtered)

    def test_categorize_changes(self):
        """Test categorizing changes into increase/decrease."""
        change_map = np.array([
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
        ], dtype=np.uint8)

        magnitude = np.array([
            [0, 0, 10, -5],
            [0, 15, -8, 0],
            [12, -3, 0, 0],
        ])

        categories = categorize_changes(change_map, magnitude)

        assert 'increase' in categories
        assert 'decrease' in categories
        assert 'no_change' in categories

        # Check increase mask
        assert categories['increase'][0, 2] == 1  # magnitude=10, change=1
        assert categories['increase'][0, 3] == 0  # magnitude=-5 (decrease)

        # Check decrease mask
        assert categories['decrease'][0, 3] == 1  # magnitude=-5, change=1
        assert categories['decrease'][0, 2] == 0  # magnitude=10 (increase)

    def test_no_changes_detected(self):
        """Test case where images are identical (no changes)."""
        image = np.random.rand(50, 50) * 100
        changes = detect_changes(image, image, threshold=1.0)

        # Should detect no changes (or very few due to numerical precision)
        assert np.sum(changes) < 10
