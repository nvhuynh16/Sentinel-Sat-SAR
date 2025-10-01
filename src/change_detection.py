"""
Change detection algorithms for multi-temporal SAR analysis.
"""
import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.ndimage import label, binary_erosion, binary_dilation
from skimage.measure import regionprops


def compute_difference(image_before: np.ndarray, image_after: np.ndarray) -> np.ndarray:
    """
    Compute simple difference between two images.

    Args:
        image_before: Earlier image (dB scale recommended)
        image_after: Later image (dB scale recommended)

    Returns:
        Difference image (after - before)
    """
    if image_before.shape != image_after.shape:
        raise ValueError(
            f"Image shapes must match: {image_before.shape} vs {image_after.shape}"
        )

    return image_after - image_before


def compute_log_ratio(image_before: np.ndarray, image_after: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute log-ratio change detection (recommended for SAR).

    Log-ratio is more robust to multiplicative speckle noise.

    Args:
        image_before: Earlier image (linear intensity scale)
        image_after: Later image (linear intensity scale)
        epsilon: Small value to avoid log(0)

    Returns:
        Log-ratio image
    """
    if image_before.shape != image_after.shape:
        raise ValueError("Image shapes must match")

    # Ensure non-negative and avoid division by zero
    img_before = np.maximum(image_before, epsilon)
    img_after = np.maximum(image_after, epsilon)

    # Log-ratio: log(after/before) = log(after) - log(before)
    log_ratio = np.log(img_after) - np.log(img_before)

    return log_ratio


def detect_changes(
    image_before: np.ndarray,
    image_after: np.ndarray,
    threshold: float = 3.0,
    method: str = 'difference'
) -> np.ndarray:
    """
    Detect significant changes between two SAR images.

    Args:
        image_before: Earlier image (dB scale for 'difference', linear for 'log_ratio')
        image_after: Later image (same scale as before)
        threshold: Change detection threshold (dB for difference, std for log_ratio)
        method: Detection method ('difference' or 'log_ratio')

    Returns:
        Binary change map (1=change, 0=no change)
    """
    if method == 'difference':
        diff = compute_difference(image_before, image_after)
        # Changes are pixels where |difference| > threshold
        changes = np.abs(diff) > threshold

    elif method == 'log_ratio':
        log_ratio = compute_log_ratio(image_before, image_after)
        # Changes are pixels outside mean ± threshold*std
        mean_lr = np.mean(log_ratio)
        std_lr = np.std(log_ratio)
        changes = np.abs(log_ratio - mean_lr) > (threshold * std_lr)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'difference' or 'log_ratio'")

    return changes.astype(np.uint8)


def compute_change_magnitude(image_before: np.ndarray, image_after: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude of change between two images.

    Args:
        image_before: Earlier image
        image_after: Later image

    Returns:
        Change magnitude (absolute difference)
    """
    diff = compute_difference(image_before, image_after)
    return np.abs(diff)


def refine_change_map(change_map: np.ndarray, min_size: int = 10) -> np.ndarray:
    """
    Refine binary change map by removing small isolated regions.

    Args:
        change_map: Binary change map
        min_size: Minimum region size (pixels) to keep

    Returns:
        Refined change map
    """
    # Label connected components
    labeled, num_features = label(change_map)

    # Filter by size
    refined = np.zeros_like(change_map)
    for region in regionprops(labeled):
        if region.area >= min_size:
            coords = region.coords
            refined[coords[:, 0], coords[:, 1]] = 1

    return refined.astype(np.uint8)


def detect_bright_spots(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    threshold: Optional[float] = None,
    percentile: float = 99.0
) -> List[Dict]:
    """
    Detect bright spots in SAR image (planes, ships, buildings).

    Args:
        image: SAR image (dB scale or normalized)
        bbox: Optional bounding box (y_min, x_min, y_max, x_max) to search within
        threshold: Detection threshold (if None, uses percentile)
        percentile: Percentile for automatic threshold (default: 99.0 = top 1%)

    Returns:
        List of detected objects with properties
    """
    # Extract region of interest
    if bbox is not None:
        y_min, x_min, y_max, x_max = bbox
        roi = image[y_min:y_max, x_min:x_max]
        offset_y, offset_x = y_min, x_min
    else:
        roi = image
        offset_y, offset_x = 0, 0

    # Determine threshold
    if threshold is None:
        threshold = np.percentile(roi, percentile)

    # Create binary mask of bright pixels
    bright_mask = roi > threshold

    # Label connected components
    labeled, num_features = label(bright_mask)

    # Extract properties of each detected region
    detections = []
    for region in regionprops(labeled, intensity_image=roi):
        detection = {
            'centroid': (region.centroid[0] + offset_y, region.centroid[1] + offset_x),
            'area': region.area,
            'intensity_mean': region.intensity_mean,
            'intensity_max': region.intensity_max,
            'bbox': (
                region.bbox[0] + offset_y,
                region.bbox[1] + offset_x,
                region.bbox[2] + offset_y,
                region.bbox[3] + offset_x
            ),
            'eccentricity': region.eccentricity,
        }
        detections.append(detection)

    # Sort by intensity (brightest first)
    detections.sort(key=lambda x: x['intensity_max'], reverse=True)

    return detections


def filter_detections_by_size(
    detections: List[Dict],
    min_area: int = 5,
    max_area: int = 1000
) -> List[Dict]:
    """
    Filter detections by area (size).

    Args:
        detections: List of detection dictionaries
        min_area: Minimum area in pixels
        max_area: Maximum area in pixels

    Returns:
        Filtered list of detections
    """
    return [d for d in detections if min_area <= d['area'] <= max_area]


def categorize_changes(change_map: np.ndarray, magnitude: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Categorize changes into increase, decrease, or no change.

    Args:
        change_map: Binary change map
        magnitude: Signed change magnitude (positive = increase, negative = decrease)

    Returns:
        Dictionary with 'increase', 'decrease', and 'no_change' masks
    """
    increase = (change_map == 1) & (magnitude > 0)
    decrease = (change_map == 1) & (magnitude < 0)
    no_change = (change_map == 0)

    return {
        'increase': increase.astype(np.uint8),
        'decrease': decrease.astype(np.uint8),
        'no_change': no_change.astype(np.uint8)
    }
