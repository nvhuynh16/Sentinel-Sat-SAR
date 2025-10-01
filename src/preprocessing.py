"""
SAR image preprocessing module.
Handles speckle filtering, enhancement, and normalization.
"""
import numpy as np
from typing import Tuple
from scipy.ndimage import median_filter as scipy_median_filter
from scipy.ndimage import uniform_filter, generic_filter
from skimage import exposure


def convert_to_db(image: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Convert SAR intensity to dB scale.

    Args:
        image: SAR intensity image (linear scale)
        epsilon: Small value to avoid log(0)

    Returns:
        Image in dB scale (10*log10)
    """
    # Ensure non-negative values
    image = np.maximum(image, 0)

    # Convert to dB scale: 10 * log10(intensity)
    # Add epsilon to avoid log(0)
    db_image = 10 * np.log10(image + epsilon)

    return db_image


def convert_from_db(db_image: np.ndarray) -> np.ndarray:
    """
    Convert SAR image from dB scale back to linear intensity.

    Args:
        db_image: SAR image in dB scale

    Returns:
        Image in linear intensity scale
    """
    return 10 ** (db_image / 10)


def apply_speckle_filter(image: np.ndarray, method: str = 'median', size: int = 5) -> np.ndarray:
    """
    Apply speckle filtering to SAR image.

    Args:
        image: SAR image (can be linear or dB scale)
        method: Filter type ('median' or 'mean')
        size: Filter window size (odd number recommended)

    Returns:
        Filtered image
    """
    if method == 'median':
        # Median filter - good for preserving edges
        filtered = scipy_median_filter(image, size=size)
    elif method == 'mean':
        # Mean/uniform filter - simple averaging
        filtered = uniform_filter(image, size=size)
    else:
        raise ValueError(f"Unknown filter method: {method}. Use 'median' or 'mean'")

    return filtered


def lee_filter(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply Lee filter for speckle reduction.

    The Lee filter preserves edges better than simple averaging filters.

    Args:
        image: SAR image (linear scale)
        window_size: Size of the filter window

    Returns:
        Filtered image
    """
    # Calculate local mean and variance
    mean = uniform_filter(image, size=window_size)
    sqr_mean = uniform_filter(image**2, size=window_size)
    variance = sqr_mean - mean**2

    # Overall variance (noise variance estimate)
    overall_variance = np.var(image)

    # Lee filter weights
    weights = variance / (variance + overall_variance + 1e-10)

    # Apply filter: filtered = mean + weights * (original - mean)
    filtered = mean + weights * (image - mean)

    return filtered


def enhance_contrast(image: np.ndarray, method: str = 'equalize') -> np.ndarray:
    """
    Enhance image contrast.

    Args:
        image: Input image (dB scale recommended)
        method: Enhancement method ('equalize', 'adaptive', 'stretch')

    Returns:
        Contrast-enhanced image
    """
    if method == 'equalize':
        # Histogram equalization
        # Normalize to 0-1 range first
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
        enhanced = exposure.equalize_hist(img_norm)
        # Scale back to original range
        enhanced = enhanced * (image.max() - image.min()) + image.min()

    elif method == 'adaptive':
        # Adaptive histogram equalization (CLAHE)
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
        # Ensure uint8 or uint16 for CLAHE
        img_uint = (img_norm * 65535).astype(np.uint16)
        enhanced = exposure.equalize_adapthist(img_uint)
        # Scale back
        enhanced = enhanced * (image.max() - image.min()) + image.min()

    elif method == 'stretch':
        # Contrast stretching (2% linear stretch)
        p2, p98 = np.percentile(image, (2, 98))
        enhanced = exposure.rescale_intensity(image, in_range=(p2, p98))

    else:
        raise ValueError(f"Unknown method: {method}. Use 'equalize', 'adaptive', or 'stretch'")

    return enhanced


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """
    Normalize image intensity to 0-1 range.

    Args:
        image: Input image

    Returns:
        Normalized image (0-1 range)
    """
    img_min = image.min()
    img_max = image.max()

    if img_max - img_min < 1e-10:
        # Avoid division by zero for constant images
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - img_min) / (img_max - img_min)
    return normalized.astype(np.float32)
