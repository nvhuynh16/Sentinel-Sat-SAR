"""
Utility functions for SAR data processing.
"""
from pathlib import Path
from typing import Optional, Tuple
import zipfile
import numpy as np
import rasterio


def extract_sentinel_product(zip_path: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Extract Sentinel-1 .SAFE product from zip file.

    Args:
        zip_path: Path to .zip file
        output_dir: Directory to extract to (default: same as zip)

    Returns:
        Path to extracted .SAFE directory
    """
    zip_path = Path(zip_path)
    if output_dir is None:
        output_dir = zip_path.parent

    output_dir = Path(output_dir)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Find the .SAFE directory
    safe_name = zip_path.stem  # Remove .zip extension
    safe_dir = output_dir / safe_name

    if not safe_dir.exists():
        raise FileNotFoundError(f"Extracted .SAFE directory not found: {safe_dir}")

    return safe_dir


def load_sar_band(safe_dir: Path, polarization: str = 'VV') -> Tuple[np.ndarray, dict]:
    """
    Load a specific polarization band from Sentinel-1 .SAFE directory.

    Args:
        safe_dir: Path to .SAFE directory
        polarization: Polarization to load ('VV' or 'VH')

    Returns:
        Tuple of (SAR image array, metadata dict)
    """
    safe_dir = Path(safe_dir)
    measurement_dir = safe_dir / 'measurement'

    if not measurement_dir.exists():
        raise FileNotFoundError(f"Measurement directory not found: {measurement_dir}")

    # Find the correct TIFF file for the polarization
    pol_pattern = f"*{polarization.lower()}*.tiff"
    tiff_files = list(measurement_dir.glob(pol_pattern))

    if not tiff_files:
        raise FileNotFoundError(
            f"No {polarization} polarization file found in {measurement_dir}"
        )

    tiff_path = tiff_files[0]

    # Load the data with rasterio
    with rasterio.open(tiff_path) as src:
        data = src.read(1)  # Read first band
        metadata = {
            'shape': src.shape,
            'crs': src.crs,
            'bounds': src.bounds,
            'transform': src.transform,
            'dtype': src.dtypes[0],
            'filename': tiff_path.name,
            'polarization': polarization
        }

    return data, metadata


def get_safe_info(safe_dir: Path) -> dict:
    """
    Get information about a Sentinel-1 .SAFE product.

    Args:
        safe_dir: Path to .SAFE directory

    Returns:
        Dictionary with product information
    """
    safe_dir = Path(safe_dir)
    name = safe_dir.name

    # Parse the filename to extract information
    # Format: S1A_IW_GRDH_1SDV_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_OOOOOO_DDDDDD_XXXX.SAFE
    parts = name.split('_')

    info = {
        'safe_dir': safe_dir,
        'product_name': name,
        'satellite': parts[0] if len(parts) > 0 else None,
        'mode': parts[1] if len(parts) > 1 else None,
        'product_type': parts[2] if len(parts) > 2 else None,
        'start_time': parts[4] if len(parts) > 4 else None,
    }

    # Check for available polarizations
    measurement_dir = safe_dir / 'measurement'
    if measurement_dir.exists():
        polarizations = []
        if list(measurement_dir.glob("*vv*.tiff")):
            polarizations.append('VV')
        if list(measurement_dir.glob("*vh*.tiff")):
            polarizations.append('VH')
        info['polarizations'] = polarizations

    return info
