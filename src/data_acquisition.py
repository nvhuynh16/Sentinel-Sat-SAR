"""
Data acquisition module for Sentinel-1 SAR imagery.
Wrapper around CopernicusAPI with convenience functions.
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Import CopernicusAPI from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from CopernicusAPI import CopernicusAPI


# Hainan Naval Base coordinates
HAINAN_BASE = {
    'name': 'Yulin Naval Base',
    'lon': 109.5450,
    'lat': 18.2300
}


def download_sentinel1_imagery(
    start_date: str,
    end_date: str,
    location: Dict[str, float] = HAINAN_BASE,
    output_dir: str = "./data/sentinel1",
    max_results: int = 6
) -> List[Path]:
    """
    Download Sentinel-1 SAR imagery for specified date range and location.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        location: Dictionary with 'lon' and 'lat' keys
        output_dir: Directory to save downloaded imagery
        max_results: Maximum number of products to download

    Returns:
        List of paths to downloaded files

    Example:
        >>> files = download_sentinel1_imagery('2025-08-01', '2025-09-01')
        >>> print(f"Downloaded {len(files)} images")
    """
    api = CopernicusAPI()

    # Authenticate using login.txt
    api.authenticate_from_file()

    # Search for Sentinel-1 products
    print(f"Searching for Sentinel-1 imagery from {start_date} to {end_date}")
    print(f"Location: {location['name']} ({location['lon']}, {location['lat']})")

    products = api.search(
        collection='SENTINEL-1',
        start_date=start_date,
        end_date=end_date,
        lon=location['lon'],
        lat=location['lat']
    )

    if not products:
        print("No products found!")
        return []

    # Limit results
    products = products[:max_results]
    print(f"Downloading {len(products)} products...")

    # Download each product
    downloaded_files = []
    for i, product in enumerate(products, 1):
        print(f"\n[{i}/{len(products)}] {product['Name']}")
        filepath = api.download(product, output_dir=output_dir)
        downloaded_files.append(filepath)

    return downloaded_files


def list_downloaded_imagery(data_dir: str = "./data/sentinel1") -> List[Path]:
    """
    List all downloaded Sentinel-1 imagery in the data directory.

    Args:
        data_dir: Directory containing downloaded imagery

    Returns:
        List of paths to .zip files
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    return sorted(data_path.glob("*.zip"))


if __name__ == "__main__":
    # Example usage
    print("Sentinel-1 Data Acquisition Module")
    print("=" * 50)

    # Check existing downloads
    existing = list_downloaded_imagery()
    print(f"\nExisting downloads: {len(existing)}")

    if not existing:
        print("\nTo download imagery, run:")
        print(">>> files = download_sentinel1_imagery('2025-08-01', '2025-09-01')")
