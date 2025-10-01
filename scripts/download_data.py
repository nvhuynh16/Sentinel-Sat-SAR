"""
Script to download Sentinel-1 SAR imagery for Djibouti Support Base.
Run this to download the dataset for analysis.
"""
from CopernicusAPI import CopernicusAPI
from pathlib import Path


def main():
    # Initialize API
    api = CopernicusAPI()

    # Authenticate
    print("Authenticating with Copernicus Data Space...")
    api.authenticate_from_file('login.txt')

    # Search parameters
    start_date = '2025-08-01'
    end_date = '2025-09-01'
    lon = 43.1497  # Djibouti Support Base
    lat = 11.5449

    print(f"\nSearching for Sentinel-1 imagery:")
    print(f"  Location: Camp Lemonnier ({lon}, {lat})")
    print(f"  Date range: {start_date} to {end_date}")

    # Search for products
    products = api.search(
        collection='SENTINEL-1',
        start_date=start_date,
        end_date=end_date,
        lon=lon,
        lat=lat
    )

    print(f"\nFound {len(products)} total products")

    # Filter for GRDH products (Ground Range Detected High resolution)
    grdh_products = [p for p in products if 'GRDH' in p['Name']]
    print(f"Filtered to {len(grdh_products)} GRDH products")

    # Limit to 6 products
    products_to_download = grdh_products[:6]

    print(f"\nWill download {len(products_to_download)} products:")
    for i, p in enumerate(products_to_download, 1):
        print(f"  {i}. {p['Name']}")

    # Create output directory
    output_dir = "./data/sentinel1"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Download each product
    print(f"\nStarting download to {output_dir}/")
    print("=" * 70)

    downloaded_files = []
    for i, product in enumerate(products_to_download, 1):
        print(f"\n[{i}/{len(products_to_download)}] Downloading: {product['Name']}")
        print(f"  Size: {product.get('ContentLength', 'Unknown')} bytes")

        try:
            filepath = api.download(product, output_dir=output_dir)
            downloaded_files.append(filepath)
            print(f"  [OK] Complete: {filepath}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n" + "=" * 70)
    print(f"Download complete! Successfully downloaded {len(downloaded_files)}/{len(products_to_download)} files")

    if downloaded_files:
        print("\nDownloaded files:")
        for f in downloaded_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
