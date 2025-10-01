"""Simple script to download remaining Sentinel-1 GRDH products."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from CopernicusAPI import CopernicusAPI

api = CopernicusAPI()
api.authenticate_from_file('login.txt')

# Products we want to download
target_products = [
    'S1A_IW_GRDH_1SDV_20250816T025212_20250816T025237_060555_078800_1619.SAFE',
    'S1A_IW_GRDH_1SDV_20250821T030036_20250821T030101_060628_078AE4_A651.SAFE',
    'S1A_IW_GRDH_1SDV_20250809T030036_20250809T030101_060453_078400_A074.SAFE',
    'S1A_IW_GRDH_1SDV_20250828T025212_20250828T025237_060730_07907B_3A43.SAFE'
]

print("Searching for products...")
products = api.search(
    collection='SENTINEL-1',
    start_date='2025-08-01',
    end_date='2025-09-01',
    lon=43.1497,
    lat=11.5449
)

# Filter for our target GRDH products
products_to_download = [p for p in products if any(target in p['Name'] for target in target_products)]

print(f"\nWill download {len(products_to_download)} products")
output_dir = "./data/sentinel1"

for i, product in enumerate(products_to_download, 1):
    print(f"\n[{i}/{len(products_to_download)}] {product['Name']}")
    try:
        filepath = api.download(product, output_dir=output_dir)
        print(f"  Downloaded: {filepath.name}")
    except Exception as e:
        print(f"  Error: {e}")

print("\nDone!")
