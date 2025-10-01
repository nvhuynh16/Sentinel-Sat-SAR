import requests
import getpass
from datetime import datetime, timedelta
from pathlib import Path

class CopernicusAPI:
    def __init__(self):
        self.auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        self.download_url = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"
        self.token = None
        self.token_expiry = None
        self.username = None
        self.password = None
        
    def authenticate(self):
        """Get access token using credentials"""
        self.username = input("Enter Copernicus username (email): ")
        self.password = getpass.getpass("Enter password: ")
        self._get_token()

    def authenticate_from_file(self, filepath='login.txt', auto_token=True):
        """
        Load credentials from a text file and optionally authenticate.

        Args:
            filepath (str): Path to credentials file. Default: 'login.txt'
                           File format:
                           Username: your.email@domain.com
                           Password: your_password
            auto_token (bool): Automatically get access token. Default: True

        Raises:
            FileNotFoundError: If credentials file doesn't exist
            ValueError: If file format is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Credentials file not found: {filepath}")

        # Read and parse the file
        with open(filepath, 'r') as f:
            lines = f.readlines()

        username_line = None
        password_line = None

        for line in lines:
            line = line.strip()
            if line.startswith('Username:'):
                username_line = line
            elif line.startswith('Password:'):
                password_line = line

        # Validate that both fields were found
        if not username_line or not password_line:
            raise ValueError(
                f"Invalid credentials file format. Expected:\n"
                f"Username: your.email@domain.com\n"
                f"Password: your_password"
            )

        # Extract values (everything after the colon)
        self.username = username_line.split(':', 1)[1].strip()
        self.password = password_line.split(':', 1)[1].strip()

        # Validate credentials are not empty
        if not self.username or not self.password:
            raise ValueError("Username or password cannot be empty")

        # Get the access token if requested
        if auto_token:
            self._get_token()
        
    def _get_token(self):
        """Internal method to get/refresh token"""
        data = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "client_id": "cdse-public"
        }
        
        response = requests.post(self.auth_url, data=data)
        response.raise_for_status()
        result = response.json()
        self.token = result["access_token"]
        # Token typically expires in 600 seconds (10 min)
        expires_in = result.get("expires_in", 600)
        self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)  # Refresh 1 min early
        print("Authentication successful!")
        
    def _ensure_valid_token(self):
        """Check and refresh token if needed"""
        if not self.token or datetime.now() >= self.token_expiry:
            print("Token expired, refreshing...")
            self._get_token()
        
    def search(self, collection, start_date, end_date, lon, lat, max_cloud_cover=None):
        """
        Search for products
        collection: 'SENTINEL-1' or 'SENTINEL-2'
        start_date, end_date: 'YYYY-MM-DD' format
        lon, lat: coordinates
        max_cloud_cover: 0-100 (only for Sentinel-2)
        """
        filter_query = (
            f"Collection/Name eq '{collection}' and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})') and "
            f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
            f"ContentDate/Start lt {end_date}T23:59:59.999Z"
        )
        
        if max_cloud_cover is not None and collection == 'SENTINEL-2':
            filter_query += f" and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {max_cloud_cover})"
        
        params = {"$filter": filter_query}
        response = requests.get(self.search_url, params=params)
        response.raise_for_status()
        
        products = response.json()['value']
        print(f"Found {len(products)} products")
        return products
        
    def download(self, product, output_dir="./downloads"):
        """Download a product"""
        if not self.username:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        self._ensure_valid_token()  # Refresh if needed
            
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        product_id = product['Id']
        product_name = product['Name']
        url = f"{self.download_url}({product_id})/$value"
        
        headers = {"Authorization": f"Bearer {self.token}"}
        
        print(f"Downloading {product_name}...")
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        filepath = Path(output_dir) / f"{product_name}.zip"
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Downloaded to {filepath}")
        return filepath