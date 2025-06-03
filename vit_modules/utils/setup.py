"""Environment setup utilities."""

import subprocess
import sys
from pathlib import Path
import zipfile
import requests


def setup_environment():
    """Setup the environment with required packages."""
    try:
        import torch
        import torchvision
        
        # Check versions
        torch_version = int(torch.__version__.split(".")[1])
        torchvision_version = int(torchvision.__version__.split(".")[1])
        
        assert torch_version >= 12 or int(torch.__version__.split(".")[0]) == 2
        assert torchvision_version >= 13
        
        print(f"torch version: {torch.__version__}")
        print(f"torchvision version: {torchvision.__version__}")
        
    except (ImportError, AssertionError):
        print("[INFO] Installing required torch/torchvision versions...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-U",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        
    # Install additional packages
    try:
        from torchinfo import summary
    except ImportError:
        print("[INFO] Installing torchinfo...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchinfo"])


def download_data(source: str, destination: str) -> Path:
    """Download and extract data from a URL.
    
    Args:
        source (str): URL to download from.
        destination (str): Directory to extract to.
        
    Returns:
        Path: Path to the extracted data.
    """
    destination_path = Path(destination)
    
    if destination_path.exists():
        print(f"[INFO] {destination} already exists, skipping download.")
        return destination_path
        
    # Download the file
    print(f"[INFO] Downloading {source}...")
    response = requests.get(source)
    
    # Save as zip file
    zip_path = destination_path.with_suffix('.zip')
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the zip file
    print(f"[INFO] Extracting to {destination}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    
    # Remove the zip file
    zip_path.unlink()
    
    return destination_path