"""
Downloads and extracts the dataset from Kaggle.
"""

import opendatasets as od
import os
import argparse
import logging
import sys

# --- Add project root to sys.path ---
# This allows imports from the 'src' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------------------------

from src.config import KAGGLE_DATASET_URL, DATA_DIR
from src.utils import setup_logging

def download_dataset(url, data_dir):
    """Uses opendatasets to download and extract the Kaggle dataset."""
    try:
        logging.info(f"Downloading dataset from {url} to {data_dir}...")
        
        # opendatasets will prompt for Kaggle username and key if not found
        od.download(url, data_dir)
        
        # The dataset is downloaded into a folder with the dataset's name
        # We want to confirm the path
        downloaded_path = os.path.join(data_dir, url.split('/')[-1])
        if os.path.exists(downloaded_path):
             logging.info(f"Dataset successfully downloaded to: {downloaded_path}")
             return downloaded_path
        else:
            logging.error(f"Dataset download complete, but expected folder '{downloaded_path}' not found.")
            return None

    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")
        logging.error("Please ensure you have a 'kaggle.json' file in ~/.kaggle/ or provide your API credentials when prompted.")
        sys.exit(1)

if __name__ == "__main__":
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    setup_logging("logs/dataset_setup.log")

    # The dataset URL and local data dir are from config
    # We just run the script
    parser = argparse.ArgumentParser(description="Download and setup the Birds 20 Species dataset from Kaggle.")
    parser.add_argument('--url', type=str, default=KAGGLE_DATASET_URL, help='Kaggle dataset URL.')
    parser.add_argument('--output_dir', type=str, default=os.path.dirname(DATA_DIR), help='Directory to download data into.')
    
    args = parser.parse_args()

    # Ensure the parent data directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    download_dataset(args.url, args.output_dir)
