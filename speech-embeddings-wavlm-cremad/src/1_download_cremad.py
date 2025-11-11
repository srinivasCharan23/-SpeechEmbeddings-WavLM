"""
Download CREMA-D Dataset for Speech Emotion Recognition
This script attempts to download the CREMA-D dataset from Hugging Face. 
If the download fails, it falls back to an official GitHub mirror or Kaggle link.
"""

import os
import logging
from datasets import load_dataset, DownloadConfig
import requests
import zipfile
import io

# ----------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------
HUGGINGFACE_DATASET = "crema-d"
GITHUB_MIRROR_URL = "https://github.com/CheyneyRyan/CREMA-D/archive/refs/heads/master.zip"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/cheyneyryan/crema-d"

# ----------------------------------------------------------
# Download function
# ----------------------------------------------------------
def download_cremad(output_dir: str = "../data/raw"):
    """
    Downloads the CREMA-D dataset from Hugging Face or falls back to a GitHub mirror.
    """
    try:
        logger.info("Attempting to download CREMA-D dataset from Hugging Face...")
        dataset = load_dataset(HUGGINGFACE_DATASET)
        logger.info("Successfully downloaded CREMA-D dataset from Hugging Face.")
        
        # Save the dataset to the output directory
        os.makedirs(output_dir, exist_ok=True)
        # Assuming the dataset is saved in a specific format, adjust as necessary
        # This part may need to be customized based on the dataset structure
        for item in dataset['train']:
            # Save each audio file to the output directory
            audio_url = item['audio']['path']
            audio_response = requests.get(audio_url)
            audio_filename = os.path.join(output_dir, os.path.basename(audio_url))
            with open(audio_filename, 'wb') as f:
                f.write(audio_response.content)
        
    except Exception as e:
        logger.warning(f"Failed to download from Hugging Face: {e}")
        logger.info("Attempting to download CREMA-D dataset from GitHub mirror...")
        try:
            response = requests.get(GITHUB_MIRROR_URL)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(output_dir)
            logger.info("Successfully downloaded and extracted CREMA-D dataset from GitHub mirror.")
        except Exception as e:
            logger.error(f"Failed to download from GitHub mirror: {e}")
            logger.info(f"Please download the dataset manually from Kaggle: {KAGGLE_DATASET_URL}")

# ----------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------
if __name__ == "__main__":
    download_cremad()
"""