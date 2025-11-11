"""
Extracts the downloaded CREMA-D dataset ZIP file and prepares audio files for processing.

This script ensures that all WAV audio files are extracted and ready for further processing.
"""

import zipfile
import os
from pathlib import Path
import logging

# ----------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Main extraction function
# ----------------------------------------------------------
def extract_cremad(zip_file_path: str, output_dir: str = "../data/raw"):
    """
    Extracts the CREMA-D dataset ZIP file to the specified output directory.
    Ensures all WAV audio files are extracted.
    """
    logger.info(f"Extracting CREMA-D dataset from: {zip_file_path}...")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        logger.info(f"✅ Extracted files to: {output_dir}")

    # List all extracted WAV files
    wav_files = list(Path(output_dir).rglob("*.wav"))
    logger.info(f"Extracted {len(wav_files)} WAV files.")

    return wav_files


# ----------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------
if __name__ == "__main__":
    zip_file_path = "../data/raw/crema_d.zip"  # Adjust the path as necessary
    extract_cremad(zip_file_path)
"""