import pandas as pd
from pathlib import Path
import logging

# ----------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Build manifest function
# ----------------------------------------------------------
def build_manifest(output_dir: str = "../data/processed"):
    """
    Compiles the processed data into a CSV file for the CREMA-D dataset.
    The CSV contains file paths and corresponding emotion labels.
    """
    logger.info("Building manifest for CREMA-D dataset...")

    # Load the processed data
    processed_data_path = Path(output_dir) / "cremad_subset.csv"
    
    if not processed_data_path.exists():
        logger.error(f"Processed data file not found: {processed_data_path}")
        return

    # Read the processed data
    df = pd.read_csv(processed_data_path)

    # Save the manifest CSV
    manifest_csv = Path(output_dir) / "cremad_manifest.csv"
    df.to_csv(manifest_csv, index=False)
    logger.info(f"✅ Saved manifest to: {manifest_csv}")

# ----------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------
if __name__ == "__main__":
    build_manifest()