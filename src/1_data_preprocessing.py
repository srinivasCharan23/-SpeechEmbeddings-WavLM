"""
Data Preprocessing for Emotion Recognition using WavLM-base (RAVDESS dataset)
Optimized for CPU-only GitHub Codespaces (no manual dataset download)
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
import logging

# ----------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Main preprocessing function
# ----------------------------------------------------------
def preprocess_ravdess(output_dir: str = "../data/processed", subset: str = "train[:40%]"):
    """
    Loads a subset of the RAVDESS dataset from HuggingFace.
    Extracts file paths and emotion labels.
    Saves processed CSV for embedding extraction.
    """
    logger.info(f"Loading RAVDESS subset: {subset} from Hugging Face Hub...")
    dataset = load_dataset("ashraq/ravdess-emotional-speech-audio", split=subset)

    # Convert dataset to DataFrame
    df = pd.DataFrame(dataset)
    df["filepath"] = df["audio"].apply(lambda x: x["path"])
    df = df[["filepath", "emotion"]]

    logger.info(f"Loaded {len(df)} samples with emotions: {df['emotion'].unique().tolist()}")

    # Save metadata CSV
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_dir) / "ravdess_subset.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"✅ Saved processed CSV to: {output_csv}")

    return df


# ----------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------
if __name__ == "__main__":
    preprocess_ravdess()
