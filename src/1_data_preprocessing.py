"""
Data Preprocessing for Emotion Recognition using WavLM-base (IEMOCAP subset)
Optimized for CPU-only GitHub Codespaces (no manual dataset download)
"""

import os
import pandas as pd
from datasets import load_dataset
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def preprocess_iemocap(output_dir: str = "../data/processed", subset: str = "train[:5%]"):
    """
    Loads a small subset of IEMOCAP dataset from HuggingFace.
    Extracts file paths, emotions, and speaker info.
    Saves processed CSV for embedding extraction.
    """
    logger.info(f"Loading IEMOCAP subset: {subset}")
    dataset = load_dataset("iemocap", split=subset)

    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    df = df.rename(columns={"path": "filepath", "label": "emotion"})
    
    # Keep only relevant columns
    df = df[["filepath", "emotion"]]
    logger.info(f"Loaded {len(df)} samples")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_dir) / "iemocap_subset.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"✅ Saved processed CSV to: {output_csv}")

    return df


if __name__ == "__main__":
    preprocess_iemocap()
