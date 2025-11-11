"""
Data Preprocessing for Emotion Recognition using WavLM-base (CREMA-D dataset)
Optimized for CPU-only GitHub Codespaces (no manual dataset download)
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
import logging
import os
import zipfile
import requests
from tqdm import tqdm
import numpy as np

# ----------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# CREMA-D emotion label mapping
# ----------------------------------------------------------
CREMAD_EMOTION_MAP = {
    'ANG': 'Angry',
    'DIS': 'Disgust',
    'FEA': 'Fear',
    'HAP': 'Happy',
    'NEU': 'Neutral',
    'SAD': 'Sad'
}

# ----------------------------------------------------------
# Fallback download function
# ----------------------------------------------------------
def download_cremad_fallback(data_dir: str = "../data"):
    """
    Download CREMA-D dataset from Kaggle mirror if Hugging Face fails.
    This is a fallback mechanism for network restrictions.
    
    Args:
        data_dir: Directory to download and extract the dataset
    
    Returns:
        Path to the extracted audio files directory
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    cremad_dir = data_path / "CREMA-D"
    cremad_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: Direct download URL for CREMA-D
    # Since Kaggle requires authentication, we'll use the official source
    logger.warning("⚠️ Fallback download is not fully automated for CREMA-D.")
    logger.warning("Please download the dataset manually from:")
    logger.warning("  - Official: https://github.com/CheyneyComputerScience/CREMA-D")
    logger.warning("  - Kaggle: https://www.kaggle.com/datasets/ejlok1/cremad")
    logger.warning(f"Extract all .wav files to: {cremad_dir}")
    
    return cremad_dir


# ----------------------------------------------------------
# Parse CREMA-D filename for emotion label
# ----------------------------------------------------------
def parse_cremad_filename(filename: str) -> str:
    """
    Parse emotion label from CREMA-D filename.
    
    Filename format: ActorID_SentenceID_Emotion_EmotionLevel.wav
    Example: 1001_DFA_ANG_XX.wav -> Angry
    
    Args:
        filename: The audio filename
        
    Returns:
        Emotion label (e.g., 'Angry', 'Happy')
    """
    try:
        # Remove .wav extension and split by underscore
        parts = filename.replace('.wav', '').split('_')
        if len(parts) >= 3:
            emotion_code = parts[2]
            return CREMAD_EMOTION_MAP.get(emotion_code, 'Unknown')
        else:
            logger.warning(f"Could not parse filename: {filename}")
            return 'Unknown'
    except Exception as e:
        logger.error(f"Error parsing filename {filename}: {e}")
        return 'Unknown'


# ----------------------------------------------------------
# Generate synthetic CREMA-D data for testing
# ----------------------------------------------------------
def generate_synthetic_cremad(output_dir: str = "../data/processed", num_samples: int = 100):
    """
    Generate synthetic CREMA-D dataset for testing when actual dataset is unavailable.
    Creates dummy audio files and metadata CSV.
    
    Args:
        output_dir: Directory to save the processed CSV
        num_samples: Number of synthetic samples to generate
    
    Returns:
        DataFrame with filepath and emotion columns
    """
    import numpy as np
    import soundfile as sf
    
    logger.warning("⚠️ Generating synthetic CREMA-D data for testing...")
    logger.warning("This is NOT real data - replace with actual CREMA-D dataset for production use")
    
    # Ensure directories exist
    data_dir = Path(output_dir).parent
    cremad_synthetic_dir = data_dir / "CREMA-D_synthetic"
    cremad_synthetic_dir.mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    emotions = list(CREMAD_EMOTION_MAP.values())
    data = []
    
    logger.info(f"Creating {num_samples} synthetic audio files...")
    
    for i in tqdm(range(num_samples), desc="Generating synthetic data"):
        # Select random emotion
        emotion = emotions[i % len(emotions)]
        
        # Generate a simple synthetic audio signal (1 second, 16kHz)
        duration = 1.0  # seconds
        sample_rate = 16000
        num_samples_audio = int(duration * sample_rate)
        
        # Create a simple tone with some noise
        t = np.linspace(0, duration, num_samples_audio)
        frequency = 200 + (i % 10) * 50  # Vary frequency
        audio = 0.3 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(num_samples_audio)
        audio = audio.astype(np.float32)
        
        # Create filename in CREMA-D format
        actor_id = 1000 + (i % 91)  # CREMA-D has 91 actors
        sentence_id = ['IEO', 'IOM', 'ITH', 'IWW', 'TAI', 'MTI'][i % 6]
        emotion_code = [k for k, v in CREMAD_EMOTION_MAP.items() if v == emotion][0]
        level = ['LO', 'MD', 'HI', 'XX'][i % 4]
        filename = f"{actor_id}_{sentence_id}_{emotion_code}_{level}.wav"
        
        filepath = cremad_synthetic_dir / filename
        
        # Save audio file
        sf.write(filepath, audio, sample_rate)
        
        data.append({
            "filepath": str(filepath.absolute()),
            "emotion": emotion
        })
    
    df = pd.DataFrame(data)
    
    logger.info(f"Generated {len(df)} synthetic samples")
    logger.info(f"Emotion distribution:")
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        logger.info(f"  {emotion}: {count}")
    
    # Save metadata CSV
    output_csv = Path(output_dir) / "cremad_subset.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"✅ Saved synthetic data CSV to: {output_csv}")
    logger.info(f"✅ Synthetic audio files saved to: {cremad_synthetic_dir}")
    
    return df


# ----------------------------------------------------------
# Main preprocessing function for CREMA-D
# ----------------------------------------------------------
def preprocess_cremad(output_dir: str = "../data/processed", subset: str = None, use_synthetic: bool = False):
    """
    Loads CREMA-D dataset from HuggingFace or fallback source.
    Extracts file paths and emotion labels.
    Saves processed CSV for embedding extraction.
    
    Args:
        output_dir: Directory to save the processed CSV
        subset: Optional subset specification (e.g., "train[:40%]")
        use_synthetic: If True, generate synthetic data instead of loading real data
    
    Returns:
        DataFrame with filepath and emotion columns
    """
    if use_synthetic:
        return generate_synthetic_cremad(output_dir)
    
    logger.info("Loading CREMA-D dataset...")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Try loading from HuggingFace first
        logger.info("Attempting to load from Hugging Face Hub (m3hrdadfi/crema-d)...")
        
        # Load the dataset
        if subset:
            dataset = load_dataset("m3hrdadfi/crema-d", split=subset)
        else:
            # Load a reasonable subset for testing in Codespaces
            dataset = load_dataset("m3hrdadfi/crema-d", split="train[:1000]")
        
        logger.info(f"✅ Successfully loaded {len(dataset)} samples from Hugging Face")
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        
        # Extract filepath and emotion from the dataset
        if "audio" in df.columns:
            df["filepath"] = df["audio"].apply(lambda x: x["path"] if isinstance(x, dict) else x)
        
        if "emotion" in df.columns:
            df["emotion"] = df["emotion"]
        elif "label" in df.columns:
            # Some datasets use 'label' instead of 'emotion'
            df["emotion"] = df["label"]
        else:
            # Parse emotion from filename if not in dataset
            logger.info("Emotion labels not found in dataset. Parsing from filenames...")
            df["emotion"] = df["filepath"].apply(lambda x: parse_cremad_filename(Path(x).name))
        
        # Keep only necessary columns
        df = df[["filepath", "emotion"]]
        
        # Filter out any 'Unknown' emotions
        df = df[df["emotion"] != "Unknown"]
        
    except Exception as e:
        logger.warning(f"⚠️ Could not load from Hugging Face: {e}")
        logger.info("Attempting fallback: looking for local CREMA-D files...")
        
        # Fallback: Look for local files
        data_dir = Path(output_dir).parent
        cremad_dir = data_dir / "CREMA-D"
        
        if not cremad_dir.exists():
            # Try to download
            cremad_dir = download_cremad_fallback(data_dir)
        
        # Scan for .wav files
        wav_files = list(cremad_dir.rglob("*.wav"))
        
        if not wav_files:
            logger.error(f"❌ No .wav files found in {cremad_dir}")
            logger.warning("Falling back to synthetic data generation...")
            return generate_synthetic_cremad(output_dir)
        
        logger.info(f"Found {len(wav_files)} audio files in {cremad_dir}")
        
        # Create DataFrame from local files
        data = []
        for wav_file in tqdm(wav_files, desc="Parsing files"):
            emotion = parse_cremad_filename(wav_file.name)
            if emotion != "Unknown":
                data.append({
                    "filepath": str(wav_file.absolute()),
                    "emotion": emotion
                })
        
        df = pd.DataFrame(data)
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Emotion distribution:")
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        logger.info(f"  {emotion}: {count}")
    
    # Save metadata CSV
    output_csv = Path(output_dir) / "cremad_subset.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"✅ Saved processed CSV to: {output_csv}")
    
    return df


# ----------------------------------------------------------
# Backward compatibility function
# ----------------------------------------------------------
def preprocess_ravdess(output_dir: str = "../data/processed", subset: str = "train[:40%]"):
    """
    Loads a subset of the RAVDESS dataset from HuggingFace.
    Extracts file paths and emotion labels.
    Saves processed CSV for embedding extraction.
    
    (Kept for backward compatibility)
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
    preprocess_cremad()
