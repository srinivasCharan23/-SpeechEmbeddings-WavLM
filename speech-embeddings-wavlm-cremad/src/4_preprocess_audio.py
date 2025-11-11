"""
Preprocess Audio Files for Emotion Recognition using WavLM-base (CREMA-D dataset)
Optimized for CPU-only GitHub Codespaces
"""

import os
import librosa
import numpy as np
import logging
from pathlib import Path

# ----------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Main preprocessing function
# ----------------------------------------------------------
def preprocess_audio(input_dir: str = "../data/raw", output_dir: str = "../data/processed"):
    """
    Preprocesses audio files to ensure they are in the correct format (16kHz WAV).
    Normalizes audio and saves the processed files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for audio_file in Path(input_dir).glob("*.wav"):
        logger.info(f"Processing audio file: {audio_file.name}")

        # Load audio file
        try:
            audio, sr = librosa.load(audio_file, sr=16000)  # Load with 16kHz sampling rate
            # Normalize audio
            audio = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]
            
            # Save processed audio
            output_file = Path(output_dir) / audio_file.name
            librosa.output.write_wav(output_file, audio, sr)
            logger.info(f"✅ Saved processed audio to: {output_file}")

        except Exception as e:
            logger.error(f"Error processing {audio_file.name}: {e}")

# ----------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------
if __name__ == "__main__":
    preprocess_audio()
