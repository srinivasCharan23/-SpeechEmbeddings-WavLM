"""
Parse Emotion Labels from CREMA-D Filenames

This script parses the emotion labels from the filenames of the extracted WAV files
from the CREMA-D dataset. It extracts the emotion class (e.g., Angry, Disgust, Fear,
Happy, Neutral, Sad) from each filename and prepares the data for further processing.
"""

import os
import pandas as pd
from pathlib import Path
import logging

# ----------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Main function to parse labels
# ----------------------------------------------------------
def parse_labels(audio_dir: str = "../data/raw", output_dir: str = "../data/processed/cremad_subset.csv"):
    """
    Parses emotion labels from the filenames of WAV files in the specified directory.
    Saves the results to a CSV file.
    """
    logger.info(f"Parsing labels from audio files in directory: {audio_dir}...")
    
    # Prepare a list to hold the parsed data
    data = []

    # Iterate through all WAV files in the specified directory
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            # Extract emotion label from the filename
            # Assuming the filename format is: <Emotion>_<OtherInfo>.wav
            emotion = filename.split("_")[0]  # Adjust based on actual filename format
            filepath = os.path.join(audio_dir, filename)
            data.append({"filepath": filepath, "emotion": emotion})

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_dir, index=False)
    logger.info(f"✅ Saved parsed labels to: {output_dir}")

    return df

# ----------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------
if __name__ == "__main__":
    parse_labels()
"""