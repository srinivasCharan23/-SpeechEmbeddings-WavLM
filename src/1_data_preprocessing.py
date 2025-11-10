"""
Data Preprocessing Script for Speech Embeddings using WavLM-base
==================================================================

This script handles the preprocessing of IEMOCAP dataset for emotion recognition.
Optimized for CPU-only execution with small dataset subset.

Based on the IEEE/ACM 2024 paper:
'From Raw Speech to Fixed Representations: A Comprehensive Evaluation 
of Speech Embedding Techniques'

Author: AI/ML Team
Date: 2024
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses IEMOCAP dataset for emotion recognition (CPU-optimized).
    
    Loads data from HuggingFace and filters to 4 emotion classes.
    """
    
    def __init__(self, data_dir: str = "../data", output_dir: str = "../data/processed"):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir: Root directory containing raw datasets
            output_dir: Directory for processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard sampling rate for WavLM
        self.target_sr = 16000
        
        # Emotion mapping for 4 classes
        self.emotion_map = {
            'neu': 'neutral',
            'neutral': 'neutral',
            'hap': 'happy',
            'happy': 'happy',
            'sad': 'sad',
            'ang': 'angry',
            'angry': 'angry',
        }
        
        logger.info(f"DataPreprocessor initialized with data_dir: {data_dir}")
        logger.info(f"Target emotions: {list(set(self.emotion_map.values()))}")
    
    def preprocess_iemocap(self, subset_percent: float = 5.0) -> pd.DataFrame:
        """
        Preprocess IEMOCAP dataset for emotion recognition using HuggingFace.
        
        Args:
            subset_percent: Percentage of training data to use (default: 5%)
        
        Returns:
            DataFrame with columns: [filepath, emotion, speaker, session]
        """
        logger.info("Loading IEMOCAP dataset from HuggingFace...")
        logger.info(f"Using {subset_percent}% of training data for CPU efficiency")
        
        try:
            # Load IEMOCAP from HuggingFace datasets
            # Using a small subset for CPU processing
            dataset = load_dataset("Zahra99/IEMOCAP", split=f"train[:{subset_percent}%]")
            
            logger.info(f"Loaded {len(dataset)} samples from IEMOCAP")
            
            data_entries = []
            
            # Process each sample
            for idx, sample in enumerate(tqdm(dataset, desc="Processing IEMOCAP")):
                # Get emotion label
                emotion_raw = sample.get('label', sample.get('emotion', 'unknown'))
                
                # Handle numeric labels (convert to string)
                if isinstance(emotion_raw, int):
                    # IEMOCAP label mapping (common encoding)
                    label_names = ['neutral', 'happy', 'sad', 'angry', 'frustrated', 'excited', 'fearful', 'surprised']
                    if 0 <= emotion_raw < len(label_names):
                        emotion_raw = label_names[emotion_raw]
                    else:
                        emotion_raw = 'unknown'
                
                # Normalize emotion to lowercase
                emotion_normalized = str(emotion_raw).lower().strip()
                
                # Map to 4 target emotions
                if emotion_normalized in self.emotion_map:
                    emotion = self.emotion_map[emotion_normalized]
                elif emotion_normalized in ['exc', 'excited']:
                    # Map excited to happy
                    emotion = 'happy'
                else:
                    # Skip other emotions
                    continue
                
                # Save audio to temporary location
                audio_data = sample['audio']
                audio_path = self.data_dir / f"temp_audio_{idx}.wav"
                
                # Save audio array to file
                import soundfile as sf
                sf.write(str(audio_path), audio_data['array'], audio_data['sampling_rate'])
                
                data_entries.append({
                    'filepath': str(audio_path),
                    'emotion': emotion,
                    'speaker': sample.get('speaker_id', f'speaker_{idx}'),
                    'session': sample.get('session', 'session_1'),
                    'sampling_rate': audio_data['sampling_rate']
                })
            
            df = pd.DataFrame(data_entries)
            
            # Log emotion distribution
            if not df.empty:
                logger.info(f"\nEmotion distribution:")
                for emotion, count in df['emotion'].value_counts().items():
                    logger.info(f"  {emotion}: {count} samples")
            
            logger.info(f"IEMOCAP preprocessing complete. {len(df)} samples processed.")
            return df
            
        except Exception as e:
            logger.error(f"Error loading IEMOCAP from HuggingFace: {e}")
            logger.info("Falling back to empty dataset...")
            return pd.DataFrame()
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Validate dataset for missing files and inconsistencies.
        
        Args:
            df: DataFrame with dataset information
            dataset_name: Name of the dataset
            
        Returns:
            Validated DataFrame
        """
        logger.info(f"Validating {dataset_name} dataset...")
        
        if df.empty:
            logger.warning(f"{dataset_name} dataset is empty")
            return df
        
        # Check for missing files
        if 'filepath' in df.columns:
            missing_files = []
            for idx, row in df.iterrows():
                if not Path(row['filepath']).exists():
                    missing_files.append(row['filepath'])
            
            if missing_files:
                logger.warning(f"Found {len(missing_files)} missing files in {dataset_name}")
                df = df[~df['filepath'].isin(missing_files)]
        
        logger.info(f"{dataset_name} validation complete. {len(df)} valid samples.")
        return df
    
    def save_metadata(self, df: pd.DataFrame, dataset_name: str):
        """
        Save preprocessed metadata to CSV.
        
        Args:
            df: DataFrame with dataset information
            dataset_name: Name of the dataset
        """
        output_path = self.output_dir / f"{dataset_name}_metadata.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Metadata saved to {output_path}")
    
    def run_all(self):
        """
        Run preprocessing for IEMOCAP dataset only.
        """
        logger.info("Starting preprocessing for IEMOCAP dataset...")
        
        # Process IEMOCAP with 5% subset
        df = self.preprocess_iemocap(subset_percent=5.0)
        df = self.validate_dataset(df, 'iemocap')
        
        if not df.empty:
            self.save_metadata(df, 'iemocap')
            logger.info(f"✓ Successfully preprocessed {len(df)} IEMOCAP samples")
        else:
            logger.error("✗ No IEMOCAP samples were processed")
        
        logger.info("Preprocessing complete!")


if __name__ == "__main__":
    
    # Initialize and run preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.run_all()
