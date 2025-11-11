"""
WavLM Feature Extraction Script
=================================

This script extracts fixed-dimensional speech embeddings using the WavLM-base model.
WavLM is a self-supervised learning model pre-trained on large-scale unlabeled speech data.

The extracted embeddings can be used for various downstream tasks:
- Emotion Identification
- Gender Identification
- Intent Identification
- Cross-language Embeddings

Based on the IEEE/ACM 2024 paper:
'From Raw Speech to Fixed Representations: A Comprehensive Evaluation 
of Speech Embedding Techniques'

Author: AI/ML Team
Date: 2024

TODO (Inthiyaz - Model Architect): Main responsibilities
- Experiment with different WavLM model variants (base, base-plus, large)
- Implement multi-layer feature fusion strategies
- Optimize pooling methods (mean, max, attention-based)
- Add audio augmentation techniques for robustness
- Implement efficient batch processing for large datasets
- Design the embedding extraction pipeline architecture
- Compare WavLM with other models (Wav2Vec2, HuBERT)
- Optimize model loading and inference for GPU/CPU
"""

import os
import logging
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from transformers import Wav2Vec2Processor, WavLMModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WavLMFeatureExtractor:
    """
    Extract speech embeddings using WavLM-base model.
    
    The WavLM model generates contextual representations from raw audio.
    We extract features from multiple layers and use pooling strategies
    to create fixed-dimensional embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base",
        device: Optional[str] = None,
        embeddings_dir: str = "../embeddings"
    ):
        """
        Initialize the WavLM feature extractor.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on (cuda/cpu)
            embeddings_dir: Directory to save extracted embeddings
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading WavLM model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("WavLM model loaded successfully")
    
    def load_audio(self, filepath: str, target_sr: int = 16000) -> torch.Tensor:
        """
        Load audio file and resample to target sampling rate.
        
        Args:
            filepath: Path to audio file
            target_sr: Target sampling rate (WavLM expects 16kHz)
            
        Returns:
            Audio tensor
        """
        try:
            waveform, sample_rate = torchaudio.load(filepath)
            
            # Resample if necessary
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            return waveform.squeeze()
        
        except Exception as e:
            logger.error(f"Error loading audio {filepath}: {e}")
            raise
    
    def extract_embedding(
        self,
        audio: torch.Tensor,
        layer: int = -1,
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Extract embedding from audio using WavLM.
        
        Args:
            audio: Audio tensor (1D)
            layer: Which transformer layer to use (-1 for last layer)
            pooling: Pooling strategy ('mean', 'max', 'first', 'last')
            
        Returns:
            Fixed-dimensional embedding as numpy array
        """
        try:
            # Process audio
            inputs = self.processor(
                audio.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get hidden states from specified layer
            hidden_states = outputs.hidden_states[layer]  # Shape: (batch, time, hidden_dim)
            
            # Apply pooling
            if pooling == "mean":
                embedding = hidden_states.mean(dim=1)
            elif pooling == "max":
                embedding = hidden_states.max(dim=1)[0]
            elif pooling == "first":
                embedding = hidden_states[:, 0, :]
            elif pooling == "last":
                embedding = hidden_states[:, -1, :]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")
            
            # Convert to numpy
            embedding = embedding.cpu().numpy().squeeze()
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            raise
    
    def extract_from_file(
        self,
        filepath: str,
        layer: int = -1,
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Extract embedding from audio file.
        
        Args:
            filepath: Path to audio file
            layer: Which transformer layer to use
            pooling: Pooling strategy
            
        Returns:
            Embedding as numpy array
        """
        audio = self.load_audio(filepath)
        return self.extract_embedding(audio, layer=layer, pooling=pooling)
    
    def process_dataset(
        self,
        metadata_path: str,
        dataset_name: str,
        layer: int = -1,
        pooling: str = "mean",
        batch_size: int = 8
    ):
        """
        Process entire dataset and save embeddings.
        Optimized for CPU-only processing with batch processing.
        
        Args:
            metadata_path: Path to metadata CSV file
            dataset_name: Name of the dataset
            layer: Which transformer layer to use
            pooling: Pooling strategy
            batch_size: Batch size for processing (reduced for CPU)
        """
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"Device: {self.device} | Batch size: {batch_size}")
        
        # Load metadata
        df = pd.read_csv(metadata_path)
        logger.info(f"Loaded {len(df)} samples from {metadata_path}")
        
        embeddings = []
        labels = []
        failed_count = 0
        
        # Process each audio file with progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {dataset_name}"):
            try:
                filepath = row['filepath']
                
                # Skip if file doesn't exist
                if not Path(filepath).exists():
                    logger.warning(f"File not found: {filepath}")
                    failed_count += 1
                    continue
                
                embedding = self.extract_from_file(filepath, layer=layer, pooling=pooling)
                embeddings.append(embedding)
                
                # Store label (dataset-specific)
                if 'emotion' in row:
                    labels.append(row['emotion'])
                elif 'speaker_id' in row:
                    labels.append(row['speaker_id'])
                elif 'intent' in row:
                    labels.append(row['intent'])
                elif 'language' in row:
                    labels.append(row['language'])
                else:
                    labels.append(None)
                
                # Clear CUDA cache periodically if using GPU (not applicable for CPU)
                if (idx + 1) % batch_size == 0:
                    import gc
                    gc.collect()
                
            except Exception as e:
                logger.warning(f"Skipping {filepath}: {e}")
                failed_count += 1
                continue
        
        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} files")
        
        # Convert to numpy arrays
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        # Save embeddings
        output_path = self.embeddings_dir / f"{dataset_name}_embeddings.npy"
        labels_path = self.embeddings_dir / f"{dataset_name}_labels.npy"
        
        np.save(output_path, embeddings)
        np.save(labels_path, labels)
        
        logger.info(f"✅ Saved embeddings to {output_path}")
        logger.info(f"✅ Saved labels to {labels_path}")
        logger.info(f"Embedding shape: {embeddings.shape}")
        logger.info(f"Successfully processed: {len(embeddings)}/{len(df)} samples")
    
    def extract_multi_layer(
        self,
        filepath: str,
        layers: List[int] = [-4, -3, -2, -1],
        pooling: str = "mean"
    ) -> Dict[int, np.ndarray]:
        """
        Extract embeddings from multiple layers.
        
        Args:
            filepath: Path to audio file
            layers: List of layer indices to extract from
            pooling: Pooling strategy
            
        Returns:
            Dictionary mapping layer indices to embeddings
        """
        audio = self.load_audio(filepath)
        
        embeddings = {}
        for layer in layers:
            embeddings[layer] = self.extract_embedding(audio, layer=layer, pooling=pooling)
        
        return embeddings


if __name__ == "__main__":
    # Initialize extractor with CPU-only mode (optimized for Codespaces)
    extractor = WavLMFeatureExtractor(device="cpu")
    
    # Sample feature extraction function
    def sample_feature_extraction(audio_file_path: str):
        """
        Sample feature extraction function demonstrating how to use WavLM.
        
        This function shows a complete workflow of:
        1. Loading an audio file
        2. Extracting embeddings using WavLM-base
        3. Returning a fixed-dimensional representation
        
        Args:
            audio_file_path: Path to the audio file (.wav, .flac, etc.)
            
        Returns:
            numpy array: 768-dimensional embedding vector
            
        Example:
            >>> embedding = sample_feature_extraction("sample_audio.wav")
            >>> print(f"Embedding shape: {embedding.shape}")
            Embedding shape: (768,)
        """
        logger.info(f"Extracting features from: {audio_file_path}")
        
        # Extract embedding with default settings
        # - Uses last layer (-1) of WavLM transformer
        # - Applies mean pooling over time dimension
        embedding = extractor.extract_from_file(
            filepath=audio_file_path,
            layer=-1,        # Last transformer layer
            pooling="mean"   # Mean pooling strategy
        )
        
        logger.info(f"Extraction complete. Embedding dimension: {embedding.shape[0]}")
        return embedding
    
    # Process CREMA-D dataset (primary focus)
    cremad_metadata = '../data/processed/cremad_subset.csv'
    if Path(cremad_metadata).exists():
        logger.info("Processing CREMA-D dataset...")
        extractor.process_dataset(cremad_metadata, 'cremad')
        
        # Save as emotion_embeddings.npz for compatibility
        embeddings_npy = extractor.embeddings_dir / "cremad_embeddings.npy"
        labels_npy = extractor.embeddings_dir / "cremad_labels.npy"
        
        if embeddings_npy.exists() and labels_npy.exists():
            embeddings = np.load(embeddings_npy)
            labels = np.load(labels_npy)
            
            # Save as .npz format
            npz_path = extractor.embeddings_dir / "emotion_embeddings.npz"
            np.savez(npz_path, embeddings=embeddings, labels=labels)
            logger.info(f"✅ Also saved as {npz_path}")
    else:
        logger.warning(f"CREMA-D metadata not found: {cremad_metadata}")
        logger.warning("Please run 1_data_preprocessing.py first")
    
    # Process other datasets (backward compatibility)
    datasets = {
        'iemocap': '../data/processed/iemocap_metadata.csv',
        'librispeech': '../data/processed/librispeech_metadata.csv',
        'slurp': '../data/processed/slurp_metadata.csv',
        'commonvoice': '../data/processed/commonvoice_metadata.csv',
        'ravdess': '../data/processed/ravdess_subset.csv'
    }
    
    for dataset_name, metadata_path in datasets.items():
        if Path(metadata_path).exists():
            logger.info(f"Processing {dataset_name} dataset...")
            extractor.process_dataset(metadata_path, dataset_name)
        else:
            logger.debug(f"Metadata file not found: {metadata_path}")
    
    logger.info("Feature extraction complete!")
