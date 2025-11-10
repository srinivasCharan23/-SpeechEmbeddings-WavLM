"""
Classifier Training Script
===========================

This script trains an MLP classifier on extracted WavLM embeddings
for emotion identification (CPU-optimized).

Based on the IEEE/ACM 2024 paper:
'From Raw Speech to Fixed Representations: A Comprehensive Evaluation 
of Speech Embedding Techniques'

Author: AI/ML Team
Date: 2024
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleMLP(nn.Module):
    """Simple MLP classifier for emotion recognition."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)


class ClassifierTrainer:
    """
    Train and evaluate MLP classifier on speech embeddings (CPU-optimized).
    """
    
    def __init__(
        self,
        embeddings_dir: str = "../embeddings",
        models_dir: str = "../models",
        random_state: int = 42
    ):
        """
        Initialize the classifier trainer.
        
        Args:
            embeddings_dir: Directory containing embeddings
            models_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Initialize label encoder and scaler
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Set device to CPU
        self.device = torch.device("cpu")
        
        logger.info(f"ClassifierTrainer initialized (CPU mode)")
        logger.info(f"Models will be saved to: {models_dir}")
    
    def load_dataset(
        self,
        dataset_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load embeddings and labels for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (embeddings, labels)
        """
        embeddings_path = self.embeddings_dir / f"{dataset_name}_embeddings.npz"
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
        
        # Load from NPZ file
        data = np.load(embeddings_path)
        embeddings = data['embeddings']
        labels = data['labels']
        
        logger.info(f"Loaded {dataset_name}: {embeddings.shape[0]} samples, "
                   f"embedding dim: {embeddings.shape[1]}")
        logger.info(f"Unique labels: {np.unique(labels)}")
        
        return embeddings, labels
    
    def prepare_data(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict[str, Any]:
        """
        Split and preprocess data for training.
        
        Args:
            embeddings: Feature embeddings
            labels: Target labels
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Dictionary with train/val/test splits
        """
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Split into train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            embeddings, labels_encoded,
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels_encoded
        )
        
        # Split train into train and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=y_trainval
        )
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def train_mlp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int,
        hidden_dim: int = 128,
        epochs: int = 50,
        lr: float = 0.001
    ) -> nn.Module:
        """
        Train MLP classifier (CPU-optimized).
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_classes: Number of emotion classes
            hidden_dim: Hidden layer dimension
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Trained MLP model
        """
        logger.info(f"Training MLP classifier...")
        logger.info(f"Hidden dim: {hidden_dim}, Epochs: {epochs}, LR: {lr}")
        
        # Get input dimension
        input_dim = X_train.shape[1]
        
        # Create model
        model = SimpleMLP(input_dim, hidden_dim, num_classes).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                _, predicted = torch.max(val_outputs, 1)
                val_acc = (predicted == y_val_tensor).float().mean().item()
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], "
                          f"Loss: {loss.item():.4f}, "
                          f"Val Loss: {val_loss.item():.4f}, "
                          f"Val Acc: {val_acc:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        logger.info(f"MLP training complete. Best val accuracy: {best_val_acc:.4f}")
        
        return model
    
    def evaluate_classifier(
        self,
        model: nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate classifier performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        X_test_tensor = torch.FloatTensor(X_test)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
        
        y_pred = predicted.numpy()
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        logger.info(f"\nEmotion Classification Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 (macro): {f1_macro:.4f}")
        logger.info(f"F1 (weighted): {f1_weighted:.4f}")
        
        # Generate classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        )
        logger.info(f"\nClassification Report:\n{report}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'y_pred': y_pred,
            'y_test': y_test
        }
    
    def save_model(
        self,
        model: nn.Module,
        dataset_name: str = "emotion"
    ):
        """
        Save trained model to disk as PyTorch model.
        
        Args:
            model: Trained model
            dataset_name: Name of the dataset
        """
        model_path = self.models_dir / f"{dataset_name}_model.pt"
        scaler_path = self.models_dir / f"{dataset_name}_scaler.pkl"
        encoder_path = self.models_dir / f"{dataset_name}_encoder.pkl"
        
        # Save PyTorch model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': {
                'input_dim': model.layers[0].in_features,
                'hidden_dim': model.layers[0].out_features,
                'num_classes': model.layers[-1].out_features
            }
        }, model_path)
        
        # Save preprocessors
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Encoder saved to {encoder_path}")
    
    def train_emotion_classifier(self) -> Dict[str, Any]:
        """
        Train emotion classifier on IEMOCAP embeddings.
        
        Returns:
            Dictionary of training results
        """
        logger.info("\n" + "="*60)
        logger.info("Training Emotion Classifier (MLP)")
        logger.info("="*60)
        
        # Load data
        embeddings, labels = self.load_dataset('emotion')
        data_splits = self.prepare_data(embeddings, labels)
        
        # Get number of classes
        num_classes = len(np.unique(data_splits['y_train']))
        logger.info(f"Number of emotion classes: {num_classes}")
        logger.info(f"Classes: {self.label_encoder.classes_}")
        
        # Train MLP
        model = self.train_mlp(
            data_splits['X_train'],
            data_splits['y_train'],
            data_splits['X_val'],
            data_splits['y_val'],
            num_classes,
            hidden_dim=128,
            epochs=50,
            lr=0.001
        )
        
        # Evaluate
        results = self.evaluate_classifier(
            model,
            data_splits['X_test'],
            data_splits['y_test']
        )
        
        # Save model
        self.save_model(model, 'emotion')
        
        return results


if __name__ == "__main__":
    trainer = ClassifierTrainer()
    
    try:
        results = trainer.train_emotion_classifier()
        logger.info("\n✓ Training complete!")
    except FileNotFoundError as e:
        logger.error(f"✗ {e}")
        logger.error("Please run feature extraction first (2_wavlm_feature_extraction.py)")
    except Exception as e:
        logger.error(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
