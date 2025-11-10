"""
Evaluation Metrics Script
==========================

This script computes comprehensive evaluation metrics for the trained MLP classifier.
Generates metrics.json and confusion_matrix.png for emotion recognition.

Based on the IEEE/ACM 2024 paper:
'From Raw Speech to Fixed Representations: A Comprehensive Evaluation 
of Speech Embedding Techniques'

Author: AI/ML Team
Date: 2024
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

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


class EvaluationMetrics:
    """
    Comprehensive evaluation of trained emotion classifier.
    
    Computes various performance metrics and generates visualizations.
    """
    
    def __init__(
        self,
        models_dir: str = "../models",
        embeddings_dir: str = "../embeddings",
        results_dir: str = "../results"
    ):
        """
        Initialize the evaluation metrics calculator.
        
        Args:
            models_dir: Directory containing trained models
            embeddings_dir: Directory containing embeddings
            results_dir: Directory to save evaluation results
        """
        self.models_dir = Path(models_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cpu")
        
        logger.info("EvaluationMetrics initialized (CPU mode)")
    
    def load_model_and_preprocessors(
        self,
        dataset_name: str = "emotion"
    ) -> Tuple[nn.Module, Any, Any]:
        """
        Load trained model and preprocessors.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (model, scaler, label_encoder)
        """
        model_path = self.models_dir / f"{dataset_name}_model.pt"
        scaler_path = self.models_dir / f"{dataset_name}_scaler.pkl"
        encoder_path = self.models_dir / f"{dataset_name}_encoder.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Recreate model architecture
        arch = checkpoint['model_architecture']
        model = SimpleMLP(arch['input_dim'], arch['hidden_dim'], arch['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Load preprocessors
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        
        logger.info(f"Loaded model from {model_path}")
        
        return model, scaler, encoder
    
    def load_test_data(
        self,
        dataset_name: str,
        scaler: Any,
        encoder: Any,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess test data.
        
        Args:
            dataset_name: Name of the dataset
            scaler: Fitted scaler for normalization
            encoder: Label encoder
            test_size: Proportion of data used for testing
            
        Returns:
            Tuple of (X_test, y_test)
        """
        embeddings_path = self.embeddings_dir / f"{dataset_name}_embeddings.npz"
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
        
        # Load data
        data = np.load(embeddings_path)
        embeddings = data['embeddings']
        labels = data['labels']
        
        # Encode labels
        labels_encoded = encoder.transform(labels)
        
        # Split data (same as in training)
        _, X_test, _, y_test = train_test_split(
            embeddings, labels_encoded,
            test_size=test_size,
            random_state=42,
            stratify=labels_encoded
        )
        
        # Normalize
        X_test = scaler.transform(X_test)
        
        return X_test, y_test
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metric values
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'cohen_kappa': float(cohen_kappa_score(y_true, y_pred))
        }
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_name: str = "confusion_matrix"
    ):
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_name: Filename for saving (without extension)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Emotion Classification', fontsize=14, fontweight='bold')
        plt.ylabel('True Emotion', fontsize=12)
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.tight_layout()
        
        output_path = self.results_dir / f"{save_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
    
    def evaluate_model(
        self,
        dataset_name: str = "emotion"
    ) -> Dict[str, Any]:
        """
        Evaluate trained emotion model.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating {dataset_name} model...")
        
        # Load model and data
        model, scaler, encoder = self.load_model_and_preprocessors(dataset_name)
        X_test, y_test = self.load_test_data(dataset_name, scaler, encoder)
        
        # Make predictions
        X_test_tensor = torch.FloatTensor(X_test)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
        
        y_pred = predicted.numpy()
        
        # Compute metrics
        metrics = self.compute_metrics(y_test, y_pred)
        
        # Generate visualizations
        class_names = encoder.classes_
        self.plot_confusion_matrix(y_test, y_pred, class_names)
        
        # Generate classification report
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True
        )
        
        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation Results for {dataset_name.upper()}")
        logger.info(f"{'='*60}")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        logger.info(f"{'='*60}")
        
        # Save metrics to JSON
        metrics_path = self.results_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        return {
            'metrics': metrics,
            'classification_report': report,
            'num_samples': len(y_test),
            'num_classes': len(class_names),
            'classes': list(class_names)
        }
    
    def run_evaluation(self):
        """
        Run comprehensive evaluation for emotion model.
        """
        logger.info("Starting evaluation...")
        
        try:
            results = self.evaluate_model('emotion')
            logger.info("\n✓ Evaluation complete!")
            logger.info(f"  - Metrics saved to: results/metrics.json")
            logger.info(f"  - Confusion matrix saved to: results/confusion_matrix.png")
            return results
        except Exception as e:
            logger.error(f"✗ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    evaluator = EvaluationMetrics()
    evaluator.run_evaluation()
