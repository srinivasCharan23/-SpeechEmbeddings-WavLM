"""
Classifier Training Script
===========================

This script trains various classifiers on the extracted WavLM embeddings
for different downstream tasks:
- Emotion recognition (IEMOCAP)
- Speaker identification (LibriSpeech)
- Intent classification (SLURP)
- Language/accent classification (CommonVoice)

Implements multiple classifier architectures:
- Support Vector Machines (SVM)
- Random Forest
- Multi-Layer Perceptron (MLP)
- Logistic Regression

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
from pathlib import Path
from typing import Dict, Tuple, Any, List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClassifierTrainer:
    """
    Train and evaluate classifiers on speech embeddings.
    
    Supports multiple classifier types and hyperparameter tuning.
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
        
        logger.info(f"ClassifierTrainer initialized")
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
        embeddings_path = self.embeddings_dir / f"{dataset_name}_embeddings.npy"
        labels_path = self.embeddings_dir / f"{dataset_name}_labels.npy"
        
        if not embeddings_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Embeddings or labels not found for {dataset_name}")
        
        embeddings = np.load(embeddings_path)
        labels = np.load(labels_path)
        
        logger.info(f"Loaded {dataset_name}: {embeddings.shape[0]} samples, "
                   f"embedding dim: {embeddings.shape[1]}")
        logger.info(f"Unique labels: {len(np.unique(labels))}")
        
        return embeddings, labels
    
    def prepare_data(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict[str, np.ndarray]:
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
    
    def get_classifier(self, classifier_type: str) -> Any:
        """
        Get classifier instance with default hyperparameters.
        
        Args:
            classifier_type: Type of classifier ('svm', 'rf', 'mlp', 'lr')
            
        Returns:
            Classifier instance
        """
        classifiers = {
            'svm': SVC(kernel='rbf', random_state=self.random_state),
            'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True
            ),
            'lr': LogisticRegression(max_iter=1000, random_state=self.random_state)
        }
        
        if classifier_type not in classifiers:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        return classifiers[classifier_type]
    
    def train_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        classifier_type: str,
        use_grid_search: bool = False
    ) -> Any:
        """
        Train a classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            classifier_type: Type of classifier
            use_grid_search: Whether to use grid search for hyperparameter tuning
            
        Returns:
            Trained classifier
        """
        logger.info(f"Training {classifier_type} classifier...")
        
        clf = self.get_classifier(classifier_type)
        
        if use_grid_search:
            # Define parameter grids for each classifier
            param_grids = {
                'svm': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
                'rf': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]},
                'mlp': {'hidden_layer_sizes': [(128,), (256, 128), (512, 256)]},
                'lr': {'C': [0.1, 1, 10]}
            }
            
            grid_search = GridSearchCV(
                clf,
                param_grids[classifier_type],
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            clf = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            clf.fit(X_train, y_train)
        
        logger.info(f"{classifier_type} training complete")
        return clf
    
    def evaluate_classifier(
        self,
        clf: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str,
        classifier_type: str
    ) -> Dict[str, float]:
        """
        Evaluate classifier performance.
        
        Args:
            clf: Trained classifier
            X_test: Test features
            y_test: Test labels
            dataset_name: Name of the dataset
            classifier_type: Type of classifier
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f"\n{dataset_name} - {classifier_type} Results:")
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
            'report': report
        }
    
    def save_model(
        self,
        clf: Any,
        dataset_name: str,
        classifier_type: str
    ):
        """
        Save trained model to disk.
        
        Args:
            clf: Trained classifier
            dataset_name: Name of the dataset
            classifier_type: Type of classifier
        """
        model_path = self.models_dir / f"{dataset_name}_{classifier_type}.pkl"
        scaler_path = self.models_dir / f"{dataset_name}_scaler.pkl"
        encoder_path = self.models_dir / f"{dataset_name}_encoder.pkl"
        
        joblib.dump(clf, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def train_all_classifiers(
        self,
        dataset_name: str,
        classifier_types: List[str] = ['svm', 'rf', 'mlp', 'lr']
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all classifier types on a dataset.
        
        Args:
            dataset_name: Name of the dataset
            classifier_types: List of classifier types to train
            
        Returns:
            Dictionary of results for each classifier
        """
        # Load data
        embeddings, labels = self.load_dataset(dataset_name)
        data_splits = self.prepare_data(embeddings, labels)
        
        results = {}
        
        # Train each classifier
        for clf_type in classifier_types:
            clf = self.train_classifier(
                data_splits['X_train'],
                data_splits['y_train'],
                clf_type
            )
            
            # Evaluate
            metrics = self.evaluate_classifier(
                clf,
                data_splits['X_test'],
                data_splits['y_test'],
                dataset_name,
                clf_type
            )
            
            # Save model
            self.save_model(clf, dataset_name, clf_type)
            
            results[clf_type] = metrics
        
        return results


if __name__ == "__main__":
    trainer = ClassifierTrainer()
    
    # Train classifiers on all datasets
    datasets = ['iemocap', 'librispeech', 'slurp', 'commonvoice']
    
    all_results = {}
    for dataset in datasets:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training classifiers for {dataset}")
            logger.info(f"{'='*60}")
            
            results = trainer.train_all_classifiers(dataset)
            all_results[dataset] = results
        except FileNotFoundError as e:
            logger.warning(f"Skipping {dataset}: {e}")
        except Exception as e:
            logger.error(f"Error processing {dataset}: {e}")
    
    logger.info("\nTraining complete!")
