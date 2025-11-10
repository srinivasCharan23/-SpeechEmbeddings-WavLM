"""
Evaluation Metrics Script
==========================

This script computes comprehensive evaluation metrics for the trained classifiers.
Includes various performance metrics and generates detailed reports for analysis.

Metrics computed:
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- Per-class performance metrics
- Cross-dataset comparisons

Based on the IEEE/ACM 2024 paper:
'From Raw Speech to Fixed Representations: A Comprehensive Evaluation 
of Speech Embedding Techniques'

Author: AI/ML Team
Date: 2024

TODO (Teammate C - Evaluator): Main responsibilities
- Implement additional metrics (ROC-AUC, PR curves for binary tasks like Gender ID)
- Create detailed per-class analysis reports
- Compare performance across different datasets and tasks
- Generate statistical significance tests between classifiers
- Create comprehensive benchmark tables
- Analyze error patterns and failure cases
- Export results in multiple formats (CSV, JSON, LaTeX tables)
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Comprehensive evaluation of trained classifiers.
    
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
        
        logger.info("EvaluationMetrics initialized")
    
    def load_model_and_preprocessors(
        self,
        dataset_name: str,
        classifier_type: str
    ) -> Tuple[Any, Any, Any]:
        """
        Load trained model and preprocessors.
        
        Args:
            dataset_name: Name of the dataset
            classifier_type: Type of classifier
            
        Returns:
            Tuple of (model, scaler, label_encoder)
        """
        model_path = self.models_dir / f"{dataset_name}_{classifier_type}.pkl"
        scaler_path = self.models_dir / f"{dataset_name}_scaler.pkl"
        encoder_path = self.models_dir / f"{dataset_name}_encoder.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        
        return model, scaler, encoder
    
    def load_test_data(
        self,
        dataset_name: str,
        scaler: Any,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess test data.
        
        Args:
            dataset_name: Name of the dataset
            scaler: Fitted scaler for normalization
            test_size: Proportion of data used for testing
            
        Returns:
            Tuple of (X_test, y_test)
        """
        embeddings_path = self.embeddings_dir / f"{dataset_name}_embeddings.npy"
        labels_path = self.embeddings_dir / f"{dataset_name}_labels.npy"
        
        embeddings = np.load(embeddings_path)
        labels = np.load(labels_path)
        
        # Split data (same as in training)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        encoder = LabelEncoder()
        labels_encoded = encoder.fit_transform(labels)
        
        _, X_test, _, y_test = train_test_split(
            embeddings, labels_encoded,
            test_size=test_size,
            random_state=42,
            stratify=labels_encoded
        )
        
        X_test = scaler.transform(X_test)
        
        return X_test, y_test
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy for multi-class metrics
            
        Returns:
            Dictionary of metric values
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        dataset_name: str,
        classifier_type: str
    ):
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            dataset_name: Name of the dataset
            classifier_type: Type of classifier
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Confusion Matrix - {dataset_name} ({classifier_type})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        output_path = self.results_dir / f"cm_{dataset_name}_{classifier_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        dataset_name: str,
        classifier_type: str
    ):
        """
        Generate and save detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            dataset_name: Name of the dataset
            classifier_type: Type of classifier
        """
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True
        )
        
        # Convert to DataFrame for better visualization
        df_report = pd.DataFrame(report).transpose()
        
        # Save to CSV
        output_path = self.results_dir / f"report_{dataset_name}_{classifier_type}.csv"
        df_report.to_csv(output_path)
        
        logger.info(f"Classification report saved to {output_path}")
        
        return report
    
    def evaluate_model(
        self,
        dataset_name: str,
        classifier_type: str
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            dataset_name: Name of the dataset
            classifier_type: Type of classifier
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating {classifier_type} on {dataset_name}")
        
        # Load model and data
        model, scaler, encoder = self.load_model_and_preprocessors(dataset_name, classifier_type)
        X_test, y_test = self.load_test_data(dataset_name, scaler)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Compute metrics
        metrics = self.compute_metrics(y_test, y_pred)
        
        # Generate visualizations
        class_names = encoder.classes_
        self.plot_confusion_matrix(y_test, y_pred, class_names, dataset_name, classifier_type)
        report = self.generate_classification_report(y_test, y_pred, class_names, dataset_name, classifier_type)
        
        # Log results
        logger.info(f"\nMetrics for {dataset_name} - {classifier_type}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return {
            'metrics': metrics,
            'classification_report': report,
            'num_samples': len(y_test),
            'num_classes': len(class_names)
        }
    
    def compare_classifiers(
        self,
        dataset_name: str,
        classifier_types: List[str] = ['svm', 'rf', 'mlp', 'lr']
    ) -> pd.DataFrame:
        """
        Compare performance of different classifiers on a dataset.
        
        Args:
            dataset_name: Name of the dataset
            classifier_types: List of classifier types to compare
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"\nComparing classifiers on {dataset_name}")
        
        results = []
        for clf_type in classifier_types:
            try:
                eval_results = self.evaluate_model(dataset_name, clf_type)
                
                result_row = {
                    'dataset': dataset_name,
                    'classifier': clf_type,
                    **eval_results['metrics']
                }
                results.append(result_row)
            except Exception as e:
                logger.warning(f"Failed to evaluate {clf_type} on {dataset_name}: {e}")
        
        df_results = pd.DataFrame(results)
        
        # Save comparison
        output_path = self.results_dir / f"comparison_{dataset_name}.csv"
        df_results.to_csv(output_path, index=False)
        logger.info(f"Comparison results saved to {output_path}")
        
        return df_results
    
    def plot_classifier_comparison(
        self,
        df_results: pd.DataFrame,
        metric: str = 'accuracy'
    ):
        """
        Plot comparison of classifiers.
        
        Args:
            df_results: DataFrame with comparison results
            metric: Metric to plot
        """
        plt.figure(figsize=(12, 6))
        
        datasets = df_results['dataset'].unique()
        classifiers = df_results['classifier'].unique()
        
        x = np.arange(len(datasets))
        width = 0.2
        
        for i, clf in enumerate(classifiers):
            clf_data = df_results[df_results['classifier'] == clf]
            values = [clf_data[clf_data['dataset'] == ds][metric].values[0] 
                     if len(clf_data[clf_data['dataset'] == ds]) > 0 else 0 
                     for ds in datasets]
            plt.bar(x + i * width, values, width, label=clf)
        
        plt.xlabel('Dataset')
        plt.ylabel(metric.capitalize())
        plt.title(f'Classifier Comparison - {metric.capitalize()}')
        plt.xticks(x + width * 1.5, datasets)
        plt.legend()
        plt.tight_layout()
        
        output_path = self.results_dir / f"comparison_{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved to {output_path}")
    
    def run_full_evaluation(
        self,
        datasets: List[str] = ['iemocap', 'librispeech', 'slurp', 'commonvoice'],
        classifier_types: List[str] = ['svm', 'rf', 'mlp', 'lr']
    ):
        """
        Run comprehensive evaluation on all datasets and classifiers.
        
        Args:
            datasets: List of dataset names
            classifier_types: List of classifier types
        """
        logger.info("Starting full evaluation...")
        
        all_results = []
        for dataset in datasets:
            try:
                df_results = self.compare_classifiers(dataset, classifier_types)
                all_results.append(df_results)
            except Exception as e:
                logger.warning(f"Failed to evaluate {dataset}: {e}")
        
        if all_results:
            # Combine all results
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Save combined results
            output_path = self.results_dir / "all_results.csv"
            combined_results.to_csv(output_path, index=False)
            logger.info(f"All results saved to {output_path}")
            
            # Generate comparison plots
            for metric in ['accuracy', 'f1_macro', 'f1_weighted']:
                self.plot_classifier_comparison(combined_results, metric)
        
        logger.info("Full evaluation complete!")


if __name__ == "__main__":
    evaluator = EvaluationMetrics()
    evaluator.run_full_evaluation()
