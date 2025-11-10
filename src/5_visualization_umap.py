"""
UMAP Visualization Script
==========================

This script generates 2D visualization of speech embeddings using UMAP
for emotion recognition.

Based on the IEEE/ACM 2024 paper:
'From Raw Speech to Fixed Representations: A Comprehensive Evaluation 
of Speech Embedding Techniques'

Author: AI/ML Team
Date: 2024
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import umap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class UMAPVisualizer:
    """
    Visualize speech embeddings using UMAP dimensionality reduction.
    
    Creates 2D plots to analyze embedding space structure for emotions.
    """
    
    def __init__(
        self,
        embeddings_dir: str = "../embeddings",
        results_dir: str = "../results",
        random_state: int = 42
    ):
        """
        Initialize the UMAP visualizer.
        
        Args:
            embeddings_dir: Directory containing embeddings
            results_dir: Directory to save visualizations
            random_state: Random seed for reproducibility
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        logger.info("UMAPVisualizer initialized")
    
    def load_embeddings(
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
        
        return embeddings, labels
    
    def fit_umap(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Apply UMAP dimensionality reduction.
        
        Args:
            embeddings: High-dimensional embeddings
            n_components: Number of dimensions to reduce to (2 for visualization)
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance between points in low-dim space
            metric: Distance metric to use
            
        Returns:
            Reduced embeddings
        """
        logger.info(f"Fitting UMAP with {n_components} components...")
        logger.info(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=self.random_state,
            verbose=True
        )
        
        embeddings_reduced = reducer.fit_transform(embeddings)
        
        logger.info(f"UMAP reduction complete. Shape: {embeddings_reduced.shape}")
        
        return embeddings_reduced
    
    def plot_2d(
        self,
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        dataset_name: str,
        save_name: str = "umap_emotion"
    ):
        """
        Create 2D scatter plot of embeddings.
        
        Args:
            embeddings_2d: 2D embeddings
            labels: Labels for coloring
            dataset_name: Name of the dataset
            save_name: Filename for saving (without extension)
        """
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        # Define color palette for emotions
        n_classes = len(label_encoder.classes_)
        colors = sns.color_palette("husl", n_classes)
        
        # Create scatter plot
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels_encoded,
            cmap='Set2',
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add legend
        handles = []
        for idx, class_name in enumerate(label_encoder.classes_):
            handles.append(plt.scatter([], [], c=[colors[idx]], label=class_name, 
                                     s=100, edgecolors='black', linewidth=0.5))
        plt.legend(handles=handles, title='Emotion', loc='best', fontsize=12, 
                  title_fontsize=13, frameon=True, shadow=True)
        
        # Labels and title
        plt.xlabel('UMAP Dimension 1', fontsize=14, fontweight='bold')
        plt.ylabel('UMAP Dimension 2', fontsize=14, fontweight='bold')
        plt.title('UMAP Visualization of Emotion Embeddings', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save plot
        output_path = self.results_dir / f"{save_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"2D plot saved to {output_path}")
    
    def visualize_dataset(
        self,
        dataset_name: str = "emotion"
    ):
        """
        Create visualization for emotion dataset.
        
        Args:
            dataset_name: Name of the dataset
        """
        logger.info(f"Visualizing {dataset_name}...")
        
        # Load embeddings
        embeddings, labels = self.load_embeddings(dataset_name)
        
        # Create 2D visualization
        embeddings_2d = self.fit_umap(embeddings, n_components=2)
        self.plot_2d(embeddings_2d, labels, dataset_name)
        
        logger.info(f"Visualization complete for {dataset_name}")
    
    def run_visualization(self):
        """
        Create UMAP visualization for emotion embeddings.
        """
        logger.info("Starting UMAP visualization...")
        
        try:
            self.visualize_dataset('emotion')
            logger.info("\n✓ Visualization complete!")
            logger.info(f"  - UMAP plot saved to: results/umap_emotion.png")
        except Exception as e:
            logger.error(f"✗ Visualization failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    visualizer = UMAPVisualizer()
    visualizer.run_visualization()
