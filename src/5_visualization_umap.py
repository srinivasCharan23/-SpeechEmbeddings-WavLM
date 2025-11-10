"""
UMAP Visualization Script
==========================

This script generates 2D and 3D visualizations of speech embeddings using UMAP
(Uniform Manifold Approximation and Projection) for dimensionality reduction.

Visualizations help understand:
- Cluster formation in embedding space
- Class separability
- Dataset characteristics
- Embedding quality

Based on the IEEE/ACM 2024 paper:
'From Raw Speech to Fixed Representations: A Comprehensive Evaluation 
of Speech Embedding Techniques'

Author: AI/ML Team
Date: 2024

TODO (Teammate D - Visualizer): Main responsibilities
- Create interactive 3D visualizations (plotly)
- Generate t-SNE visualizations for comparison with UMAP
- Plot decision boundaries for classifiers
- Create animated visualizations showing embedding evolution
- Generate comparison grids for all four tasks
- Design publication-quality figures with proper styling
- Create dashboard-style summary visualizations
- Export high-resolution figures for papers/presentations
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
    
    Creates 2D and 3D plots to analyze embedding space structure.
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
        embeddings_path = self.embeddings_dir / f"{dataset_name}_embeddings.npy"
        labels_path = self.embeddings_dir / f"{dataset_name}_labels.npy"
        
        if not embeddings_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Embeddings or labels not found for {dataset_name}")
        
        embeddings = np.load(embeddings_path)
        labels = np.load(labels_path)
        
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
            n_components: Number of dimensions to reduce to (2 or 3)
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance between points in low-dim space
            metric: Distance metric to use
            
        Returns:
            Reduced embeddings
        """
        logger.info(f"Fitting UMAP with {n_components} components...")
        
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
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ):
        """
        Create 2D scatter plot of embeddings.
        
        Args:
            embeddings_2d: 2D embeddings
            labels: Labels for coloring
            dataset_name: Name of the dataset
            title: Custom title for plot
            save_name: Custom filename for saving
        """
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        # Create scatter plot
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels_encoded,
            cmap='tab10',
            alpha=0.6,
            s=20,
            edgecolors='k',
            linewidth=0.5
        )
        
        # Add colorbar with class names
        cbar = plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)))
        cbar.ax.set_yticklabels(label_encoder.classes_)
        cbar.set_label('Class', rotation=270, labelpad=20)
        
        # Labels and title
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        
        if title:
            plt.title(title, fontsize=14, fontweight='bold')
        else:
            plt.title(f'UMAP Visualization - {dataset_name}', fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if save_name:
            output_path = self.results_dir / f"{save_name}.png"
        else:
            output_path = self.results_dir / f"umap_2d_{dataset_name}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"2D plot saved to {output_path}")
    
    def plot_3d(
        self,
        embeddings_3d: np.ndarray,
        labels: np.ndarray,
        dataset_name: str,
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ):
        """
        Create 3D scatter plot of embeddings.
        
        Args:
            embeddings_3d: 3D embeddings
            labels: Labels for coloring
            dataset_name: Name of the dataset
            title: Custom title for plot
            save_name: Custom filename for saving
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot
        scatter = ax.scatter(
            embeddings_3d[:, 0],
            embeddings_3d[:, 1],
            embeddings_3d[:, 2],
            c=labels_encoded,
            cmap='tab10',
            alpha=0.6,
            s=20,
            edgecolors='k',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(label_encoder.classes_)), pad=0.1)
        cbar.ax.set_yticklabels(label_encoder.classes_)
        cbar.set_label('Class', rotation=270, labelpad=20)
        
        # Labels and title
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_zlabel('UMAP Dimension 3', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'UMAP 3D Visualization - {dataset_name}', fontsize=14, fontweight='bold')
        
        # Save plot
        if save_name:
            output_path = self.results_dir / f"{save_name}.png"
        else:
            output_path = self.results_dir / f"umap_3d_{dataset_name}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"3D plot saved to {output_path}")
    
    def create_grid_plot(
        self,
        datasets: List[str],
        n_components: int = 2
    ):
        """
        Create grid of UMAP plots for multiple datasets.
        
        Args:
            datasets: List of dataset names
            n_components: Number of UMAP dimensions (2 or 3)
        """
        n_datasets = len(datasets)
        n_cols = 2
        n_rows = (n_datasets + 1) // 2
        
        fig = plt.figure(figsize=(20, 10 * n_rows))
        
        for idx, dataset in enumerate(datasets, 1):
            try:
                # Load embeddings
                embeddings, labels = self.load_embeddings(dataset)
                
                # Apply UMAP
                embeddings_reduced = self.fit_umap(embeddings, n_components=n_components)
                
                # Create subplot
                if n_components == 2:
                    ax = fig.add_subplot(n_rows, n_cols, idx)
                    
                    label_encoder = LabelEncoder()
                    labels_encoded = label_encoder.fit_transform(labels)
                    
                    scatter = ax.scatter(
                        embeddings_reduced[:, 0],
                        embeddings_reduced[:, 1],
                        c=labels_encoded,
                        cmap='tab10',
                        alpha=0.6,
                        s=10
                    )
                    
                    ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('UMAP Dimension 1')
                    ax.set_ylabel('UMAP Dimension 2')
                    ax.grid(True, alpha=0.3)
                
            except Exception as e:
                logger.warning(f"Failed to create plot for {dataset}: {e}")
        
        plt.tight_layout()
        output_path = self.results_dir / f"umap_grid_{n_components}d.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Grid plot saved to {output_path}")
    
    def visualize_dataset(
        self,
        dataset_name: str,
        create_2d: bool = True,
        create_3d: bool = True
    ):
        """
        Create visualizations for a single dataset.
        
        Args:
            dataset_name: Name of the dataset
            create_2d: Whether to create 2D plot
            create_3d: Whether to create 3D plot
        """
        logger.info(f"Visualizing {dataset_name}...")
        
        # Load embeddings
        embeddings, labels = self.load_embeddings(dataset_name)
        
        # Create 2D visualization
        if create_2d:
            embeddings_2d = self.fit_umap(embeddings, n_components=2)
            self.plot_2d(embeddings_2d, labels, dataset_name)
        
        # Create 3D visualization
        if create_3d:
            embeddings_3d = self.fit_umap(embeddings, n_components=3)
            self.plot_3d(embeddings_3d, labels, dataset_name)
        
        logger.info(f"Visualization complete for {dataset_name}")
    
    def visualize_all(
        self,
        datasets: List[str] = ['iemocap', 'librispeech', 'slurp', 'commonvoice']
    ):
        """
        Create visualizations for all datasets.
        
        Args:
            datasets: List of dataset names
        """
        logger.info("Creating visualizations for all datasets...")
        
        for dataset in datasets:
            try:
                self.visualize_dataset(dataset)
            except Exception as e:
                logger.warning(f"Failed to visualize {dataset}: {e}")
        
        # Create grid comparison
        try:
            self.create_grid_plot(datasets, n_components=2)
        except Exception as e:
            logger.warning(f"Failed to create grid plot: {e}")
        
        logger.info("All visualizations complete!")


if __name__ == "__main__":
    visualizer = UMAPVisualizer()
    visualizer.visualize_all()
