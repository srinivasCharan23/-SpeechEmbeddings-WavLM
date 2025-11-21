#!/usr/bin/env python3
"""
Generate PNG diagrams for architecture visualization.
Uses matplotlib and PIL to create clean, professional diagrams.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_pipeline_diagram():
    """Create pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.5, '🎵 Speech Emotion Recognition Pipeline', 
            fontsize=18, fontweight='bold', ha='center')
    ax.text(5, 5.0, 'WavLM/HuBERT Feature Extraction → Classification',
            fontsize=12, ha='center', color='gray', style='italic')
    
    stages = [
        ('🎤 Audio\nInput', 0.5, 'WAV Files', '#667eea'),
        ('⚙️ Preprocessing', 1.8, 'Normalize &\nResample', '#667eea'),
        ('🧠 Feature\nExtraction\n(WavLM/HuBERT)', 3.3, 'Embeddings:\n768-dim', '#667eea'),
        ('📊 Pooling', 5.2, 'Mean/Max', '#667eea'),
        ('🎯 Classify\n(MLP/SVM/XGB)', 6.8, 'Classification', '#667eea'),
        ('😊 Emotion', 8.3, 'Label', '#764ba2'),
    ]
    
    # Draw stages
    for i, (title, x, desc, color) in enumerate(stages):
        box = FancyBboxPatch((x - 0.4, 2.5), 0.8, 1.2, 
                            boxstyle="round,pad=0.05", 
                            edgecolor='#333', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, 3.3, title, fontsize=10, fontweight='bold', 
               ha='center', va='center', color='white')
        ax.text(x, 1.8, desc, fontsize=8, ha='center', 
               va='center', color='#555', style='italic')
        
        # Draw arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((x + 0.4, 3.1), (stages[i+1][1] - 0.4, 3.1),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='#333', linewidth=2)
            ax.add_patch(arrow)
    
    # Dataset annotation
    ax.text(2.7, 4.5, 'IEMOCAP / CREMA-D', 
           fontsize=9, ha='center', color='#667eea', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#667eea', linewidth=1.5))
    
    # Legend box
    legend_text = (
        'Pipeline Details:\n'
        '• Audio Input: WAV files from IEMOCAP or CREMA-D\n'
        '• Preprocessing: Normalize & resample to 16 kHz\n'
        '• Feature Extraction: WavLM-base (768-dim) or HuBERT-large (1024-dim)\n'
        '• Pooling: Mean or max pooling over time steps\n'
        '• Classification: MLP, SVM, or XGBoost on embeddings\n'
        '• Output: Emotion class (anger, sadness, happiness, neutral, etc.)'
    )
    ax.text(5, 0.4, legend_text, fontsize=9, ha='center', va='top',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='#f9f9f9', 
                    edgecolor='#667eea', linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig('/workspaces/-SpeechEmbeddings-WavLM/architecture_pipeline.png', 
               dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print("✓ Pipeline diagram saved: architecture_pipeline.png")
    plt.close()

def create_mlp_diagram():
    """Create MLP classifier architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Title
    ax.text(5, 8.5, '🧠 MLP Classifier Architecture',
           fontsize=18, fontweight='bold', ha='center')
    ax.text(5, 8.0, 'Feed-forward Neural Network for Emotion Classification',
           fontsize=12, ha='center', color='gray', style='italic')
    
    layers = [
        ('Input Layer', 0.8, 'Embeddings\n768-dim\n(WavLM-base)', '#667eea', ['', '']),
        ('Hidden Layer 1', 2.5, 'Dense\n256 units\nReLU', '#764ba2', ['', '']),
        ('Hidden Layer 2', 4.2, 'Dense\n128 units\nReLU', '#667eea', ['', '']),
        ('Regularization', 5.9, 'Dropout\np=0.5\nTraining', '#ff6b6b', ['', '']),
        ('Output Layer', 7.6, 'Output\n4-7 classes\nSoftmax', '#51cf66', ['', '']),
    ]
    
    # Draw layers
    for i, (name, x, desc, color, params) in enumerate(layers):
        # Layer box
        box = FancyBboxPatch((x - 0.35, 4.5), 0.7, 2.5,
                            boxstyle="round,pad=0.08",
                            edgecolor='#333', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, 6.5, desc, fontsize=9, fontweight='bold',
               ha='center', va='center', color='white')
        
        # Layer name
        ax.text(x, 7.3, name, fontsize=10, fontweight='bold',
               ha='center', va='center', color='#333')
        
        # Draw connections to next layer
        if i < len(layers) - 1:
            for dy in [-0.6, 0, 0.6]:
                arrow = FancyArrowPatch((x + 0.35, 5.5 + dy),
                                      (layers[i+1][1] - 0.35, 5.5 + dy),
                                      arrowstyle='->', mutation_scale=15,
                                      color='#aaa', linewidth=0.8, alpha=0.6)
                ax.add_patch(arrow)
    
    # Training info box
    training_info = (
        'Training Configuration\n'
        '─' * 60 + '\n'
        'Loss Function: Cross-Entropy Optimizer: Adam (lr=0.001)\n'
        'Batch Size: 32 Epochs: 100 (early stopping)\n'
        'Activation: ReLU (hidden), Softmax (output)\n'
        'Weight Init: He Normal Regularization: L2 + Dropout\n'
        'Validation Split: 20% Total Parameters: ~200K'
    )
    
    ax.text(5, 3.2, training_info, fontsize=9, ha='center', va='top',
           family='monospace',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='#f9f9f9',
                    edgecolor='#ccc', linewidth=1.5))
    
    # Architecture summary
    summary = (
        'Architecture Summary:\n'
        '• Input: 768-dim embeddings from WavLM-base (or 1024 for HuBERT-large)\n'
        '• Hidden Layer 1: 256 units with ReLU for feature learning\n'
        '• Hidden Layer 2: 128 units with ReLU for higher-level representations\n'
        '• Dropout: 50% during training to prevent overfitting\n'
        '• Output: 4 classes (neutral, sadness, happiness, anger) or more\n'
        '• Total Trainable Parameters: ~200K'
    )
    ax.text(5, 0.6, summary, fontsize=8.5, ha='center', va='top',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f5ff',
                    edgecolor='#667eea', linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig('/workspaces/-SpeechEmbeddings-WavLM/architecture_mlp.png',
               dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print("✓ MLP diagram saved: architecture_mlp.png")
    plt.close()

if __name__ == '__main__':
    try:
        create_pipeline_diagram()
        create_mlp_diagram()
        print("\n✅ All PNG diagrams generated successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
