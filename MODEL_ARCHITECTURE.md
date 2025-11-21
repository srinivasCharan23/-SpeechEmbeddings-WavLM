# Model Architecture and Pipeline

This document captures the architecture for the emotion recognition project. It reflects our setup: a small IEMOCAP subset with WavLM-base, and CREMA-D evaluated with high-capacity embeddings.

## High-level Pipeline

**Interactive Visualizations (Non-Mermaid):**
- **HTML Version:** [`architecture_pipeline.html`](architecture_pipeline.html)
- **PNG Image:** [`architecture_pipeline.png`](architecture_pipeline.png)

### ASCII Pipeline:
```
Audio Input (WAV: IEMOCAP/CREMA-D)
    ↓
[1] Data Preprocessing (normalize, 16 kHz)
    ↓
[2] Feature Extraction (WavLM-base/HuBERT-large + pooling)
    ↓
Embeddings → embeddings/emotion_embeddings*.npz
    ↓
[3] Train Classifiers (MLP/SVM/XGBoost)
    ↓
Models → models/*.pt, *_scaler.pkl, *_encoder.pkl
    ↓
[4] Evaluation (Accuracy/F1/Confusion Matrix)
    ↓
[5] Visualization (UMAP 2D plots)
    ↓
Results → results/*.json, confusion_matrix_*.png
```

- Preprocessing: `src/1_data_preprocessing.py` (IEMOCAP small subset via HuggingFace; CREMA-D via metadata)
- Feature extraction: `src/2_wavlm_feature_extraction.py` (default: `microsoft/wavlm-base`; set `model_name` for other SSL models such as HuBERT-large)
- Training: `src/3_train_classifiers.py` (MLP)
- Evaluation: `src/4_evaluation_metrics.py`
- Visualization: `src/5_visualization_umap.py`
- Embeddings: `embeddings/emotion_embeddings.npz`, `embeddings/emotion_embeddings_hubert_large.npz`
- Results: `results/*.json`, `results/*confusion_matrix*.png`, `results/per_class_metrics_*.png`

## Configurations Used

- IEMOCAP (small subset): WavLM-base on CPU for quick validation
- CREMA-D: High-capacity embeddings (e.g., HuBERT-large). The extractor supports switching models via `model_name`.

Example (conceptual):
```python
# In src/2_wavlm_feature_extraction.py
extractor = WavLMFeatureExtractor(model_name="microsoft/wavlm-base")  # IEMOCAP subset
# For CREMA-D with a larger model, set:
# extractor = WavLMFeatureExtractor(model_name="facebook/hubert-large-ll60k")
```

## MLP Classifier Architecture

**Interactive Visualizations:**
- **HTML Version:** [`architecture_mlp.html`](architecture_mlp.html)
- **PNG Image:** [`architecture_mlp.png`](architecture_mlp.png)

### ASCII Architecture:
```
Input (768-dim from WavLM-base or 1024-dim from HuBERT-large)
    ↓
[Dense 256 + ReLU + Dropout(0.3)]
    ↓
[Dense 128 + ReLU + Dropout(0.3)]
    ↓
[Dense num_classes (4 or 6) + Softmax]
    ↓
Emotion Class Output
```

**Implemented in:** `src/3_train_classifiers.py` (`SimpleMLP` class)

Notes:
- Emotion classes in this setup: Neutral, Happy, Sad, Angry (4-class setting for IEMOCAP subset). CREMA-D labels follow its 6-class scheme when used to generate embeddings.
- CPU-optimized pipeline; embeddings saved as `.npz` with labels for reproducibility.
