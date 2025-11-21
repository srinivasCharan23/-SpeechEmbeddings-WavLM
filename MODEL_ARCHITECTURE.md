# Model Architecture and Pipeline

This document captures the architecture for the emotion recognition project. It reflects our setup: a small IEMOCAP subset with WavLM-base, and CREMA-D evaluated with high-capacity embeddings.

## High-level Pipeline

```mermaid
flowchart TD
    A[Raw Audio Datasets]:::ds -->|IEMOCAP via HuggingFace (small subset)\nCREMA-D local WAVs| B[1) Data Preprocessing]
    B --> C[2) Feature Extraction\nWavLM-base or HuBERT-large\nPooling: mean/max/first/last]
    C -->|embeddings, labels| D[(NPZ store\nembeddings/emotion_embeddings*.npz)]
    D --> E[3) Train Classifiers\nMLP (this repo)\nSVM/XGBoost (results supported)]
    E --> F[(Artifacts\nmodels/*.pt, *_scaler.pkl, *_encoder.pkl)]
    E --> G[4) Evaluation\nAccuracy / F1 / Confusion Matrix]
    D --> H[5) UMAP Visualization\n2D embedding plots]
    G --> R[Results\nresults/*.json, confusion_matrix_*.png]
    H --> R

    classDef ds fill:#eef,stroke:#3366cc,stroke-width:1px
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

```mermaid
flowchart LR
    X[Input Embedding\n768 (WavLM-base) or 1024 (HuBERT-large)] --> L1[Linear(in → hidden_dim)\nReLU + Dropout(0.3)]
    L1 --> L2[Linear(hidden_dim → hidden_dim/2)\nReLU + Dropout(0.3)]
    L2 --> O[Linear(hidden_dim/2 → num_classes=4)\nSoftmax at inference]
```

Implemented in `src/3_train_classifiers.py` (`SimpleMLP`).

Notes:
- Emotion classes in this setup: Neutral, Happy, Sad, Angry (4-class setting for IEMOCAP subset). CREMA-D labels follow its 6-class scheme when used to generate embeddings.
- CPU-optimized pipeline; embeddings saved as `.npz` with labels for reproducibility.
