# CREMA-D Pipeline Guide

## 🎯 Overview

This guide explains how to run the complete speech emotion recognition pipeline using the **CREMA-D dataset** on **CPU-only** environments (optimized for GitHub Codespaces with 4 cores and 16GB RAM).

## 📊 Dataset Information

- **Name**: CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- **Classes**: 6 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad)
- **Audio Format**: WAV files at 16kHz
- **Size**: ~340 MB
- **Hugging Face ID**: `m3hrdadfi/crema-d`

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Full Pipeline

Execute all 5 scripts sequentially:

```bash
cd src

# Step 1: Data Preprocessing
python 1_data_preprocessing.py

# Step 2: Feature Extraction (WavLM embeddings)
python 2_wavlm_feature_extraction.py

# Step 3: Train Classifiers
python 3_train_classifiers.py

# Step 4: Evaluate Models
python 4_evaluation_metrics.py

# Step 5: Visualize Embeddings
python 5_visualization_umap.py
```

## 📁 Expected Outputs

After running the pipeline, you'll have:

### `data/processed/`
- `cremad_subset.csv` - Metadata with filepaths and emotions

### `embeddings/`
- `cremad_embeddings.npy` - WavLM embeddings (N × 768)
- `cremad_labels.npy` - Emotion labels (N,)
- `emotion_embeddings.npz` - Combined format

### `models/`
- `cremad_mlp.pkl` - MLP classifier
- `cremad_lr.pkl` - Logistic Regression classifier
- `cremad_scaler.pkl` - Feature scaler
- `cremad_encoder.pkl` - Label encoder

### `results/`
- `cm_cremad_*.png` - Confusion matrices
- `report_cremad_*.csv` - Classification reports
- `comparison_*.png` - Performance comparison plots
- `umap_2d_cremad.png` - 2D UMAP visualization
- `umap_3d_cremad.png` - 3D UMAP visualization
- `all_results.csv` - Combined results

## 🔧 Configuration Options

### 1. Data Preprocessing

**Using Real CREMA-D Dataset:**

Download from official sources:
- Official: https://github.com/CheyneyComputerScience/CREMA-D
- Kaggle: https://www.kaggle.com/datasets/ejlok1/cremad

Extract all `.wav` files to `data/CREMA-D/` and run:

```bash
python 1_data_preprocessing.py
```

**Using Synthetic Data (for testing):**

The script automatically generates synthetic data if CREMA-D is unavailable.

### 2. Feature Extraction

**With Real WavLM Model:**

If you have network access to Hugging Face:

```python
# The script will automatically download microsoft/wavlm-base
python 2_wavlm_feature_extraction.py
```

**With Mock Embeddings (offline mode):**

The script automatically falls back to mock embeddings if WavLM can't be downloaded. Mock embeddings are random but allow you to test the full pipeline.

⚠️ **Note**: Mock embeddings will result in poor classifier performance and are only for testing the pipeline workflow.

### 3. Classifier Training

**CPU Optimization:**

The pipeline uses optimized settings for CPU:
- Smaller MLP architecture (128→64 neurons)
- Reduced training iterations (300 max)
- Early stopping
- Multi-core processing (`n_jobs=-1`)

**Supported Classifiers:**
- MLP (Multi-Layer Perceptron) - Default
- LR (Logistic Regression) - Default
- SVM (Support Vector Machine) - Available but slow on CPU
- RF (Random Forest) - Available but slow on CPU

To change classifiers, edit `src/3_train_classifiers.py`:

```python
results = trainer.train_all_classifiers(
    dataset,
    classifier_types=['mlp', 'lr', 'svm', 'rf']  # Add/remove as needed
)
```

## 📈 Performance Metrics

The pipeline computes:
- **Accuracy**: Overall correct predictions
- **Precision**: Correctness of positive predictions
- **Recall**: Ability to find all positive instances
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

Results are saved in `results/` directory.

## 🎨 Visualizations

### UMAP Embeddings

The pipeline generates 2D and 3D UMAP visualizations showing:
- Clustering of emotions in embedding space
- Class separability
- Embedding quality

Colors correspond to different emotions:
- **Red**: Angry
- **Blue**: Disgust
- **Green**: Fear
- **Yellow**: Happy
- **Purple**: Neutral
- **Orange**: Sad

## ⚙️ CPU Optimization Details

The pipeline is optimized for CPU-only environments:

### Memory Management
- Batch processing with garbage collection
- Reduced batch size (8 samples)
- Early stopping to prevent long training

### Computation Efficiency
- Parallel processing where possible
- Smaller neural network architectures
- Fast classifiers (LR, MLP over SVM, RF)

### Expected Runtime (100 samples on 4-core CPU)
1. Data Preprocessing: ~1 minute
2. Feature Extraction: ~15 seconds (with mock), ~5 minutes (with real WavLM)
3. Classifier Training: ~30 seconds
4. Evaluation: ~5 seconds
5. Visualization: ~30 seconds

**Total: ~2-7 minutes** depending on real vs mock embeddings

## 🐛 Troubleshooting

### Issue: "Can't load feature extractor for 'microsoft/wavlm-base'"

**Solution**: Network access to Hugging Face is blocked. The script automatically falls back to mock embeddings. This is normal in restricted environments.

### Issue: "No .wav files found in ../data/CREMA-D"

**Solution**: Download CREMA-D dataset manually and extract to `data/CREMA-D/`. Alternatively, the script will generate synthetic data automatically.

### Issue: Low classifier performance (< 20% accuracy)

**Cause**: Using mock embeddings instead of real WavLM features.

**Solution**: Download the real CREMA-D dataset and ensure network access to load the WavLM model, or accept that mock embeddings are for pipeline testing only.

### Issue: Out of memory errors

**Solution**: 
- Reduce batch size in `2_wavlm_feature_extraction.py`
- Use fewer samples in `1_data_preprocessing.py`
- Close other applications

## 📝 Notes

### Current Limitations

1. **Mock Embeddings**: Due to network restrictions in some environments, the pipeline may use random mock embeddings. These are suitable for testing the pipeline but won't produce meaningful results.

2. **Synthetic Data**: When CREMA-D dataset is unavailable, synthetic audio is generated. This allows testing but should be replaced with real data for production use.

3. **CPU Performance**: Training on CPU is slower than GPU. For large datasets (>1000 samples), consider:
   - Using a GPU-enabled environment
   - Training overnight
   - Using simpler classifiers (LR over MLP)

### Production Deployment

For production use:
1. Download real CREMA-D dataset
2. Use real WavLM model (requires network access to Hugging Face)
3. Consider using more samples (the current default is 100 for quick testing)
4. Train all 4 classifier types for comparison
5. Use cross-validation for robust performance estimates

## 🤝 Support

For issues or questions:
- Check the main README.md
- Review SETUP_GUIDE.md
- Open an issue on GitHub

## 📚 References

- **CREMA-D Paper**: Cao, H., et al. (2014). "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset"
- **WavLM Paper**: Chen, S., et al. (2021). "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing"
- **Hugging Face WavLM**: https://huggingface.co/microsoft/wavlm-base

---

**Last Updated**: November 2025
**Pipeline Version**: 1.0
**Tested on**: GitHub Codespaces (4-core, 16GB RAM, CPU-only)
