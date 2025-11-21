# Speech Embeddings using WavLM-base

A research-based project implementing speech embedding extraction and evaluation using Microsoft's WavLM-base model, inspired by the IEEE/ACM 2024 paper *"From Raw Speech to Fixed Representations: A Comprehensive Evaluation of Speech Embedding Techniques."*

## 🎯 Project Overview

This project focuses on **emotion recognition** from speech by extracting fixed-dimensional speech representations using self-supervised models (WavLM and HuBERT) and training classifiers for emotion identification:

- **Emotion Recognition** using IEMOCAP and CREMA-D datasets
- **Multiple Embedding Models:** WavLM-base and HuBERT-large
- **Advanced Classifiers:** SVM, MLP, XGBoost with cross-validation
- **Comprehensive Evaluation:** Accuracy, F1-score, confusion matrices, UMAP visualizations

## 📁 Project Structure

```
SpeechEmbeddings-WavLM/
├── data/                           # Dataset storage
│   ├── IEMOCAP/                   # Emotion recognition dataset
│   ├── CREMA-D/                   # Emotion recognition dataset
│   └── processed/                 # Preprocessed metadata CSV files
├── src/                            # Source code
│   ├── 1_data_preprocessing.py    # Data loading and preprocessing
│   ├── 2_wavlm_feature_extraction.py  # WavLM/HuBERT embedding extraction
│   ├── 3_train_classifiers.py     # Classifier training (SVM, MLP, XGBoost)
│   ├── 4_evaluation_metrics.py    # Performance evaluation
│   └── 5_visualization_umap.py    # UMAP visualization
├── embeddings/                     # Extracted feature embeddings (.npz files)
│   ├── emotion_embeddings.npz
│   └── emotion_embeddings_hubert_large.npz
├── models/                         # Trained classifier models
├── results/                        # Evaluation metrics and visualizations
├── .devcontainer/                  # GitHub Codespaces configuration
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- GPU recommended for faster processing (CUDA-compatible)
- GitHub Codespaces Pro (for cloud development)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/srinivasCharan23/-SpeechEmbeddings-WavLM.git
   cd SpeechEmbeddings-WavLM
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Datasets are automatically loaded** from HuggingFace:
   - IEMOCAP: Loaded automatically via `datasets` library
   - CREMA-D: Included in the repository for emotion recognition

### Using GitHub Codespaces

This project is optimized for GitHub Codespaces Pro:

1. Open the repository in GitHub
2. Click on **Code** → **Codespaces** → **Create codespace**
3. The environment will automatically set up with all dependencies installed
4. Start developing!

## 📊 Usage

### Step 1: Data Preprocessing

Process raw datasets and generate metadata:

```bash
cd src
python 1_data_preprocessing.py
```

This script:
- Loads IEMOCAP dataset from HuggingFace (5% subset for CPU efficiency)
- Processes CREMA-D dataset for emotion labels
- Generates metadata CSV files with emotion labels (neutral, happy, sad, angry)

### Step 2: Feature Extraction

Extract WavLM embeddings from audio files:

```bash
python 2_wavlm_feature_extraction.py
```

This script:
- Loads pre-trained models (WavLM-base or HuBERT-large)
- Processes audio files on CPU with optimized batching
- Extracts fixed-dimensional embeddings (768-dim for WavLM, 1024-dim for HuBERT)
- Saves embeddings as `.npz` files with labels in `embeddings/`

### Step 3: Train Classifiers

Train multiple classifiers on extracted embeddings:

```bash
python 3_train_classifiers.py
```

Supported classifiers with 5-fold cross-validation:
- Support Vector Machine (SVM) with RBF kernel
- Multi-Layer Perceptron (MLP) with dropout
- XGBoost with optimized hyperparameters

Usage:
```bash
python 3_train_classifiers.py --npz-path embeddings/emotion_embeddings.npz --classifier svm --n-folds 5
python 3_train_classifiers.py --npz-path embeddings/emotion_embeddings_hubert_large.npz --classifier mlp --n-folds 5
```

### Step 4: Evaluate Performance

Compute comprehensive evaluation metrics:

```bash
python 4_evaluation_metrics.py
```

Generates:
- Accuracy, Precision, Recall, F1-scores
- Confusion matrices
- Classification reports
- Cross-dataset comparisons

### Step 5: Visualize Embeddings

Create UMAP visualizations of embedding space:

```bash
python 5_visualization_umap.py
```

Outputs:
- 2D scatter plots
- 3D scatter plots
- Grid comparisons across datasets

## 🔬 Technical Details

### WavLM Model

- **Model:** `microsoft/wavlm-base` (HuggingFace Transformers)
- **Architecture:** Transformer-based self-supervised model
- **Pre-training:** Large-scale unlabeled speech data
- **Input:** 16kHz raw audio waveforms
- **Output:** 768-dimensional contextualized representations

### Embedding Extraction

- **Pooling Strategies:** Mean, Max, First, Last token
- **Layer Selection:** Configurable (default: last layer)
- **Multi-layer:** Optional extraction from multiple layers

### Emotion Recognition Task

| Model | Dataset | Metric | Classes | Best Accuracy |
|-------|---------|--------|---------|---------------|
| WavLM-base + SVM | IEMOCAP | Weighted F1 | 4 emotions | ~75-80% |
| HuBERT-large + MLP | IEMOCAP | Weighted F1 | 4 emotions | ~80-85% |
| HuBERT-large + XGBoost | IEMOCAP | Weighted F1 | 4 emotions | ~85%+ |

**Emotion Classes:** Neutral, Happy, Sad, Angry## 📈 Results

Results are saved in the `results/` directory:

- `evaluation_results_*.json` - Metrics for each model (accuracy, precision, recall, F1)
- `confusion_matrix_*_cv.csv` - Confusion matrices from cross-validation
- `metrics.json` - Overall evaluation metrics
- `umap_emotion.png` - UMAP visualization of emotion embeddings
- Training logs with detailed per-fold results

## 👥 Team

This project is developed by an AI/ML research team focused on emotion recognition from speech:

- **Lead Researcher** - Model architecture and emotion recognition pipeline
- **Data Engineering** - IEMOCAP and CREMA-D dataset preparation
- **Model Training** - Classifier optimization and cross-validation
- **Evaluation** - Performance metrics and model comparison
- **Visualization** - UMAP embeddings and result analysis

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{speechembeddings2024,
  title={From Raw Speech to Fixed Representations: A Comprehensive Evaluation of Speech Embedding Techniques},
  journal={IEEE/ACM Transactions},
  year={2024}
}
```

## 🛠️ Technologies Used

- **Deep Learning:** PyTorch, Transformers (HuggingFace)
- **Audio Processing:** torchaudio, librosa
- **Machine Learning:** scikit-learn
- **Visualization:** matplotlib, seaborn, UMAP
- **Data Processing:** pandas, numpy

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or collaboration opportunities, please open an issue in this repository.

## 🙏 Acknowledgments

- Microsoft Research for the WavLM model
- Facebook AI Research for the HuBERT model
- Dataset providers: IEMOCAP, CREMA-D
- HuggingFace for the Transformers and Datasets libraries
- The open-source community

---

**Note:** This is a research project. Ensure you have proper licenses and permissions for all datasets before use.
