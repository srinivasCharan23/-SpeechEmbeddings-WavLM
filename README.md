# Speech Embeddings using WavLM-base

A research-based project implementing speech embedding extraction and evaluation using Microsoft's WavLM-base model, inspired by the IEEE/ACM 2024 paper *"From Raw Speech to Fixed Representations: A Comprehensive Evaluation of Speech Embedding Techniques."*

## 🎯 Project Overview

This project extracts fixed-dimensional speech representations from raw audio using the self-supervised WavLM model and evaluates them on multiple downstream tasks:

- **Emotion Identification** (IEMOCAP dataset)
- **Gender Identification** (LibriSpeech dataset)
- **Intent Identification** (SLURP dataset)
- **Cross-language Embeddings** (CommonVoice English + Hindi)

## 📁 Project Structure

```
SpeechEmbeddings-WavLM/
├── data/                           # Dataset storage
│   ├── IEMOCAP/                   # Emotion recognition dataset
│   ├── LibriSpeech/               # Speaker identification dataset
│   ├── SLURP/                     # Intent classification dataset
│   ├── CommonVoice/               # Language/accent dataset
│   └── processed/                 # Preprocessed metadata
├── src/                            # Source code
│   ├── 1_data_preprocessing.py    # Data loading and preprocessing
│   ├── 2_wavlm_feature_extraction.py  # WavLM embedding extraction
│   ├── 3_train_classifiers.py     # Classifier training
│   ├── 4_evaluation_metrics.py    # Performance evaluation
│   └── 5_visualization_umap.py    # UMAP visualization
├── embeddings/                     # Extracted feature embeddings (.npy files)
├── models/                         # Trained classifier checkpoints
├── results/                        # Evaluation metrics and visualizations
├── .devcontainer/                  # GitHub Codespaces configuration
│   └── devcontainer.json
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

3. **Download datasets** (place them in the `data/` directory):
   - [IEMOCAP](https://sail.usc.edu/iemocap/)
   - [LibriSpeech](https://www.openslr.org/12)
   - [SLURP](https://github.com/pswietojanski/slurp)
   - [CommonVoice](https://commonvoice.mozilla.org/)

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
- Loads audio files from each dataset
- Validates file integrity
- Generates metadata CSV files with labels

### Step 2: Feature Extraction

Extract WavLM embeddings from audio files:

```bash
python 2_wavlm_feature_extraction.py
```

This script:
- Loads the pre-trained WavLM-base model
- Processes audio files in batches
- Extracts fixed-dimensional embeddings (768-dim by default)
- Saves embeddings as `.npy` files in `embeddings/`

### Step 3: Train Classifiers

Train multiple classifiers on extracted embeddings:

```bash
python 3_train_classifiers.py
```

Supported classifiers:
- Support Vector Machine (SVM)
- Random Forest (RF)
- Multi-Layer Perceptron (MLP)
- Logistic Regression (LR)

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

- **Architecture:** Transformer-based self-supervised model
- **Pre-training:** Large-scale unlabeled speech data
- **Input:** 16kHz raw audio waveforms
- **Output:** 768-dimensional contextualized representations

### Embedding Extraction

- **Pooling Strategies:** Mean, Max, First, Last token
- **Layer Selection:** Configurable (default: last layer)
- **Multi-layer:** Optional extraction from multiple layers

### Downstream Tasks

| Task | Dataset | Metric | Classes |
|------|---------|--------|---------|
| Emotion Identification | IEMOCAP | Weighted F1 | 4-8 emotions |
| Gender Identification | LibriSpeech | Accuracy | Male/Female |
| Intent Identification | SLURP | Macro F1 | 18 intents |
| Cross-language Embeddings | CommonVoice (EN+HI) | Accuracy | English/Hindi |

## 📈 Results

Results are saved in the `results/` directory:

- `all_results.csv` - Combined metrics for all experiments
- `comparison_*.csv` - Per-dataset classifier comparisons
- `cm_*.png` - Confusion matrices
- `umap_*.png` - UMAP visualizations
- `report_*.csv` - Detailed classification reports

## 👥 Team Roles

This project is developed by a 5-member AI/ML research team focused on speech processing and representation learning:

- **Inthiyaz** - Model Architect (WavLM embedding & pipeline design)
- **Teammate A** - Data Engineer (dataset preparation and preprocessing)
- **Teammate B** - Trainer (Emotion & Gender classification model training)
- **Teammate C** - Evaluator (metrics computation and performance analysis)
- **Teammate D** - Visualizer (UMAP visualizations and results presentation)

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
- Dataset providers: IEMOCAP, LibriSpeech, SLURP, CommonVoice
- HuggingFace for the Transformers library
- The open-source community

---

**Note:** This is a research project. Ensure you have proper licenses and permissions for all datasets before use.
