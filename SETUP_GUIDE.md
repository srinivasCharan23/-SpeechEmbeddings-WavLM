# Speech Embeddings WavLM - Setup Guide

## 🎯 Quick Start

This repository is ready to use in GitHub Codespaces Pro with GPU support!

### Option 1: Using GitHub Codespaces (Recommended)

1. **Open in Codespaces:**
   - Go to the repository on GitHub
   - Click **Code** → **Codespaces** → **Create codespace on main**
   - Select a machine type with GPU (Codespaces Pro required)

2. **Wait for setup:**
   - The devcontainer will automatically install all dependencies
   - This takes 2-5 minutes on first launch

3. **Verify GPU access:**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

4. **Start working:**
   - All scripts are in the `src/` directory
   - See your role assignments below

### Option 2: Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/srinivasCharan23/-SpeechEmbeddings-WavLM.git
   cd SpeechEmbeddings-WavLM
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download datasets** (see Dataset Setup below)

## 📋 Team Roles & Responsibilities

### Inthiyaz - Model Architect (WavLM embedding & pipeline)
**Primary Script:** `src/2_wavlm_feature_extraction.py`

**Your Tasks:**
- Experiment with different WavLM model variants (base, base-plus, large)
- Implement multi-layer feature fusion strategies
- Optimize pooling methods (mean, max, attention-based)
- Add audio augmentation techniques for robustness
- Design the embedding extraction pipeline architecture

**Key Functions to Implement:**
- Multi-layer feature extraction
- Advanced pooling strategies
- Model optimization for GPU/CPU

---

### Teammate A - Data Engineer (dataset preparation)
**Primary Script:** `src/1_data_preprocessing.py`

**Your Tasks:**
- IEMOCAP: Parse session folders, extract emotion labels
- LibriSpeech: Extract gender labels from SPEAKERS.TXT
- SLURP: Parse JSON annotations for intent labels
- CommonVoice: Focus on English + Hindi subsets

**Key Functions to Implement:**
- `preprocess_iemocap()` - Emotion Identification
- `preprocess_librispeech()` - Gender Identification
- `preprocess_slurp()` - Intent Identification
- `preprocess_commonvoice()` - Cross-language embeddings

---

### Teammate B - Trainer (Emotion & Gender classification)
**Primary Script:** `src/3_train_classifiers.py`

**Your Tasks:**
- Fine-tune hyperparameters for each classifier type
- Implement cross-validation strategies
- Optimize training for Emotion and Gender classification tasks
- Add early stopping and regularization techniques
- Experiment with different feature scaling methods

**Key Functions to Focus On:**
- `train_classifier()` - Optimize training
- `get_classifier()` - Add new classifier types
- Grid search optimization

---

### Teammate C - Evaluator (metrics computation)
**Primary Script:** `src/4_evaluation_metrics.py`

**Your Tasks:**
- Implement additional metrics (ROC-AUC, PR curves)
- Create detailed per-class analysis reports
- Compare performance across different datasets and tasks
- Generate statistical significance tests
- Create comprehensive benchmark tables

**Key Functions to Implement:**
- Additional metric calculations
- Cross-dataset comparisons
- Statistical tests
- Export to multiple formats

---

### Teammate D - Visualizer (UMAP + results)
**Primary Script:** `src/5_visualization_umap.py`

**Your Tasks:**
- Create interactive 3D visualizations (plotly)
- Generate t-SNE visualizations for comparison
- Plot decision boundaries for classifiers
- Design publication-quality figures
- Create dashboard-style summary visualizations

**Key Functions to Implement:**
- Interactive 3D plots
- t-SNE comparison
- Grid visualizations
- Animation support

## 📊 Target Tasks

1. **Emotion Identification (IEMOCAP)**
   - Dataset: IEMOCAP
   - Classes: 4-8 emotions (neutral, happy, sad, angry, etc.)
   - Metric: Weighted F1-score

2. **Gender Identification (LibriSpeech)**
   - Dataset: LibriSpeech
   - Classes: Male/Female
   - Metric: Accuracy

3. **Intent Identification (SLURP)**
   - Dataset: SLURP
   - Classes: 18 intents
   - Metric: Macro F1-score

4. **Cross-language Embeddings (CommonVoice)**
   - Dataset: CommonVoice (English + Hindi)
   - Classes: English/Hindi
   - Metric: Accuracy

## 🗂️ Dataset Setup

Place datasets in the following structure:

```
data/
├── IEMOCAP/
│   ├── Session1/
│   ├── Session2/
│   └── ...
├── LibriSpeech/
│   ├── train-clean-100/
│   ├── SPEAKERS.TXT
│   └── ...
├── SLURP/
│   ├── slurp_real/
│   ├── slurp_synth/
│   ├── train.jsonl
│   └── ...
└── CommonVoice/
    ├── en/
    │   ├── clips/
    │   └── validated.tsv
    └── hi/
        ├── clips/
        └── validated.tsv
```

## 🔄 Workflow

1. **Data Preprocessing** (Teammate A)
   ```bash
   cd src
   python 1_data_preprocessing.py
   ```

2. **Feature Extraction** (Inthiyaz)
   ```bash
   python 2_wavlm_feature_extraction.py
   ```

3. **Train Classifiers** (Teammate B)
   ```bash
   python 3_train_classifiers.py
   ```

4. **Evaluate Models** (Teammate C)
   ```bash
   python 4_evaluation_metrics.py
   ```

5. **Create Visualizations** (Teammate D)
   ```bash
   python 5_visualization_umap.py
   ```

## 🛠️ Technical Details

### WavLM Model
- **Model:** `microsoft/wavlm-base` from HuggingFace
- **Input:** 16kHz audio waveforms
- **Output:** 768-dimensional embeddings
- **Pooling:** Mean, Max, First, Last token

### Sample Feature Extraction

See `src/2_wavlm_feature_extraction.py` for a complete example:

```python
from src.2_wavlm_feature_extraction import WavLMFeatureExtractor

extractor = WavLMFeatureExtractor()
embedding = extractor.extract_from_file("audio.wav")
print(f"Embedding shape: {embedding.shape}")  # (768,)
```

## 📈 Expected Outputs

- **Embeddings:** `embeddings/*.npy`
- **Models:** `models/*.pkl`
- **Results:** `results/*.csv`, `results/*.png`
- **Visualizations:** `results/umap_*.png`

## 🐛 Troubleshooting

### GPU not available in Codespaces
- Make sure you selected a GPU-enabled machine type
- Check: `nvidia-smi` in terminal
- Verify: `torch.cuda.is_available()` returns `True`

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.10+)

### Dataset not found
- Verify datasets are in correct directories under `data/`
- Check file permissions
- Ensure .gitkeep files haven't been deleted

## 📚 Resources

- **WavLM Paper:** https://arxiv.org/abs/2110.13900
- **HuggingFace WavLM:** https://huggingface.co/microsoft/wavlm-base
- **IEMOCAP:** https://sail.usc.edu/iemocap/
- **LibriSpeech:** https://www.openslr.org/12
- **SLURP:** https://github.com/pswietojanski/slurp
- **CommonVoice:** https://commonvoice.mozilla.org/

## ✅ Validation Checklist

Before starting work, verify:

- [ ] Repository cloned/opened in Codespaces
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] GPU available (if using Codespaces Pro)
- [ ] Datasets downloaded and placed correctly
- [ ] Can import scripts without errors
- [ ] Understand your role and responsibilities
- [ ] Read the TODOs in your assigned script(s)

## 🤝 Collaboration

- Use meaningful commit messages
- Document your code changes
- Update TODOs as you complete tasks
- Share results in team meetings
- Ask for help when stuck!

---

**Ready to start? Pick your role and begin with the TODOs in your script!**
