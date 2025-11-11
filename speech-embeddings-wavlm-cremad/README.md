# Speech Embeddings using WavLM-base - CREMA-D Dataset

This project implements a speech emotion recognition pipeline using the WavLM-base model, specifically designed to process the CREMA-D dataset. The pipeline is optimized to run entirely on CPU within GitHub Codespaces.

## Project Structure

```
speech-embeddings-wavlm-cremad
├── src
│   ├── 1_download_cremad.py      # Downloads the CREMA-D dataset from Hugging Face or a fallback source
│   ├── 2_extract_cremad.py       # Extracts the downloaded ZIP file containing the dataset
│   ├── 3_parse_labels.py         # Parses emotion labels from the filenames of the WAV files
│   ├── 4_preprocess_audio.py      # Preprocesses audio files to ensure correct format
│   └── 5_build_manifest.py        # Compiles processed data into a CSV file
├── data
│   ├── raw                        # Directory for storing raw data files
│   └── processed                  # Directory for storing processed data files
├── tests
│   └── test_pipeline.py           # Unit tests for the pipeline scripts
├── requirements.txt               # Lists project dependencies
├── .gitignore                     # Specifies files to ignore in Git
└── README.md                      # Project documentation
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd speech-embeddings-wavlm-cremad
   ```

2. **Install Dependencies**
   Ensure you have Python 3.6 or higher installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the CREMA-D Dataset**
   Run the following script to download the dataset:
   ```bash
   python src/1_download_cremad.py
   ```

4. **Extract the Dataset**
   After downloading, extract the dataset using:
   ```bash
   python src/2_extract_cremad.py
   ```

5. **Parse Emotion Labels**
   Parse the emotion labels from the filenames:
   ```bash
   python src/3_parse_labels.py
   ```

6. **Preprocess Audio Files**
   Preprocess the audio files to ensure they are in the correct format:
   ```bash
   python src/4_preprocess_audio.py
   ```

7. **Build the Manifest**
   Finally, compile the processed data into a CSV file:
   ```bash
   python src/5_build_manifest.py
   ```

## About the CREMA-D Dataset

The CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset) is a dataset that contains audio recordings of actors expressing various emotions. It is widely used for training and evaluating emotion recognition systems.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing the datasets library.
- The creators of the CREMA-D dataset for their contributions to the field of emotion recognition.