import unittest
import os
import pandas as pd

class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.raw_data_dir = '../data/raw'
        self.processed_data_dir = '../data/processed'
        self.processed_csv_path = os.path.join(self.processed_data_dir, 'cremad_subset.csv')

    def test_download_cremad(self):
        # Test if the CREMA-D dataset is downloaded
        self.assertTrue(os.path.exists(self.raw_data_dir), "Raw data directory does not exist.")
        # Additional checks can be added to verify specific files

    def test_extract_cremad(self):
        # Test if the extraction of the CREMA-D dataset was successful
        extracted_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.wav')]
        self.assertGreater(len(extracted_files), 0, "No WAV files found in the raw data directory.")

    def test_parse_labels(self):
        # Test if labels are parsed correctly from filenames
        # Assuming the filenames follow a specific format
        sample_filename = "crema-D_1_01_01_01_01.wav"  # Example filename
        emotion = sample_filename.split('_')[2]  # Extracting emotion from filename
        self.assertIn(emotion, ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad'], "Parsed emotion is not valid.")

    def test_preprocess_audio(self):
        # Test if audio preprocessing is done correctly
        # This would require checking the format of the processed audio files
        # Placeholder for actual audio processing checks

        # Example check (assuming processed files are saved in a specific directory)
        processed_audio_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.wav')]
        self.assertGreater(len(processed_audio_files), 0, "No processed audio files found.")

    def test_build_manifest(self):
        # Test if the manifest CSV is created successfully
        self.assertTrue(os.path.exists(self.processed_csv_path), "Processed CSV file does not exist.")
        df = pd.read_csv(self.processed_csv_path)
        self.assertGreater(len(df), 0, "Processed CSV is empty.")
        self.assertIn('filepath', df.columns, "CSV does not contain 'filepath' column.")
        self.assertIn('emotion', df.columns, "CSV does not contain 'emotion' column.")

if __name__ == '__main__':
    unittest.main()