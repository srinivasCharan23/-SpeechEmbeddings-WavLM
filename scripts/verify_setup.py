#!/usr/bin/env python3
import os
from pathlib import Path
import sys

def count_wavs(root: Path):
    return sum(1 for _ in root.rglob('*.wav'))

def main():
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / 'data'
    results_dir = repo / 'results'
    embeddings_dir = repo / 'embeddings'

    print('=== Dataset Check ===')
    crema_candidates = [
        data_dir / 'CREMA-D' / 'AudioWAV',
        data_dir / 'CREMA-D' / 'CREMA-D-master' / 'AudioWAV',
        data_dir / 'CREMA-D'
    ]
    crema_wavs = 0
    crema_root = None
    for cand in crema_candidates:
        if cand.exists():
            cnt = count_wavs(cand)
            if cnt > crema_wavs:
                crema_wavs = cnt
                crema_root = cand
    if crema_root is None or crema_wavs == 0:
        print('- CREMA-D: MISSING (no .wav files found)')
        print('  Place audio under data/CREMA-D/CREMA-D-master/AudioWAV/')
    else:
        print(f'- CREMA-D: OK ({crema_wavs} wav files under {crema_root})')

    iemocap_dir = data_dir / 'IEMOCAP'
    if not iemocap_dir.exists():
        print('- IEMOCAP folder missing (will be loaded via HuggingFace)')
    else:
        print('- IEMOCAP folder present (if empty, script will fetch subset)')

    print('\n=== Embeddings Check ===')
    e1 = embeddings_dir / 'emotion_embeddings.npz'
    e2 = embeddings_dir / 'emotion_embeddings_hubert_large.npz'
    print(f'- {e1.name}: {"FOUND" if e1.exists() else "MISSING"}')
    print(f'- {e2.name}: {"FOUND" if e2.exists() else "MISSING"}')

    print('\n=== Results Check ===')
    expected = [
        'confusion_matrix_wavlm_svm_cv.png',
        'confusion_matrix_hubert_svm_cv.png',
        'per_class_metrics_wavlm_svm_cv.png',
        'per_class_metrics_hubert_svm_cv.png',
        'umap_emotion.png',
        'metrics.json'
    ]
    for name in expected:
        p = results_dir / name
        print(f'- {name}: {"FOUND" if p.exists() else "MISSING"}')

    print('\nTips:')
    print('- If embeddings are missing, run: python src/2_wavlm_feature_extraction.py')
    print('- To train on existing embeddings: python src/3_train_classifiers.py')
    print('- To generate metrics/plots: python src/4_evaluation_metrics.py and python src/5_visualization_umap.py')

if __name__ == '__main__':
    sys.exit(main())
