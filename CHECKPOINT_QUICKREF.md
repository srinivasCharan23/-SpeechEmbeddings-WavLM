# Checkpoint Quick Reference

## TL;DR - Just 3 Commands

### 1️⃣ Start Extraction
```bash
python src/2_wavlm_feature_extraction.py --augment
```

### 2️⃣ Stop Anytime (Ctrl+C)
```bash
^C  # Press Ctrl+C to interrupt
# Output: "💾 Progress saved! Run the same command to resume."
```

### 3️⃣ Resume Later
```bash
python src/2_wavlm_feature_extraction.py --augment
# Output: "🔄 Resuming: found 2000 already-processed samples"
```

---

## Common Use Cases

### Split Long Run Across Days
```bash
# Day 1: Process for 2 hours
python src/2_wavlm_feature_extraction.py --augment
# ... work for 2 hours ...
# Press Ctrl+C

# Day 2: Continue where you left off
python src/2_wavlm_feature_extraction.py --augment
# ... completes in ~2-3 more hours ...
```

### Test Small Batch First
```bash
# Process 50 samples to test
python src/2_wavlm_feature_extraction.py --augment --checkpoint-interval 10
# Press Ctrl+C after ~10-15 samples

# Check output
ls embeddings/checkpoints/
# checkpoint_0000-0009.npz

# Continue if results look good
python src/2_wavlm_feature_extraction.py --augment
```

### Clean Start After Problems
```bash
# Delete all checkpoints
rm -rf embeddings/checkpoints/

# Start fresh
python src/2_wavlm_feature_extraction.py --augment --no-resume
```

---

## Checkpoints Location
```
embeddings/checkpoints/
├── checkpoint_0000-0099.npz
├── checkpoint_0100-0199.npz
├── checkpoint_0200-0299.npz
└── ...
```

---

## What Gets Saved

✅ WavLM embeddings (1024-dim vectors)  
✅ Emotion labels  
✅ Audio file paths  
✅ Augmentation variants  

❌ Model weights (loaded fresh each time)  
❌ Audio files (read from disk)  

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint-interval` | 100 | Samples between checkpoints |
| `--no-resume` | False | Ignore checkpoints, start fresh |
| `--augment` | False | Enable audio augmentation |
| `--use-gpu` | False | Use GPU if available |

---

## Troubleshooting

### ❓ Resume didn't work?
Check you used the **exact same command**:
- Same `--csv` path
- Same `--out` path
- Same `--augment` flag

### ❓ Out of disk space?
```bash
# Check checkpoint size
du -sh embeddings/checkpoints/

# Use larger intervals
python src/2_wavlm_feature_extraction.py --augment --checkpoint-interval 200
```

### ❓ Want to see progress?
```bash
# Count checkpoint files
ls embeddings/checkpoints/ | wc -l
# Each file = 100 samples processed

# Check latest checkpoint
ls -lt embeddings/checkpoints/ | head -2
```

---

## Full Documentation

See [CHECKPOINT_GUIDE.md](CHECKPOINT_GUIDE.md) for comprehensive details.
