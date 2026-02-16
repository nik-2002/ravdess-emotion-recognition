#!/usr/bin/env python3
"""Regenerate the sample waveform PNG for the technical report."""

import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

# Get base directory
BASE_DIR = Path(__file__).parent.parent

# Use a local audio file from ravdess_emotion_demo (faster than Google Drive main data)
sample_file = BASE_DIR / "ravdess_emotion_demo" / "data" / "uploads" / "recording_1763438113786.wav"
print(f"Using sample file: {sample_file}")

# Load audio
y, sr = librosa.load(sample_file, sr=22050)

# Create the waveform plot
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Sample Waveform of Speech Audio")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()

# Save to docs/EDA_plots
output_path = BASE_DIR / "docs" / "EDA_plots" / "01_sample_waveform.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")

plt.close()
print("Done!")
