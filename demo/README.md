# RAVDESS Speech Emotion Demo

## What this is

- A small demo app that uses either a **Convolutional Neural Network (CNN)** on **mel spectrograms** or a **PyCaret-trained MLP on MFCC features**.
- Trained on the **RAVDESS** emotional speech dataset (8 emotions).
- This Streamlit app lets you record a sentence with an emotion and see what the model predicts.

## Project structure

```
ravdess_emotion_demo/
├── app.py                     # Streamlit entry point
├── src/                       # Model + inference helpers
├── models/                    # Trained weights and label/config metadata
├── assets/
│   └── plots/
│       ├── cnn/               # Visuals surfaced in the app
│       └── eda/               # Extra EDA charts for documentation
├── data/
│   └── uploads/               # Temporary recordings saved at runtime
├── environment.yml            # Conda environment specification
└── setup_and_run.sh           # Helper script for WSL setup + launch
```

The app will automatically create the `data/uploads/` folder if it does not exist. You can safely delete its contents to clear previously uploaded samples.

## Requirements

- Windows with **WSL (Ubuntu)**
- **Conda** (Miniconda or Anaconda)
- Basic terminal usage

## Quick setup (recommended)

From inside WSL:

```bash
cd ravdess_emotion_demo
./setup_and_run.sh
```

This script will:

- Create or reuse the conda environment `ravdess-audio` from `environment.yml`.
- Install `ffmpeg` using `apt` (may ask for your sudo password).
- Start the Streamlit app.

When Streamlit starts, open the URL it prints (usually `http://localhost:8501`) in your browser.

## Manual setup (alternative)

1. Create and activate the environment:

```bash
cd ravdess_emotion_demo
conda env create -n ravdess-audio -f environment.yml  # only once
conda activate ravdess-audio
```

2. Install ffmpeg in WSL Ubuntu:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

3. Run the app:

```bash
streamlit run app.py
```

## How to use the demo

1. Open the Streamlit URL in your browser.
2. Choose the sentence and the emotion you want to imitate.
3. Record your voice on Windows (Voice Recorder or any app) and save as **WAV**, **M4A**, or **MP3**.
4. Upload the file in the app **or** capture a new take directly in the browser (allow microphone access when prompted).
5. Select whether to run the **Mel spectrogram CNN** or the **MFCC PyCaret MLP**.
6. Click **"Analyze emotion"**.
7. The app will:
   - Convert the audio to a standard WAV format.
   - Compute the features for the selected model (mel spectrograms for the CNN or MFCC statistics for the MLP).
   - Show the predicted emotion and the full probability distribution.
   - Compare the model's guess with the emotion you tried to act.

## Model options

- **Mel spectrogram CNN (default):** converts each clip into a normalized mel spectrogram image and runs a convolutional neural network.
- **MFCC + PyCaret MLP:** extracts mean/std statistics of 20 MFCC coefficients and their deltas, then feeds them into a PyCaret-trained multilayer perceptron.
