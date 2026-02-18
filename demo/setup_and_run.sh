#!/usr/bin/env bash
set -e

ENV_NAME=ravdess-audio

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$CONDA_EXE" ]; then
    echo "Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

eval "$("$CONDA_EXE" shell.bash hook)"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating conda environment '$ENV_NAME' from environment.yml..."
    conda env create -n "$ENV_NAME" -f environment.yml
fi

# Activate the environment
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate "$ENV_NAME"

# Check for ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo "ffmpeg is already installed. Skipping installation."
else
    echo "Installing ffmpeg via apt (may require sudo)..."
    sudo apt update
    sudo apt install -y ffmpeg
fi

echo "Starting Streamlit app..."
streamlit run app.py
