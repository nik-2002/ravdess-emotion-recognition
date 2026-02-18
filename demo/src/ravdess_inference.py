"""Inference utilities for the RAVDESS mel-spectrogram CNN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import librosa.util
import numpy as np
import torch
import torch.nn.functional as F

from .simple_cnn import ImprovedMelCNN


device = torch.device("cpu")
_device = device


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "ravdess_melspec_cnn_best.pt"
CONFIG_PATH = MODELS_DIR / "ravdess_melspec_config.json"
LABELS_PATH = MODELS_DIR / "emotion_label_mapping.json"
TARGET_FRAMES = 200

_model: ImprovedMelCNN | None = None
_config: Dict[str, Any] | None = None
_emotion_to_idx: Dict[str, int] | None = None
_idx_to_emotion: Dict[int, str] | None = None


def _load_config_and_labels() -> None:
    """Load mel-spectrogram config and label mappings if needed."""
    global _config, _emotion_to_idx, _idx_to_emotion
    if _config is not None and _emotion_to_idx is not None and _idx_to_emotion is not None:
        return

    with CONFIG_PATH.open("r", encoding="utf-8") as cfg_file:
        _config = json.load(cfg_file)

    with LABELS_PATH.open("r", encoding="utf-8") as labels_file:
        labels_data = json.load(labels_file)

    emotion_to_idx_raw = labels_data.get("EMOTION_TO_IDX", {})
    idx_to_emotion_raw = labels_data.get("IDX_TO_EMOTION", {})

    _emotion_to_idx = {str(k): int(v) for k, v in emotion_to_idx_raw.items()}
    _idx_to_emotion = {int(k): str(v) for k, v in idx_to_emotion_raw.items()}


def _build_model() -> None:
    """Instantiate and load the CNN if it has not been created."""
    global _model
    if _model is not None:
        return

    _load_config_and_labels()
    if _config is None or _idx_to_emotion is None:
        raise RuntimeError("Configuration or label mappings failed to load.")

    n_mels = int(_config["N_MELS"])
    n_classes = len(_idx_to_emotion)
    model = ImprovedMelCNN(n_mels=n_mels, n_classes=n_classes)
    state = torch.load(WEIGHTS_PATH, map_location=_device)
    model.load_state_dict(state)
    model.to(_device)
    model.eval()
    _model = model


def _wav_to_melspec_tensor(wav_path: str | Path) -> torch.Tensor:
    """Convert a waveform file into a normalized mel-spectrogram tensor."""
    _load_config_and_labels()
    if _config is None:
        raise RuntimeError("Mel-spectrogram configuration is not loaded.")

    sr = int(_config.get("SAMPLE_RATE", 22050))
    n_mels = int(_config["N_MELS"])
    hop_length = int(_config.get("HOP_LENGTH", 512))
    fmin = float(_config.get("FMIN", 0.0))
    fmax = float(_config.get("FMAX", sr // 2))

    audio, _ = librosa.load(str(wav_path), sr=sr)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = librosa.util.fix_length(mel_db, size=TARGET_FRAMES, axis=1)
    mean = float(mel_db.mean())
    std = float(mel_db.std())
    if std < 1e-8:
        std = 1.0
    mel_norm = (mel_db - mean) / std
    tensor = torch.from_numpy(mel_norm).float().unsqueeze(0).unsqueeze(0)
    return tensor


def predict_emotion_cnn(wav_path: str | Path, top_k: int = 3) -> Dict[str, Any]:
    """Run inference on a WAV file and return predicted emotion probabilities."""
    _build_model()
    if _model is None or _idx_to_emotion is None:
        raise RuntimeError("Model or label mappings are unavailable.")

    path = Path(wav_path)
    mel_tensor = _wav_to_melspec_tensor(path).to(_device)

    with torch.no_grad():
        logits = _model(mel_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    probs_list: List[Tuple[str, float]] = []
    for idx, prob in enumerate(probabilities.tolist()):
        label = _idx_to_emotion[idx]
        probs_list.append((label, float(prob)))

    probs_list.sort(key=lambda item: item[1], reverse=True)
    top_k = max(1, top_k)
    top_k_pairs = probs_list[:top_k]
    probs_dict = {label: prob for label, prob in probs_list}
    predicted_label = top_k_pairs[0][0] if top_k_pairs else None

    return {
        "predicted": predicted_label,
        "probs": probs_dict,
        "top_k": top_k_pairs,
    }


if __name__ == "__main__":
    sample_wav = (
        PROJECT_ROOT.parent
        / "data"
        / "ravdess_data"
        / "Actor_01"
        / "03-01-01-01-01-01-01.wav"  # Correct path relative to demo root
    )
    result = predict_emotion_cnn(sample_wav)
    print("Predicted:", result["predicted"])
    print("Top K:", result["top_k"])
