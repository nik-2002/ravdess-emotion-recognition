"""PyCaret MFCC-based inference utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import librosa
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PYCARET_MODEL_PATH = MODELS_DIR / "ravdess_mfcc_pycaret_best.pkl"

TARGET_SR = 22050
N_MFCC = 20

_pycaret_model = None
_pycaret_classes: List[str] | None = None


def _feature_names(prefix: str) -> Iterable[str]:
    for idx in range(1, N_MFCC + 1):
        yield f"{prefix}{idx}_mean"
        yield f"{prefix}{idx}_std"


FEATURE_COLUMNS: List[str] = list(_feature_names("mfcc")) + list(
    _feature_names("delta_mfcc")
)


def _ensure_model_loaded() -> None:
    """Load the PyCaret pipeline lazily to avoid repeated disk reads."""
    global _pycaret_model, _pycaret_classes
    if _pycaret_model is not None:
        return

    if not PYCARET_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"PyCaret model not found at {PYCARET_MODEL_PATH}. "
            "Ensure the .pkl file is present."
        )

    _pycaret_model = joblib.load(PYCARET_MODEL_PATH)
    _pycaret_classes = _extract_class_labels(_pycaret_model)
    if not _pycaret_classes:
        raise RuntimeError(
            "Unable to determine class labels from the PyCaret pipeline."
        )


def _extract_class_labels(model: Any) -> List[str]:
    """Attempt to pull the ordered class labels off the pipeline."""
    if hasattr(model, "classes_"):
        classes = getattr(model, "classes_", None)
        if classes is not None:
            return list(classes)

    steps = getattr(model, "steps", None)
    if steps:
        final_estimator = steps[-1][1]
        if hasattr(final_estimator, "classes_"):
            classes = getattr(final_estimator, "classes_", None)
            if classes is not None:
                return list(classes)

    named_steps = getattr(model, "named_steps", None)
    if named_steps:
        trained = named_steps.get("trained_model")
        if trained is not None and hasattr(trained, "classes_"):
            classes = getattr(trained, "classes_", None)
            if classes is not None:
                return list(classes)

    return []


def _extract_mfcc_dataframe(wav_path: Path) -> pd.DataFrame:
    """Compute MFCC + delta summary statistics to feed into the MLP."""
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file {wav_path} was not found.")

    audio, _ = librosa.load(str(wav_path), sr=TARGET_SR)
    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)

    features: Dict[str, float] = {}

    def _populate(feature_matrix: np.ndarray, prefix: str) -> None:
        for idx in range(N_MFCC):
            coeff = idx + 1
            coeff_values = feature_matrix[idx]
            mean = float(np.mean(coeff_values))
            std = float(np.std(coeff_values))
            if np.isnan(mean):
                mean = 0.0
            if np.isnan(std):
                std = 0.0
            features[f"{prefix}{coeff}_mean"] = mean
            features[f"{prefix}{coeff}_std"] = std

    _populate(mfcc, "mfcc")
    _populate(delta, "delta_mfcc")

    # Ensure column order matches what the pipeline saw during training.
    ordered = {column: features.get(column, 0.0) for column in FEATURE_COLUMNS}
    return pd.DataFrame([ordered])


def predict_emotion_mlp(wav_path: str | Path) -> Dict[str, Any]:
    """Run the MFCC PyCaret MLP on a WAV file and return probabilities."""
    _ensure_model_loaded()
    assert _pycaret_model is not None
    assert _pycaret_classes is not None

    df = _extract_mfcc_dataframe(Path(wav_path))
    probabilities: np.ndarray | None = None

    if hasattr(_pycaret_model, "predict_proba"):
        probabilities = _pycaret_model.predict_proba(df)[0]

    if probabilities is None:
        predicted = _pycaret_model.predict(df)[0]
        return {
            "predicted": str(predicted),
            "probs": {},
            "top_k": [],
        }

    class_probs: List[Tuple[str, float]] = []
    for label, prob in zip(_pycaret_classes, probabilities):
        class_probs.append((label, float(prob)))

    class_probs.sort(key=lambda entry: entry[1], reverse=True)
    top_label = class_probs[0][0] if class_probs else None

    return {
        "predicted": top_label,
        "probs": {label: prob for label, prob in class_probs},
        "top_k": class_probs[:3],
    }
