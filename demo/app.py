"""Streamlit interface for the RAVDESS emotion demo."""

# To run this app:
# 1. Activate the ravdess-audio environment.
# 2. Install streamlit if needed:  pip install streamlit
# 3. From the project root, run:
#       streamlit run app.py

from pathlib import Path
import io
import time
import os
import subprocess

import pandas as pd
import streamlit as st
from PIL import Image
from audiorecorder import audiorecorder

from src.mfcc_pycaret_inference import predict_emotion_mlp
from src.ravdess_inference import predict_emotion_cnn


PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
CNN_PLOTS_DIR = ASSETS_DIR / "plots" / "cnn"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"

EMOTIONS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

STATEMENTS = {
    "Statement 1": "Kids are talking by the door.",
    "Statement 2": "Dogs are sitting by the door.",
}

TARGET_SR = 22050
Image.MAX_IMAGE_PIXELS = None

MODEL_OPTIONS = [
    {
        "label": "Mel spectrogram CNN (default)",
        "key": "cnn",
        "runner": predict_emotion_cnn,
        "description": "Deep CNN trained on mel spectrogram images of the RAVDESS clips.",
    },
    {
        "label": "MFCC + PyCaret MLP",
        "key": "mlp",
        "runner": predict_emotion_mlp,
        "description": "Classical MLP fed with MFCC statistics and trained via PyCaret.",
    },
]
MODEL_OPTIONS_BY_LABEL = {option["label"]: option for option in MODEL_OPTIONS}


def convert_to_wav(input_path: Path) -> Path:
    """Convert arbitrary audio to mono 22050 Hz WAV using ffmpeg."""
    input_path = Path(input_path)
    wav_path = input_path.with_suffix(".wav")
    
    # Handle case where input and output paths are the same
    if input_path.resolve() == wav_path.resolve():
        temp_path = wav_path.with_stem(wav_path.stem + "_temp")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ar",
            str(TARGET_SR),
            "-ac",
            "1",
            str(temp_path),
        ]
        subprocess.run(cmd, check=True)
        temp_path.replace(wav_path)
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ar",
            str(TARGET_SR),
            "-ac",
            "1",
            str(wav_path),
        ]
        subprocess.run(cmd, check=True)
        
    return wav_path


def _write_audio_bytes(filename: str, audio_bytes: bytes) -> Path:
    """Persist arbitrary audio bytes in the uploads directory."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    path = UPLOADS_DIR / filename
    with open(path, "wb") as file_handle:
        file_handle.write(audio_bytes)
    return path


def _audiosegment_to_wav_bytes(recorded_audio: object | None) -> bytes | None:
    """Convert a pydub AudioSegment from streamlit-audiorecorder to raw WAV bytes."""
    if recorded_audio is None:
        return None
    try:
        length = len(recorded_audio)
    except TypeError:
        length = 0
    if not length:
        return None

    buffer = io.BytesIO()
    # `export` is provided by pydub.AudioSegment and writes WAV bytes to buffer
    recorded_audio.export(buffer, format="wav")
    return buffer.getvalue()


def image_placeholder(label: str, height: int = 200) -> None:
    st.markdown(
        f"""
        <div style="
            border: 2px dashed #888;
            border-radius: 8px;
            height: {height}px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #888;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_cnn_plot(filename: str, caption: str) -> None:
    """Display a CNN visualization image without emitting pillow warnings."""
    image_path = CNN_PLOTS_DIR / filename
    with Image.open(image_path) as pil_image:
        st.image(
            pil_image.copy(),
            caption=caption,
            width="stretch",
        )


page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "🎙 Live Demo",
        "⚙️ How We Built It",
        "📊 Dataset & Links",
    ],
)

st.set_page_config(
    page_title="RAVDESS Emotion Demo",
    page_icon="🎭",
    layout="wide",
)

def load_css(file_path: Path) -> None:
    """Load a CSS file and inject it into the Streamlit app."""
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css(PROJECT_ROOT / "assets" / "style.css")

if page == "🏠 Overview":
    st.markdown(
        """
        <div class="custom-card">
            <h1 style="margin-bottom: 0px;">RAVDESS</h1>
            <h2 style="margin-top: 0px; margin-bottom: 1rem; color: #cb785c;">Speech Emotion Recognition 🎭</h2>
            <p style="font-size: 1.1rem; max-width: 800px;">
                This project uses <b>deep learning</b> to recognize emotions from short speech clips.
                We trained a <b>Convolutional Neural Network (CNN)</b> on the
                <b>RAVDESS Emotional Speech Audio</b> dataset. The model listens to a spoken sentence,
                converts it into a melspectrogram, and predicts one of 8 emotions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("What you can do")
        st.info(
            "**1. Read** one of the official RAVDESS sentences.\n\n"
            "**2. Act** it out with an emotion of your choice.\n\n"
            "**3. Record** your voice and upload it.\n\n"
            "**4. See** which emotion the model thinks you sounded like."
        )

        st.caption(
            "Tip: The closer your recording is to the original dataset "
            "(clear speech, little noise, similar sentence length), "
            "the easier it is for the model to guess correctly."
        )

    with col2:
        st.subheader("Model Performance")
        tab1, tab2 = st.tabs(["Confusion Matrix", "ROC Curve"])
        with tab1:
            show_cnn_plot(
                "ravdess_melspec_confusion_matrix.png",
                "CNN confusion matrix on mel spectrogram inputs",
            )
        with tab2:
            show_cnn_plot(
                "ravdess_melspec_roc_curve.png",
                "ROC curve across emotions",
            )

elif page == "🎙 Live Demo":
    st.markdown("## 🎙 Live Demo")
    st.caption("Test the model with your own voice!")

    main_col1, main_col2 = st.columns([1, 1], gap="large")

    with main_col1:
        st.markdown("### 1. Setup")
        with st.container(border=True):
            st.markdown("**Step A: Choose a sentence**")
            statement_options = list(STATEMENTS.values())
            statement_text = st.selectbox(
                "Read this sentence:",
                statement_options,
                index=0,
            )

            st.markdown("---")
            st.markdown("**Step B: Choose target emotion**")
            target_emotion = st.selectbox(
                "Emulate this emotion:",
                EMOTIONS,
                index=5, # Neutral default
            )

            st.markdown("---")
            st.markdown("**Step C: Choose Model**")
            model_label = st.radio(
                "Inference model:",
                [option["label"] for option in MODEL_OPTIONS],
                index=0,
                horizontal=True,
            )
            selected_model = MODEL_OPTIONS_BY_LABEL[model_label]
            st.caption(selected_model["description"])

    with main_col2:
        st.markdown("### 2. Record & Analyze")
        
        tab_rec, tab_up = st.tabs(["Record", "Upload"])
        
        recorded_audio = None
        uploaded_file = None
        
        with tab_rec:
            st.info("Click the microphone to start recording.")
            recorded_audio = audiorecorder("Click to record", "Recording...")
        
        with tab_up:
            uploaded_file = st.file_uploader(
                "Upload a recording (wav, m4a, mp3)", type=["wav", "m4a", "mp3"]
            )

        analyze_button_placeholder = st.empty()
        wav_path: Path | None = None
        audio_source_label: str | None = None

        recorded_bytes = _audiosegment_to_wav_bytes(recorded_audio)
        if recorded_bytes:
            st.success("Audio captured!")
            st.audio(recorded_bytes, format="audio/wav")
            filename = f"recording_{int(time.time() * 1000)}.wav"
            raw_path = _write_audio_bytes(filename, recorded_bytes)
            wav_path = convert_to_wav(raw_path)
            audio_source_label = "Recorded in browser"
        elif uploaded_file is not None:
            st.success("File uploaded!")
            uploaded_bytes = uploaded_file.getvalue()
            st.audio(uploaded_bytes)
            raw_path = _write_audio_bytes(uploaded_file.name, uploaded_bytes)
            wav_path = convert_to_wav(raw_path)
            audio_source_label = "Uploaded file"

    if wav_path is not None:
        st.markdown("---")
        if analyze_button_placeholder.button("Analyze Emotion", type="primary", use_container_width=True):
            try:
                with st.spinner("Analyzing audio..."):
                    predict_fn = selected_model["runner"]
                    result = predict_fn(wav_path)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Model run failed: {exc}")
            else:
                predicted = result.get("predicted")
                probs = result.get("probs", {})
                
                # Results Display
                st.markdown("### 🔍 Model Prediction")
                
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    if predicted:
                        is_match = predicted == target_emotion
                        match_color = "#4ade80" if is_match else "#fbbf24"
                        match_text = "Match!" if is_match else "Mismatch"
                        
                        st.markdown(
                            f"""
                            <div style="background-color: {match_color}; padding: 20px; border-radius: 10px; text-align: center; color: #1e293b;">
                                <h3 style="margin:0;">{predicted.upper()}</h3>
                                <p style="margin:0;">Detected Emotion</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        if not is_match:
                            st.caption(f"You aimed for: **{target_emotion}**")
                        else:
                            st.caption("Great acting!")
                            
                with res_col2:
                   if probs:
                        probs_df = pd.DataFrame(
                            list(probs.items()), columns=["Emotion", "Probability"]
                        ).sort_values("Probability", ascending=False)
                        st.bar_chart(probs_df.set_index("Emotion"), height=200)

    else:
         if analyze_button_placeholder.button("Analyze Emotion", disabled=True):
            pass

elif page == "⚙️ How We Built It":
    st.markdown("## ⚙️ How We Built the Model")

    with st.container(border=True):
        st.markdown("### 1. Dataset and Labels")
        st.write(
            "We trained the model on the **RAVDESS Emotional Speech Audio** dataset. "
            "RAVDESS contains recordings of 24 actors saying two simple sentences with "
            "8 different emotions at two intensity levels. The speech part has 1,440 clips."
        )

    with st.container(border=True):
        st.markdown("### 2. Preprocessing: Waveform to Mel Spectrogram")
        col1, col2 = st.columns([1, 1])
        with col1:
             st.write(
                "- Conversion to mono, 22,050 Hz WAV.\n"
                "- Computation of **mel spectrogram** (frequency vs time) using `librosa`.\n"
                "- Conversion to **log scale** (dB) and normalization."
            )
        with col2:
            show_cnn_plot(
                "ravdess_melspec_samples.png",
                "Sample mel spectrograms used for CNN training",
            )

    with st.container(border=True):
        st.markdown("### 3. CNN Architecture")
        st.write(
            "Our model uses a Convolutional Neural Network (CNN) designed for image classification, adapted for spectrograms."
        )
        show_cnn_plot(
            "cnn_pipeline_visualization.png",
            "CNN architecture pipeline for emotion classification",
        )
        st.info(
            "- **Input**: Single-channel mel spectrogram.\n"
            "- **Layers**: Convolutional + ReLU + Pooling layers to extract features.\n"
            "- **Output**: Probability distribution over 8 emotions."
        )

    with st.container(border=True):
        st.markdown("### 4. Inference Pipeline")
        st.image(str(CNN_PLOTS_DIR / "Inference_Pipeline.png"), caption="Inference pipeline from recording to prediction")

    with st.container(border=True):
        st.markdown("### 5. Training Results")
        image_placeholder("Placeholder: training curve (training_curve.png)", height=180)

elif page == "📊 Dataset & Links":
    st.header("📊 Dataset & Links")

    st.subheader("RAVDESS Emotional Speech Audio")
    st.write(
        "We use the **RAVDESS Emotional Speech Audio** dataset. "
        "Each clip is a professional actor reading one of two sentences with "
        "a specific emotion and intensity. The emotions are: "
        "**neutral, calm, happy, sad, angry, fearful, disgust, surprised**."
    )

    st.markdown(
        "- Kaggle (speech audio-only): "
        "[RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)\n"
        "- The original project page and full dataset (speech + song, audio + video) are linked from Kaggle."
    )

    st.subheader("Why we stick to two sentences")
    st.write(
        "RAVDESS uses only two carrier sentences:\n"
        "- \"**Kids are talking by the door.**\"\n"
        "- \"**Dogs are sitting by the door.**\"\n\n"
        "In the demo, we ask you to read one of these sentences to keep your recording "
        "as close as possible to the training data."
    )

    st.subheader("Limitations")
    st.markdown(
        "- The model was trained on **acted** emotions, not spontaneous speech.\n"
        "- Recordings in a noisy environment or with very different sentences can confuse the model.\n"
        "- The model only knows 8 emotions and will always choose one of them."
    )

    st.subheader("Emotion distribution in the RAVDESS speech subset")
    image_placeholder("Placeholder: emotion distribution chart (emotion_distribution.png)", height=220)
