import streamlit as st
import numpy as np
import librosa
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
from contextlib import redirect_stdout
import joblib


def calculate_window_features(y_window, sr):
    """
    Calculates 26 mean audio features for a given audio window.
    """
    nan_features = {
        "chroma_stft": np.nan,
        "rms": np.nan,
        "spectral_centroid": np.nan,
        "spectral_bandwidth": np.nan,
        "rolloff": np.nan,
        "zero_crossing_rate": np.nan,
    }
    for i in range(1, 21):
        nan_features[f"mfcc{i}"] = np.nan

    if len(y_window) == 0:
        return nan_features

    try:
        chroma_stft = librosa.feature.chroma_stft(y=y_window, sr=sr).mean()
        rms_mean = librosa.feature.rms(y=y_window).mean()
        spectral_centroid_mean = librosa.feature.spectral_centroid(
            y=y_window, sr=sr
        ).mean()
        spectral_bandwidth_mean = librosa.feature.spectral_bandwidth(
            y=y_window, sr=sr
        ).mean()
        rolloff_mean = librosa.feature.spectral_rolloff(y=y_window, sr=sr).mean()
        zero_crossing_rate_mean = librosa.feature.zero_crossing_rate(y_window).mean()
        mfccs_means = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=20).mean(axis=1)

        features_dict = {
            "chroma_stft": chroma_stft,
            "rms": rms_mean,
            "spectral_centroid": spectral_centroid_mean,
            "spectral_bandwidth": spectral_bandwidth_mean,
            "rolloff": rolloff_mean,
            "zero_crossing_rate": zero_crossing_rate_mean,
        }
        for i in range(20):
            features_dict[f"mfcc{i+1}"] = (
                mfccs_means[i] if mfccs_means.size == 20 else np.nan
            )

        return features_dict
    except Exception:
        return nan_features


def process_audio(y, sr, window_duration_sec=1.0):
    """
    Segments an audio signal into windows and extracts features for each window.
    """
    samples_per_window = int(window_duration_sec * sr)
    all_window_features = []

    if len(y) < samples_per_window:
        return []

    num_full_windows = len(y) // samples_per_window

    for i in range(num_full_windows):
        start_sample_idx = i * samples_per_window
        end_sample_idx = start_sample_idx + samples_per_window
        y_window = y[start_sample_idx:end_sample_idx]
        segment_features = calculate_window_features(y_window, sr)
        all_window_features.append(segment_features)

    return all_window_features


def get_model_summary(model):
    """Captures model.summary() output into a string."""
    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    summary_string = stream.getvalue()
    return summary_string


@st.cache_resource
def load_all_models_and_scaler():
    """Loads all models and the scaler, cached for performance."""
    model_files = {
        "LSTM": "models/Optimized_LSTM_model.h5",
        "GRU": "models/Optimized_GRU_model.h5",
        "BiLSTM": "models/Optimized_BiLSTM_model.h5",
        "BiGRU": "models/Optimized_BiGRU_model.h5",
    }

    # Keras needs to know how to interpret the 'Orthogonal' initializer when loading
    custom_objects = {"Orthogonal": tf.keras.initializers.Orthogonal}

    models = {}
    for name, file_path in model_files.items():
        models[name] = load_model(file_path, custom_objects=custom_objects)

    scaler = joblib.load("models/scaler.gz")
    return models, scaler


def main():
    st.set_page_config(layout="wide")
    # app_styling()

    st.title("Deepfake Audio Detection")
    st.write("Upload an audio file to determine if it is authentic or a deepfake.")

    with st.container():
        uploaded_file = st.file_uploader(
            "Choose an audio file", type=["wav", "mp3"], label_visibility="collapsed"
        )

    try:
        models, scaler = load_all_models_and_scaler()
    except Exception as e:
        st.error(f"Error loading models or scaler: {e}")
        st.error(
            "Please ensure the model files (.h5) and the 'scaler.gz' file are in the 'models' folder relative to the Streamlit app."
        )
        return

    if "ran_prediction" not in st.session_state:
        st.session_state.ran_prediction = False

    if uploaded_file is not None and not st.session_state.ran_prediction:
        st.audio(uploaded_file, format="audio/wav")

        with st.spinner("Analyzing audio..."):
            try:
                audio_bytes = io.BytesIO(uploaded_file.read())
                y, sr = librosa.load(audio_bytes, sr=None)
            except Exception as e:
                st.error(f"Error loading audio file: {e}")
                st.error(
                    "The uploaded file might be corrupted or in an unsupported format."
                )
                return

            window_features = process_audio(y, sr)

            if not window_features:
                st.warning(
                    "Audio file is too short to be analyzed. Please upload a file longer than 1 second."
                )
                return

            df_features = pd.DataFrame(window_features).dropna()

            if df_features.empty:
                st.warning("Could not extract any valid features from the audio file.")
                return

            scaled_features = scaler.transform(df_features)
            X = np.reshape(
                scaled_features, (scaled_features.shape[0], 1, scaled_features.shape[1])
            )

        st.header("Analysis Results")

        cols = st.columns(len(models))

        for i, (model_name, model) in enumerate(models.items()):
            probabilities = model.predict(X).flatten()
            avg_probability = np.mean(probabilities)
            is_real = avg_probability > 0.5
            confidence = avg_probability if is_real else 1 - avg_probability

            with cols[i]:
                st.metric(
                    label=f"{model_name} Prediction",
                    value="Real" if is_real else "Deepfake",
                    delta=f"Confidence: {confidence:.2%}",
                    delta_color="normal" if is_real else "inverse",
                )

        st.session_state.ran_prediction = True

        st.subheader("Model Summaries")
        for model_name, model in models.items():
            with st.expander(f"View {model_name} Summary"):
                summary_string = get_model_summary(model)
                st.text(summary_string)

        st.subheader("Extracted Features (Unscaled)")
        st.dataframe(df_features)

    elif uploaded_file is None:
        st.session_state.ran_prediction = False


if __name__ == "__main__":
    main()
