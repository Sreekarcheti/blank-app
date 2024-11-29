import streamlit as st
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Functions for spectrogram generation and visualization
def get_mel_spectrogram(y: np.ndarray, sr: int = 22050, n_mels: int = 128):
    mel_spec = lb.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = lb.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def plot_mel_spectrogram(mel_spec_db, sr=22050, title="Mel Spectrogram"):
    plt.figure(figsize=(10, 4))
    lb.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt.gcf())  # Display in Streamlit

# Load pre-trained CNN model
model = load_model('path_to_your_saved_model.h5')  # Provide the correct path
le = LabelEncoder().fit([0, 1])  # Assuming 0 = real, 1 = fake

# Streamlit App UI
st.title("Deepfake Audio Detection")
st.header("Upload an Audio File")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
if uploaded_file is not None:
    # Load and process audio
    st.audio(uploaded_file, format="audio/wav")
    y, sr = lb.load(uploaded_file, sr=None)
    st.write(f"Loaded audio with {len(y)} samples at {sr} Hz")

    # Generate and display Mel Spectrogram
    st.subheader("Mel Spectrogram")
    mel_spec = get_mel_spectrogram(y, sr)
    plot_mel_spectrogram(mel_spec, sr, title="Mel Spectrogram of Uploaded Audio")

    # Prepare spectrogram for model input
    mel_spec_resized = np.expand_dims(mel_spec, axis=(0, -1))  # Reshape for CNN input

    # Prediction
    st.subheader("Prediction")
    prediction = model.predict(mel_spec_resized)
    label = le.inverse_transform([int(round(prediction[0][0]))])[0]
    st.write(f"The model predicts: **{'Fake Audio' if label else 'Real Audio'}**")
