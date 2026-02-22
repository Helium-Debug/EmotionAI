import librosa
import numpy as np

file_path = "Audio_archive"
def extract_features(file_path):
    # Load audio (force same sample rate)
    y, sr = librosa.load(file_path, sr=22050, duration=3, offset=0.5)

    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)

    # Combine all features
    return np.hstack([mfcc_mean, chroma_mean, mel_mean])
