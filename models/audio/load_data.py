import os
import numpy as np
from extract_features import extract_features

file_path = "Audio_archive"

EMOTION_MAP = {
    "Angry": "angry",
    "Disgusted": "disgust",
    "Fearful": "fear",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprised": "surprise"
}

def load_custom_audio_dataset(base_path):
    X, y = [], []

    emotions_path = os.path.join(base_path, "Emotions")

    for folder in os.listdir(emotions_path):
        emotion = EMOTION_MAP.get(folder)
        if emotion is None:
            continue

        folder_path = os.path.join(emotions_path, folder)

        for file in os.listdir(folder_path):
            if file.lower().endswith(".wav"):
                file_path = os.path.join(folder_path, file)

                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(emotion)
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")

    return np.array(X), np.array(y)
