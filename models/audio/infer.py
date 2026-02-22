import joblib
import numpy as np
from models.audio.extract_features import extract_features


model = joblib.load("models/audio/audio_emotion_model.pkl")

def predict_emotion(audio_path):
    features = extract_features(audio_path).reshape(1, -1)

    probs = model.predict_proba(features)[0]
    labels = model.classes_

    idx = np.argmax(probs)
    return labels[idx], float(probs[idx])

if __name__ == "__main__":
    # CHANGE THIS PATH TO ANY TEST AUDIO FILE
    test_audio = "my_audio_data/Emotions/Happy/sample.wav"
    print(predict_emotion(test_audio))
