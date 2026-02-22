import joblib
from models.text.preprocess import clean_text
import numpy as np

model = joblib.load("models/text/text_emotion_model_calibrated.pkl")


def predict_emotion(text):
    text = clean_text(text)

    probs = model.predict_proba([text])[0]
    labels = model.classes_

    idx = np.argmax(probs)
    emotion = labels[idx]
    confidence = probs[idx]

    return emotion, float(confidence)

if __name__ == "__main__":
    print(predict_emotion("I am very very excited today."))