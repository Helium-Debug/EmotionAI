import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

EMOTIONS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

model = load_model("video_emotion_model.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48,48))
    face = face.reshape(1,48,48,1) / 255.0

    preds = model.predict(face)[0]
    emotion = EMOTIONS[np.argmax(preds)]
    confidence = np.max(preds)

    cv2.putText(
        frame,
        f"{emotion} ({confidence:.2f})",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
