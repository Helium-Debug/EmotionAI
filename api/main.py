from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore

from models.text.infer import predict_emotion as predict_text
from models.audio.infer import predict_emotion as predict_audio
from fusion.fusion_logic import fuse_predictions

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Multimodal Emotion Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load video model once at startup
VIDEO_MODEL_PATH = "models/video/video_emotion_model.h5"
video_model = load_model(VIDEO_MODEL_PATH)

EMOTIONS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}


# -----------------------------
# TEXT ENDPOINT
# -----------------------------
@app.post("/emotion/text")
async def emotion_text(text: dict):
    content = text.get("text")
    emotion, confidence = predict_text(content)

    return {
        "emotion": emotion,
        "confidence": confidence
    }


# -----------------------------
# AUDIO ENDPOINT
# -----------------------------
@app.post("/emotion/audio")
async def emotion_audio(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    emotion, confidence = predict_audio(tmp_path)
    os.remove(tmp_path)

    return {
        "emotion": emotion,
        "confidence": confidence
    }


# -----------------------------
# IMAGE ENDPOINT
# -----------------------------
@app.post("/emotion/image")
async def emotion_image(file: UploadFile = File(...)):

    image = Image.open(file.file)
    img = np.array(image.convert("L"))
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0

    preds = video_model.predict(img)[0]
    emotion = EMOTIONS[np.argmax(preds)]
    confidence = float(np.max(preds))

    return {
        "emotion": emotion,
        "confidence": confidence
    }


# -----------------------------
# FUSION ENDPOINT
# -----------------------------
@app.post("/emotion/fusion")
async def emotion_fusion(
    text: str = None,
    audio: UploadFile = File(None),
    image: UploadFile = File(None)
):

    predictions = {}

    # Text
    if text:
        predictions["text"] = predict_text(text)

    # Audio
    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        predictions["audio"] = predict_audio(tmp_path)
        os.remove(tmp_path)

    # Image
    if image:
        img = Image.open(image.file)
        img = np.array(img.convert("L"))
        img = cv2.resize(img, (48, 48))
        img = img.reshape(1, 48, 48, 1) / 255.0

        preds = video_model.predict(img)[0]
        emotion = EMOTIONS[np.argmax(preds)]
        confidence = float(np.max(preds))
        predictions["video"] = (emotion, confidence)

    if not predictions:
        return JSONResponse(
            content={"error": "No input provided"},
            status_code=400
        )

    fusion_result = fuse_predictions(predictions)

    return fusion_result
