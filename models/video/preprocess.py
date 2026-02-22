import numpy as np

EMOTIONS = [
    "angry", "disgust", "fear",
    "happy", "sad", "surprise", "neutral"
]

def preprocess_pixels(pixel_string):
    pixels = np.array(pixel_string.split(), dtype="float32")
    pixels = pixels.reshape(48, 48, 1)
    pixels /= 255.0
    return pixels
