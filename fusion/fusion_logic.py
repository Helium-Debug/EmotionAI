EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

# Default modality weights
DEFAULT_WEIGHTS = {
    "text": 0.3,
    "audio": 0.3,
    "video": 0.4
}


def fuse_predictions(predictions, weights=DEFAULT_WEIGHTS):
    """
    predictions = {
        "text": ("happy", 0.75),
        "audio": ("sad", 0.60),
        "video": ("happy", 0.85)
    }
    """

    score_board = {emotion: 0.0 for emotion in EMOTIONS}

    used_modalities = []

    for modality, result in predictions.items():
        if result is None:
            continue

        emotion, confidence = result
        weight = weights.get(modality, 0)

        score_board[emotion] += confidence * weight
        used_modalities.append(modality)

    final_emotion = max(score_board, key=score_board.get)
    final_confidence = score_board[final_emotion]

    return {
        "final_emotion": final_emotion,
        "confidence": round(final_confidence, 3),
        "modalities_used": used_modalities,
        "detailed_scores": score_board
    }
