import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

from preprocess import clean_text
from load_data import load_goemotions

EMOTIONS = [
    "anger", "disgust", "fear", "joy",
    "sadness", "surprise", "neutral"
]

def get_dominant_emotion(row):
    active = [emo for emo in EMOTIONS if row[emo] == 1]
    return active[0] if active else "neutral"

# Load data
df = load_goemotions()
df["text"] = df["text"].astype(str).apply(clean_text)

# Label handling
if "label" in df.columns:
    y = df["label"]
else:
    EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    df["label"] = df.apply(get_dominant_emotion, axis=1)
    y = df["label"]

X = df["text"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Base pipeline
base_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    ))
])

# Fit base model
base_pipeline.fit(X_train, y_train)

# 🔥 CALIBRATION STEP
calibrated_model = CalibratedClassifierCV(
    base_pipeline,
    method="sigmoid",   # Platt Scaling (best for small/medium data)
    cv=5
)

calibrated_model.fit(X_train, y_train)

# Evaluation
y_pred = calibrated_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save calibrated model
joblib.dump(calibrated_model, "text_emotion_model_calibrated.pkl")

print("✅ Calibrated Text Emotion Model saved")
