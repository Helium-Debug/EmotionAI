import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

from load_data import load_custom_audio_dataset

# Dataset path (RELATIVE PATH)
DATASET_PATH = "Audio_archive"

print("📥 Loading dataset...")
X, y = load_custom_audio_dataset(DATASET_PATH)

print("📊 Class distribution:")
print(Counter(y))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest classifier
rf = RandomForestClassifier(
    n_estimators=100,      # was 300
    max_depth=20,          # was 30
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


rf.fit(X_train, y_train)

# 🔥 Confidence calibration
calibrated_rf = CalibratedClassifierCV(
    rf,
    method="sigmoid",
    cv="prefit"
)



calibrated_rf.fit(X_train, y_train)

# Evaluation
y_pred = calibrated_rf.predict(X_test)
print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(calibrated_rf, "audio_emotion_model.pkl")
print("\n✅ Audio emotion model trained and saved as audio_emotion_model.pkl")
