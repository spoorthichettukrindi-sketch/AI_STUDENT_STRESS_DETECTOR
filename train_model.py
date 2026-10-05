from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent

DATA_PATH = BASE / "student_stress.csv"
MODEL_PATH = BASE / "stress_model.joblib"

FEATURES = [
    "sleep_hours",
    "study_hours",
    "screen_time_hours",
    "physical_activity_days",
    "academic_workload",
    "exam_frequency",
    "mood_score",
    "social_support_score",
    "attendance_percent"
]

TARGET = "stress_level"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Separate input and output
X = df[FEATURES]
y = df[TARGET]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Models
models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = {}

# Train and evaluate
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    results[name] = accuracy


# Select best model
best_model_name = max(
    results,
    key=results.get
)

best_model = models[best_model_name]

# Save model
joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nBest Model:", best_model_name)
print("Model saved successfully.")