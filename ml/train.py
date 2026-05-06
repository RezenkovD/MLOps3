from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model.joblib"
REFERENCE_PATH = ROOT / "reference_stats.joblib"

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def train_and_save(model_path: Path = MODEL_PATH, reference_path: Path = REFERENCE_PATH) -> float:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    joblib.dump(model, model_path)
    joblib.dump({"X": X_train, "feature_names": FEATURE_NAMES}, reference_path)
    return accuracy


if __name__ == "__main__":
    acc = train_and_save()
    print(f"Model trained. Test accuracy: {acc:.4f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved reference to: {REFERENCE_PATH}")
