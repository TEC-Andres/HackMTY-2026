"""
Trains the final resonance-stability model and saves it to ../model.joblib.

Uses train+val combined (both already validated separately — see
../README.md) since the real judging set is a third, hidden split
anyway; more training data only helps at this point.

Run (from anywhere):
    python train_model.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent  # resonance-detector/
sys.path.insert(0, str(ROOT))
from resonance_core import FEATURE_NAMES  # noqa: E402

MODEL_PATH = ROOT / "model.joblib"


def main():
    train = pd.read_csv(ROOT / "resonance_features_train_full.csv").dropna(subset=FEATURE_NAMES)
    val = pd.read_csv(ROOT / "resonance_features_val_full.csv").dropna(subset=FEATURE_NAMES)
    full = pd.concat([train, val], ignore_index=True)

    X = full[FEATURE_NAMES]
    y = (full.label == "synthetic").astype(int)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"Cross-val AUC on combined train+val: {scores.mean():.3f} +/- {scores.std():.3f}")

    clf.fit(X, y)
    joblib.dump({"pipeline": clf, "feature_names": FEATURE_NAMES}, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
