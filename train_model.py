"""
train_model.py (v2)
--------------------
Trains a Logistic Regression with:
  - Richer TF-IDF features (char n-grams + word n-grams)
  - Class-balanced training
  - CalibratedClassifierCV for reliable probabilities
  - Cross-validation reporting
  - Saves model/ artifacts for app.py

Run: python train_model.py
"""

import sys
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, brier_score_loss,
)
from sklearn.pipeline import FeatureUnion

from utils import clean_text

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH       = Path("data/news_dataset.csv")
MODEL_DIR       = Path("model")
MODEL_PATH      = MODEL_DIR / "logistic_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# ── Feature config ────────────────────────────────────────────────────────────
WORD_TFIDF = dict(
    max_features=15_000,
    ngram_range=(1, 3),      # unigrams, bigrams, trigrams
    min_df=1,
    max_df=0.97,
    sublinear_tf=True,
    analyzer="word",
)

CHAR_TFIDF = dict(
    max_features=8_000,
    ngram_range=(3, 5),      # character n-grams for stylometric signals
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
    analyzer="char_wb",
)

LR_PARAMS = dict(
    C=0.8,
    max_iter=2000,
    solver="lbfgs",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


def load_data():
    if not DATA_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATA_PATH}")
        print("  → Run: python generate_dataset.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    required = {"text", "label"}
    if not required.issubset(df.columns):
        print(f"[ERROR] CSV must have columns: {required}")
        sys.exit(1)

    df = df.dropna(subset=["text", "label"])
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].str.upper().str.strip()
    df = df[df["label"].isin(["FAKE", "REAL"])].reset_index(drop=True)

    print(f"[DATA] {len(df)} samples  |  FAKE: {(df.label=='FAKE').sum()}  REAL: {(df.label=='REAL').sum()}")
    return df


def build_vectorizer():
    """
    Combine word-level and character-level TF-IDF features.
    This captures both semantic content AND stylometric patterns.
    """
    word_vec = TfidfVectorizer(**WORD_TFIDF)
    char_vec = TfidfVectorizer(**CHAR_TFIDF)
    return word_vec, char_vec


def train():
    print("\n" + "=" * 55)
    print("  VERILENS AI — MODEL TRAINING v2")
    print("=" * 55)

    df = load_data()

    # Clean text
    print("\n[1/5] Cleaning text...")
    df["clean"] = df["text"].apply(clean_text)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=0.20, random_state=42, stratify=df["label"]
    )
    print(f"      Train: {len(X_train)}  Test: {len(X_test)}")

    # Vectorize
    print("\n[2/5] Building feature matrix (word + char TF-IDF)...")
    word_vec, char_vec = build_vectorizer()

    from scipy.sparse import hstack
    Xw_train = word_vec.fit_transform(X_train)
    Xw_test  = word_vec.transform(X_test)

    Xc_train = char_vec.fit_transform(X_train)
    Xc_test  = char_vec.transform(X_test)

    X_train_full = hstack([Xw_train, Xc_train])
    X_test_full  = hstack([Xw_test,  Xc_test])

    total_feats = X_train_full.shape[1]
    print(f"      Feature dims: {total_feats:,}  (word: {Xw_train.shape[1]:,} + char: {Xc_train.shape[1]:,})")

    # Train
    print("\n[3/5] Training Logistic Regression...")
    base_lr = LogisticRegression(**LR_PARAMS)
    base_lr.fit(X_train_full, y_train)

    # Calibrate
    print("\n[4/5] Calibrating confidence scores (Platt scaling)...")
    calibrated_model = CalibratedClassifierCV(base_lr, cv=5, method="sigmoid")
    calibrated_model.fit(X_train_full, y_train)

    # Evaluate
    print("\n[5/5] Evaluating...")
    y_pred  = calibrated_model.predict(X_test_full)
    y_proba = calibrated_model.predict_proba(X_test_full)[:, 1]   # prob of REAL

    acc   = accuracy_score(y_test, y_pred)
    auc   = roc_auc_score((y_test == "REAL").astype(int), y_proba)
    brier = brier_score_loss((y_test == "REAL").astype(int), y_proba)

    print(f"\n      Accuracy  : {acc*100:.2f}%")
    print(f"      ROC-AUC   : {auc:.4f}  (higher = better discrimination)")
    print(f"      Brier     : {brier:.4f} (lower = better calibration)")
    print("\n" + classification_report(y_test, y_pred))

    # Save
    MODEL_DIR.mkdir(exist_ok=True)

    # We save the calibrated model directly.
    # For vectorization, save a combined "vectorizer" dict
    # so utils.py can reconstruct features at inference time.
    joblib.dump(calibrated_model, MODEL_PATH)

    # Save a wrapper that applies both vectorizers + hstack at inference
    from _vectorizer_wrapper import CombinedVectorizer
    combined_vec = CombinedVectorizer(word_vec, char_vec)
    joblib.dump(combined_vec, VECTORIZER_PATH)

    print(f"\n[✓] Model saved     → {MODEL_PATH}")
    print(f"[✓] Vectorizer saved → {VECTORIZER_PATH}")
    print("\n[DONE] Run: streamlit run app.py\n")
    return acc


if __name__ == "__main__":
    train()
