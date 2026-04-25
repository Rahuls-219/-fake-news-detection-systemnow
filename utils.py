"""
utils.py (v2)
--------------
Handles: text cleaning, model loading, prediction, calibrated confidence,
         keyword extraction, and insight generation.

Key improvements over v1:
  - Temperature-scaled confidence calibration
  - Richer stopword list
  - Semantic keyword scoring (TF-IDF weighted)
  - Improved insight generator with more signal patterns
"""

import re
import string
import joblib
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH      = Path("model/logistic_model.pkl")
VECTORIZER_PATH = Path("model/tfidf_vectorizer.pkl")

# ── Extended stopwords ────────────────────────────────────────────────────────
STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "not", "with", "this", "that", "are", "was",
    "be", "by", "as", "from", "its", "they", "we", "he", "she", "you",
    "i", "my", "your", "our", "their", "has", "have", "had", "been",
    "will", "would", "could", "should", "do", "does", "did", "so",
    "if", "all", "can", "more", "about", "up", "out", "than", "then",
    "into", "after", "over", "also", "new", "just", "now", "years",
    "said", "says", "one", "two", "three", "four", "five", "first",
    "last", "next", "many", "most", "some", "such", "like", "time",
    "very", "still", "well", "back", "even", "much", "go", "come",
    "get", "make", "know", "take", "see", "look", "want", "give",
    "use", "find", "tell", "ask", "seem", "feel", "try", "leave",
    "call", "keep", "let", "begin", "show", "hear", "play", "run",
    "move", "live", "believe", "hold", "bring", "happen", "write",
    "provide", "sit", "stand", "lose", "pay", "meet", "include",
    "continue", "set", "learn", "change", "lead", "understand",
    "watch", "follow", "stop", "create", "speak", "read", "spend",
    "grow", "open", "walk", "win", "offer", "remember", "love",
    "consider", "appear", "buy", "wait", "serve", "die", "send",
    "expect", "build", "stay", "fall", "cut", "reach", "kill",
    "remain", "suggest", "raise", "pass", "sell", "require", "report",
    "decide", "pull", "break", "receive", "agree", "support",
}

# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip URLs/numbers/punctuation, remove stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    """Return (model, vectorizer). Raises FileNotFoundError if not trained."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model files not found. Run: python train_model.py"
        )
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


# ── Confidence calibration ────────────────────────────────────────────────────
# Temperature scaling: T > 1 softens overconfident predictions
_TEMPERATURE = 1.4

def _calibrate(proba: np.ndarray, temperature: float = _TEMPERATURE) -> np.ndarray:
    """Apply temperature scaling to raw softmax probabilities."""
    logits = np.log(proba + 1e-12) / temperature
    exp    = np.exp(logits - logits.max())
    return exp / exp.sum()


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(text: str, model, vectorizer) -> dict | None:
    """
    Returns dict with:
      label, confidence, fake_prob, real_prob,
      top_keywords, insight, calibrated (bool)
    """
    if not text or not text.strip():
        return None
    if model is None or vectorizer is None:
        return None

    cleaned = clean_text(text)
    vector  = vectorizer.transform([cleaned])

    raw_proba = model.predict_proba(vector)[0]

    # Calibrate
    cal_proba              = _calibrate(raw_proba)
    fake_prob, real_prob   = cal_proba[0], cal_proba[1]

    label      = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = max(fake_prob, real_prob) * 100

    top_keywords = get_top_keywords(vector, vectorizer, n=5)
    insight      = generate_insight(label, confidence, top_keywords, text)

    return {
        "label":        label,
        "confidence":   round(confidence, 1),
        "fake_prob":    round(fake_prob * 100, 1),
        "real_prob":    round(real_prob * 100, 1),
        "top_keywords": top_keywords,
        "insight":      insight,
        "calibrated":   True,
    }


# ── Keyword extraction ────────────────────────────────────────────────────────
def get_top_keywords(vector, vectorizer, n: int = 5) -> list[str]:
    feature_names = vectorizer.get_feature_names_out()
    scores        = vector.toarray()[0]
    top_idx       = np.argsort(scores)[::-1][:n]
    keywords      = [feature_names[i] for i in top_idx if scores[i] > 0]
    return keywords or ["(no strong signals)"]


# ── Insight generator ─────────────────────────────────────────────────────────
SENSATIONAL_WORDS = {
    "shocking", "exposed", "breaking", "alert", "urgent", "revealed",
    "secret", "banned", "deleted", "hidden", "suppressed", "leaked",
    "whistleblower", "conspiracy", "miracle", "cure", "cover", "truth",
    "won't believe", "share before", "wake up", "they don't want",
    "mainstream media", "big pharma", "new world order", "deep state",
    "hoax", "scam", "fraud", "lies", "fake", "propaganda",
}

CREDIBILITY_WORDS = {
    "study", "research", "report", "university", "scientists", "data",
    "official", "government", "published", "confirmed", "announced",
    "percent", "survey", "analysis", "findings", "review", "evidence",
    "according", "researchers", "experts", "statistics", "journal",
    "peer-reviewed", "institute", "agency", "department",
}


def generate_insight(label: str, confidence: float, keywords: list, text: str) -> str:
    text_lower  = text.lower()
    sens_hits   = [w for w in SENSATIONAL_WORDS if w in text_lower]
    cred_hits   = [w for w in CREDIBILITY_WORDS if w in text_lower]
    kw_str      = ", ".join(f'"{k}"' for k in keywords[:3]) if keywords else "key terms"

    if label == "FAKE":
        strength = "strong" if confidence >= 80 else "moderate" if confidence >= 60 else "some"
        trigger  = f"sensational language ('{sens_hits[0]}')" if sens_hits else "unusual phrasing"
        return (
            f"⚠️ {strength.capitalize()} signals of misinformation detected. "
            f"The text uses {trigger} and terms like {kw_str} "
            f"that frequently appear in fabricated content. "
            f"Always verify with trusted sources before sharing."
        )
    else:
        trigger = f"credibility markers ('{cred_hits[0]}')" if cred_hits else "neutral, factual language"
        return (
            f"✅ This appears to be legitimate reporting. "
            f"It uses {trigger} and terms like {kw_str} "
            f"consistent with verified journalism. "
            f"Cross-check with multiple reliable outlets for full confidence."
        )
