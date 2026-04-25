"""
_vectorizer_wrapper.py
-----------------------
Helper class that bundles word + char TF-IDF vectorizers
so we can joblib.dump/load a single object.
"""

from scipy.sparse import hstack


class CombinedVectorizer:
    """Wraps two TfidfVectorizers and presents a sklearn-compatible interface."""

    def __init__(self, word_vec, char_vec):
        self.word_vec = word_vec
        self.char_vec = char_vec

    def transform(self, texts):
        Xw = self.word_vec.transform(texts)
        Xc = self.char_vec.transform(texts)
        return hstack([Xw, Xc])

    def get_feature_names_out(self):
        import numpy as np
        return np.concatenate([
            self.word_vec.get_feature_names_out(),
            self.char_vec.get_feature_names_out(),
        ])
