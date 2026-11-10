"""Classifiers on the cached language-model features, and blending with a text model."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ID_COL, SEED, TEXT_COL
from src.data import load_test, load_train, load_unlabeled
from src.evaluate import rank_average
from src.features import normalize_style

from lm_features import FEATURES, load_cached


@lru_cache(maxsize=4)
def _feature_table(model_name: str) -> pd.DataFrame:
    """Cached features indexed by the layout-free text they were computed from."""
    feats = load_cached(model_name)
    texts = pd.concat([load_train(verbose=False), load_unlabeled(verbose=False), load_test(verbose=False)])
    table = feats.loc[texts[ID_COL]].copy()
    table.index = texts[TEXT_COL].map(normalize_style).to_numpy()
    return table[~table.index.duplicated()]


class CachedLMFeatures(BaseEstimator, TransformerMixin):
    """Text -> its precomputed language-model features (a lookup, nothing is learned).

    Computing the features takes about 0.1 s per text, so they are computed once by
    ``lm_features.py`` and looked up here. The lookup key is the layout-free text, so a
    text gets the same features with or without its line breaks.
    """

    def __init__(self, columns=None, model_name: str = "gpt2"):
        self.columns, self.model_name = columns, model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        table = _feature_table(self.model_name)
        keys = [normalize_style(t) for t in X]
        missing = [k for k in keys if k not in table.index]
        if missing:
            raise KeyError(f"{len(missing)} texts have no cached features; run lm_features.py on them first")
        return table.loc[keys, list(self.columns or FEATURES)].to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.columns or FEATURES), dtype=object)


def lm_logreg(C: float = 1.0, columns=None, model_name: str = "gpt2") -> Pipeline:
    """Logistic regression on the standardised language-model features."""
    return Pipeline([("lm", CachedLMFeatures(columns, model_name)), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])


def lm_boost(max_depth: int = 3, learning_rate: float = 0.05, max_iter: int = 300,
             columns=None, model_name: str = "gpt2") -> Pipeline:
    """Gradient-boosted trees on the same features (can use combinations of features)."""
    return Pipeline([("lm", CachedLMFeatures(columns, model_name)),
                     ("clf", HistGradientBoostingClassifier(max_depth=max_depth, learning_rate=learning_rate,
                                                            max_iter=max_iter, random_state=SEED))])


def blend(text_scores, lm_scores, lm_weight: float) -> np.ndarray:
    """Weighted rank average of two models' scores. ``lm_weight`` = 0 uses only the text
    model, 1 uses only the language-model classifier."""
    return rank_average([text_scores, lm_scores], weights=[1 - lm_weight, lm_weight])


def blend_by_domain(text_scores, lm_scores, dom, lm_weight: float) -> np.ndarray:
    """Same blend, ranked inside each domain (used with per-domain AUROC)."""
    text_scores, lm_scores, dom = map(np.asarray, (text_scores, lm_scores, dom))
    out = np.zeros(len(dom))
    for d in np.unique(dom):
        m = dom == d
        out[m] = blend(text_scores[m], lm_scores[m], lm_weight)
    return out
