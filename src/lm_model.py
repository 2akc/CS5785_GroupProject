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

from .config import ID_COL, SEED, TEXT_COL
from .data import load_test, load_train, load_unlabeled
from .evaluate import predict_proba1, rank_average
from .features import normalize_style
from .lm_features import FEATURES, load_cached
from .lm_onepass import ONEPASS_FEATURES
from .lm_onepass import cache_path as _onepass_path
from .lm_onepass import load_cached as _load_onepass
from .repetition_features import RP_FEATURES, WORD_FEATURES
from .repetition_features import cache_path as _rep_path
from .repetition_features import load_cached as _load_rep

# The same predictability + repetition features, computed by SmolLM2-360M instead of GPT-2.
SMOL_FEATURES = ["smol_" + f for f in ONEPASS_FEATURES]
# Everything the final classifier uses: GPT-2 predictability + GPT-2 repetition + word-count
# repetition + the SmolLM2 versions (87 features).
ALL_LM_FEATURES = FEATURES + RP_FEATURES + WORD_FEATURES + SMOL_FEATURES


@lru_cache(maxsize=4)
def _feature_table(model_name: str) -> pd.DataFrame:
    """Cached features indexed by the layout-free text they were computed from.

    Holds the 24 predictability features and, when that cache exists, the repetition
    features from ``src/repetition_features.py`` as extra columns."""
    feats = load_cached(model_name)
    if _rep_path(model_name).exists():
        feats = feats.join(_load_rep(model_name).drop(columns="file"))
    if _onepass_path().exists():                       # second language model (SmolLM2), columns prefixed "smol_"
        feats = feats.join(_load_onepass(prefix="smol_"))
    texts = pd.concat([load_train(verbose=False), load_unlabeled(verbose=False), load_test(verbose=False)])
    table = feats.loc[texts[ID_COL]].copy()
    table.index = texts[TEXT_COL].map(normalize_style).to_numpy()
    return table[~table.index.duplicated()]


class CachedLMFeatures(BaseEstimator, TransformerMixin):
    """Text -> its precomputed language-model features (a lookup, nothing is learned).

    Computing the features takes about 0.1 s per text, so they are computed once by
    ``src/lm_features.py`` and looked up here. The lookup key is the layout-free text, so a
    text gets the same features with or without its line breaks.
    """

    # ``columns`` defaults to the 24 predictability features. Pass e.g.
    # ``FEATURES + RP_FEATURES`` to add the repetition-penalty features.
    def __init__(self, columns=None, model_name: str = "gpt2"):
        self.columns, self.model_name = columns, model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        table = _feature_table(self.model_name)
        keys = [normalize_style(t) for t in X]
        missing = [k for k in keys if k not in table.index]
        if missing:
            raise KeyError(f"{len(missing)} texts have no cached features; run `python -m src.lm_features` on them first")
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


class BaggedBoost(BaseEstimator):
    """Average of ``n_models`` boosted-tree models, each trained on a bootstrap sample
    (rows drawn with replacement) with its own seed. Averaging many slightly different
    models gives steadier scores than one model."""

    def __init__(self, n_models: int = 10, max_depth: int = 3, learning_rate: float = 0.05, max_iter: int = 300,
                 min_samples_leaf: int = 40, l2_regularization: float = 1.0, max_features: float = 0.5,
                 seed: int = SEED):
        self.n_models, self.max_depth, self.learning_rate, self.max_iter = n_models, max_depth, learning_rate, max_iter
        self.min_samples_leaf, self.l2_regularization, self.max_features, self.seed = (
            min_samples_leaf, l2_regularization, max_features, seed)

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.seed)
        self.models_ = []
        for k in range(self.n_models):
            rows = rng.integers(0, len(y), len(y))
            self.models_.append(HistGradientBoostingClassifier(
                max_depth=self.max_depth, learning_rate=self.learning_rate, max_iter=self.max_iter,
                min_samples_leaf=self.min_samples_leaf, l2_regularization=self.l2_regularization,
                max_features=self.max_features, random_state=self.seed + k).fit(X[rows], y[rows]))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)
        return np.column_stack([1 - p, p])


class LinearPlusTrees(BaseEstimator):
    """Mean of two probabilities: a logistic regression and a bag of careful boosted trees.

    The trees are sharper on kinds of AI text seen in training; the linear model carries
    over better to kinds it has not seen. Their plain average keeps most of both."""

    def __init__(self, C: float = 1.0, n_models: int = 10, seed: int = SEED):
        self.C, self.n_models, self.seed = C, n_models, seed

    def fit(self, X, y):
        self.linear_ = Pipeline([("scale", StandardScaler()),
                                 ("clf", LogisticRegression(C=self.C, max_iter=5000))]).fit(X, y)
        self.trees_ = BaggedBoost(n_models=self.n_models, seed=self.seed).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (self.linear_.predict_proba(X)[:, 1] + self.trees_.predict_proba(X)[:, 1]) / 2
        return np.column_stack([1 - p, p])


def lm_linear_plus_trees(columns=None, C: float = 1.0, n_models: int = 10, model_name: str = "gpt2") -> Pipeline:
    """Text -> cached features -> ``LinearPlusTrees``."""
    return Pipeline([("lm", CachedLMFeatures(columns, model_name)), ("clf", LinearPlusTrees(C=C, n_models=n_models))])


class RankBlend(BaseEstimator):
    """Weighted rank average of a text model and a language-model classifier.

    ``lm_weight`` is the weight of the language-model classifier (0.8 was chosen on training
    rows in notebooks/joling/05_newer_language_model.ipynb). Ranks are taken within the
    texts passed to ``predict_proba``, so score a whole set (e.g. all of test.csv) in one call.
    """

    def __init__(self, text_model, lm_model, lm_weight: float = 0.8):
        self.text_model, self.lm_model, self.lm_weight = text_model, lm_model, lm_weight

    def fit(self, X, y):
        from sklearn.base import clone
        self.text_model_ = clone(self.text_model).fit(X, y)
        self.lm_model_ = clone(self.lm_model).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = list(X)
        p = blend(predict_proba1(self.text_model_, X), predict_proba1(self.lm_model_, X), self.lm_weight)
        return np.column_stack([1 - p, p])
