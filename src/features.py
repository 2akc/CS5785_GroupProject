"""Feature extractors. Each ``build_features(name)`` returns an unfitted sklearn transformer
that maps an iterable of raw strings to a (sparse or dense) feature matrix.

    from src.features import build_features
    pipe = make_pipeline(build_features("word+char"), LogisticRegression(max_iter=2000))
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

# Words, plus each punctuation mark as its own token, so the word model also sees
# punctuation habits (the default sklearn pattern throws punctuation away).
WORD_TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"


def word_tfidf(**kw) -> TfidfVectorizer:
    params = dict(ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=True,
                  token_pattern=WORD_TOKEN_PATTERN, dtype=np.float32)
    params.update(kw)
    return TfidfVectorizer(**params)


def char_tfidf(**kw) -> TfidfVectorizer:
    # lowercase=False keeps capitalization signal in the character n-grams.
    params = dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2,
                  lowercase=False, dtype=np.float32)
    params.update(kw)
    return TfidfVectorizer(**params)


# --------------------------------------------------------------------------------------
# Stylometric features
# --------------------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[A-Za-z0-9']+|[^\W\d_]+", re.UNICODE)
_STOP = frozenset(ENGLISH_STOP_WORDS)

STYLO_FEATURES = [
    "n_chars", "n_words", "avg_word_len", "n_sents", "avg_sent_len", "std_sent_len",
    "type_token_ratio", "comma_rate", "semicolon_rate", "colon_rate", "em_dash_rate",
    "double_hyphen_rate", "exclaim_rate", "question_rate", "paren_rate", "quote_rate",
    "curly_quote_frac", "ellipsis_rate", "uppercase_ratio", "digit_ratio", "non_ascii_ratio",
    "stopword_ratio", "newline_count", "double_space_count", "lowercase_sent_start_frac",
]


def stylometric_vector(text: str) -> list[float]:
    n_chars = len(text)
    words = _WORD.findall(text)
    n_words = max(len(words), 1)
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    sent_lens = np.array([len(_WORD.findall(s)) for s in sents] or [0], dtype=float)
    letters = sum(c.isalpha() for c in text) or 1
    lower_words = [w.lower() for w in words]
    straight_q = text.count('"') + text.count("'")
    curly_q = sum(text.count(c) for c in "“”‘’")
    per_word = lambda c: text.count(c) / n_words  # noqa: E731
    return [
        n_chars,
        len(words),
        np.mean([len(w) for w in words]) if words else 0.0,
        len(sents),
        sent_lens.mean(),
        sent_lens.std(),
        len(set(lower_words)) / n_words,
        per_word(","),
        per_word(";"),
        per_word(":"),
        per_word("—"),
        per_word("--"),
        per_word("!"),
        per_word("?"),
        (text.count("(") + text.count(")")) / n_words,
        (straight_q + curly_q) / n_words,
        curly_q / (straight_q + curly_q) if (straight_q + curly_q) else 0.0,
        (text.count("...") + text.count("…")) / n_words,
        sum(c.isupper() for c in text) / letters,
        sum(c.isdigit() for c in text) / max(n_chars, 1),
        sum(ord(c) > 127 for c in text) / max(n_chars, 1),
        sum(w in _STOP for w in lower_words) / n_words,
        text.count("\n"),
        text.count("  "),
        np.mean([s.lstrip()[:1].islower() for s in sents]) if sents else 0.0,
    ]


class StylometricFeatures(BaseEstimator, TransformerMixin):
    """Hand-crafted style statistics (dense, unscaled). Stateless."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray([stylometric_vector(t) for t in X], dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(STYLO_FEATURES, dtype=object)


def stylometric(scale: bool = True):
    steps = [("stylo", StylometricFeatures())]
    if scale:  # needed for linear models; tree models can use scale=False
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------

FEATURE_SETS = ("word_tfidf", "char_tfidf", "word+char", "stylometric", "all")


def build_features(name: str, **kw):
    """Return an unfitted transformer for one of ``FEATURE_SETS``.

    Keyword overrides: ``word={...}`` / ``char={...}`` are passed to the TF-IDF vectorizers,
    ``scale=False`` disables scaling of stylometric features.
    """
    word_kw, char_kw, scale = kw.pop("word", {}), kw.pop("char", {}), kw.pop("scale", True)
    if kw:
        raise TypeError(f"unexpected arguments {list(kw)}")
    if name == "word_tfidf":
        return word_tfidf(**word_kw)
    if name == "char_tfidf":
        return char_tfidf(**char_kw)
    if name == "stylometric":
        return stylometric(scale)
    if name == "word+char":
        return FeatureUnion([("word", word_tfidf(**word_kw)), ("char", char_tfidf(**char_kw))])
    if name == "all":
        return FeatureUnion([("word", word_tfidf(**word_kw)), ("char", char_tfidf(**char_kw)),
                             ("stylo", stylometric(scale))])
    raise ValueError(f"unknown feature set {name!r}; choose from {FEATURE_SETS}")


class FitWithExtraText(BaseEstimator, TransformerMixin):
    """Wrap a transformer so ``fit`` also sees extra unlabeled texts (e.g. unlabeled.csv).

    Used for "TF-IDF vocabulary / IDF fit on train + unlabeled". Labels are never used,
    so this is plain unsupervised pre-fitting of the vocabulary.
    """

    def __init__(self, transformer, extra_texts=()):
        self.transformer = transformer
        self.extra_texts = extra_texts

    def fit(self, X, y=None):
        from sklearn.base import clone
        self.transformer_ = clone(self.transformer).fit(list(X) + list(self.extra_texts))
        return self

    def transform(self, X):
        return self.transformer_.transform(X)
