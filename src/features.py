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


def normalize_ws(text: str) -> str:
    """Collapse all whitespace (newlines, tabs, double spaces) to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


# Features that only describe layout (line breaks, spacing), not the writing itself. In train,
# AI texts have about 2x the line breaks, but test is full of poems, recipes and hard-wrapped
# abstracts, so these features flag human test texts as AI. robust=True removes them.
FORMAT_FEATURES = ("newline_count", "double_space_count")
_KEEP_ROBUST = [i for i, f in enumerate(STYLO_FEATURES) if f not in FORMAT_FEATURES]


class StylometricFeatures(BaseEstimator, TransformerMixin):
    """Hand-crafted style statistics (dense, unscaled). Stateless.

    ``robust=True`` normalizes whitespace first (so sentence splitting ignores line breaks)
    and drops the layout-only features in ``FORMAT_FEATURES``.
    """

    def __init__(self, robust: bool = False):
        self.robust = robust

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.robust:
            M = np.asarray([stylometric_vector(normalize_ws(t)) for t in X], dtype=np.float32)
            return M[:, _KEEP_ROBUST]
        return np.asarray([stylometric_vector(t) for t in X], dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(STYLO_FEATURES, dtype=object)
        return names[_KEEP_ROBUST] if self.robust else names


# --------------------------------------------------------------------------------------
# Function-word skeleton: topic-free n-grams
# --------------------------------------------------------------------------------------

_SKEL_TOKEN = re.compile(r"\w+|[^\w\s]")


def skeleton_text(text: str) -> str:
    """Keep function words and punctuation, replace content words by their shape.

    "The Eiffel Tower is 330 m tall." -> "the X X is 0 x x ."
    The resulting n-grams describe sentence structure and style, not topic.
    """
    out = []
    for tok in _SKEL_TOKEN.findall(text):
        low = tok.lower()
        if low in _STOP or not tok[0].isalnum():
            out.append(low)
        elif tok.isdigit():
            out.append("0")
        else:
            out.append("X" if tok[0].isupper() else "x")
    return " ".join(out)


def skeleton_tfidf(**kw) -> TfidfVectorizer:
    params = dict(preprocessor=skeleton_text, token_pattern=r"\S+", lowercase=False,
                  ngram_range=(1, 4), min_df=3, sublinear_tf=True, dtype=np.float32)
    params.update(kw)
    return TfidfVectorizer(**params)


# --------------------------------------------------------------------------------------
# Formatting-robust text: same features, but layout and typography can't leak in
# --------------------------------------------------------------------------------------

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def normalize_style(text: str) -> str:
    """Straight quotes, one dash style, single spaces. Removes typographic/layout signals
    (curly quotes, em dashes, line breaks) that differ by source, not by author."""
    text = re.sub(r"\s*(—|–|--)\s*", " - ", text.translate(_QUOTES))
    return normalize_ws(text)


def has_blank_line_breaks(text: str) -> bool:
    """Paragraphs separated by blank lines (\\n\\n) at least as often as by single \\n.
    In train.csv every such text (2,734 of 10,000) is AI; humans use single line breaks."""
    blank = len(re.findall(r"\n\s*\n", text))
    single = len(re.findall(r"(?<!\n)\n(?!\n)", text))
    return blank >= max(1, single)


def _norm_lower(text):
    return normalize_style(text).lower()


def _norm_skeleton(text):
    return skeleton_text(normalize_style(text))


def robust_features(idf: bool = True, word=None, char=None, skel=None, weights=None):
    """word + char_wb + skeleton TF-IDF, all computed on ``normalize_style`` text.

    ``word`` / ``char`` / ``skel``: vectorizer overrides, or ``False`` to drop that block.
    ``weights``: per-block multipliers, e.g. ``{"char": 0.5}``.
    """
    blocks = [("word", word, lambda kw: word_tfidf(preprocessor=_norm_lower, use_idf=idf, **kw)),
              ("char", char, lambda kw: char_tfidf(preprocessor=normalize_style, use_idf=idf, **kw)),
              ("skel", skel, lambda kw: skeleton_tfidf(preprocessor=_norm_skeleton, use_idf=idf, **kw))]
    return FeatureUnion([(n, make(kw or {})) for n, kw, make in blocks if kw is not False],
                        transformer_weights=weights)


def stylometric(scale: bool = True, robust: bool = False):
    steps = [("stylo", StylometricFeatures(robust=robust))]
    if scale:  # needed for linear models; tree models can use scale=False
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------

FEATURE_SETS = ("word_tfidf", "char_tfidf", "word+char", "stylometric", "all",
                "stylometric_robust", "all_robust", "skeleton", "word+char+skeleton")


def build_features(name: str, **kw):
    """Return an unfitted transformer for one of ``FEATURE_SETS``.

    Keyword overrides: ``word={...}`` / ``char={...}`` are passed to the TF-IDF vectorizers,
    ``skel={...}`` to the skeleton vectorizer, ``scale=False`` disables scaling of
    stylometric features. ``*_robust`` sets use whitespace-normalized style features.
    """
    word_kw, char_kw, scale = kw.pop("word", {}), kw.pop("char", {}), kw.pop("scale", True)
    skel_kw = kw.pop("skel", {})
    if kw:
        raise TypeError(f"unexpected arguments {list(kw)}")
    if name == "word_tfidf":
        return word_tfidf(**word_kw)
    if name == "char_tfidf":
        return char_tfidf(**char_kw)
    if name == "stylometric":
        return stylometric(scale)
    if name == "stylometric_robust":
        return stylometric(scale, robust=True)
    if name == "skeleton":
        return skeleton_tfidf(**skel_kw)
    if name == "word+char+skeleton":
        return FeatureUnion([("word", word_tfidf(**word_kw)), ("char", char_tfidf(**char_kw)),
                             ("skel", skeleton_tfidf(**skel_kw))])
    if name == "all_robust":
        return FeatureUnion([("word", word_tfidf(**word_kw)), ("char", char_tfidf(**char_kw)),
                             ("stylo", stylometric(scale, robust=True))])
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
