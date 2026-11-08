"""Named model pipelines (used by notebooks and predict.py) and semi-supervised helpers.

Add a final pipeline here so ``python predict.py --model <name>``
can rebuild it from scratch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .evaluate import auroc, predict_proba1
from .features import FitWithExtraText, build_features, normalize_ws


def _logreg(C=1.0):
    return LogisticRegression(C=C, max_iter=2000, solver="liblinear")


class NBScaler(BaseEstimator, TransformerMixin):
    """NB-SVM feature scaling (Wang & Manning, 2012): multiply each feature by its Naive Bayes
    log-count ratio r = log((p / |p|) / (q / |q|)), so a linear model starts from NB's evidence."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        y = np.asarray(y)
        p = self.alpha + X[y == 1].sum(0)
        q = self.alpha + X[y == 0].sum(0)
        self.r_ = np.asarray(np.log((p / p.sum()) / (q / q.sum()))).ravel()
        return self

    def transform(self, X):
        return sp.csr_matrix(X).multiply(self.r_).tocsr()


def _char_ws(**kw):
    """char (not char_wb) n-grams on whitespace-normalized text: sees punctuation and spacing
    between words, but not line breaks or layout."""
    return dict(analyzer="char", preprocessor=normalize_ws, **kw)


MODELS = {
    # word (1-2) + char_wb (3-5) TF-IDF -> logistic regression. The reference baseline.
    "baseline": lambda: Pipeline([("features", build_features("word+char")), ("clf", _logreg(1.0))]),
    # + all style features. Val 0.980 but Kaggle 0.804: line-break features don't transfer.
    "logreg_all": lambda: Pipeline([("features", build_features("all")), ("clf", _logreg(1.0))]),
    # + style features computed on whitespace-normalized text, layout features removed.
    "logreg_all_robust": lambda: Pipeline([("features", build_features("all_robust")), ("clf", _logreg(1.0))]),
    # word + char n-grams across word boundaries (whitespace-normalized).
    "logreg_char": lambda: Pipeline([("features", build_features("word+char", char=_char_ws())),
                                     ("clf", _logreg(1.0))]),
    # Topic-free only: function-word skeleton n-grams (content words replaced by their shape).
    "skeleton_only": lambda: Pipeline([("features", build_features("skeleton")), ("clf", _logreg(4.0))]),
    # baseline + topic-free function-word skeleton n-grams.
    "logreg_skeleton": lambda: Pipeline([("features", build_features("word+char+skeleton")),
                                         ("clf", _logreg(4.0))]),
    # word + char (whitespace-normalized) + skeleton.
    "logreg_char_skeleton": lambda: Pipeline([("features", build_features("word+char+skeleton", char=_char_ws())),
                                              ("clf", _logreg(1.0))]),
    # NB-SVM: NB log-count ratios as feature weights, then logistic regression.
    "nbsvm": lambda: Pipeline([("features", build_features("word+char", word={"use_idf": False},
                                                           char={"use_idf": False})),
                               ("nb", NBScaler()), ("clf", _logreg(4.0))]),
    # Multinomial Naive Bayes on word + char counts (TF-IDF without idf, non-negative).
    "mnb": lambda: Pipeline([("features", build_features("word+char")), ("clf", MultinomialNB(alpha=0.05))]),
}


def make_model(name: str, extra_texts=None):
    """Fresh, unfitted pipeline by name. ``extra_texts`` pre-fits the feature step's
    vocabulary on additional unlabeled text (train + unlabeled TF-IDF)."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    model = MODELS[name]()
    if extra_texts is not None:
        steps = dict(model.steps)
        model.set_params(features=FitWithExtraText(steps["features"], tuple(extra_texts)))
    return model


# --------------------------------------------------------------------------------------
# Pseudo-labelling / self-training
# --------------------------------------------------------------------------------------


def select_confident(probs, threshold: float = 0.9, balance: bool = False):
    """Indices + hard labels where p >= threshold (AI) or p <= 1 - threshold (human).

    ``balance=True`` keeps the same number of each class (the most confident ones), which
    stops the pseudo-labelled set drifting towards whichever class the model favours.
    """
    probs = np.asarray(probs)
    pos = np.where(probs >= threshold)[0]
    neg = np.where(probs <= 1 - threshold)[0]
    if balance:
        n = min(len(pos), len(neg))
        pos = pos[np.argsort(-probs[pos])][:n]
        neg = neg[np.argsort(probs[neg])][:n]
    idx = np.concatenate([pos, neg])
    return idx, (probs[idx] >= 0.5).astype(int)


def self_train(estimator, X_lab, y_lab, X_unl, threshold: float = 0.9, rounds: int = 1,
               balance: bool = False, X_val=None, y_val=None, verbose: bool = True):
    """Classic self-training. Each round: fit on labelled + pseudo-labelled data, score the
    remaining unlabelled pool, move confident examples (with fixed hard labels) into the
    training set. ``rounds=1`` is one-shot pseudo-labelling.

    Returns ``(final_model, history_df, pseudo_df)``; ``pseudo_df`` has the positional index
    into ``X_unl``, the pseudo label, the probability, and the round it was added.
    """
    X_lab, X_unl = list(X_lab), list(X_unl)
    y_lab = np.asarray(y_lab)
    pool = np.arange(len(X_unl))
    added_idx, added_y, added_p, added_round = [], [], [], []
    history = []

    def fit_current():
        X = X_lab + [X_unl[i] for i in added_idx]
        y = np.concatenate([y_lab, np.asarray(added_y, dtype=int)])
        return clone(estimator).fit(X, y)

    model = fit_current()
    for r in range(rounds + 1):
        rec = {"round": r, "n_pseudo_total": len(added_idx)}
        if X_val is not None:
            rec["val_auroc"] = auroc(y_val, predict_proba1(model, X_val))
        if r == rounds or len(pool) == 0:
            rec["n_added"] = 0
            history.append(rec)
            break
        p = predict_proba1(model, [X_unl[i] for i in pool])
        sel, lab = select_confident(p, threshold, balance)
        rec.update(n_added=len(sel), frac_ai_added=float(lab.mean()) if len(sel) else np.nan)
        history.append(rec)
        if verbose:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
        if len(sel) == 0:
            break
        added_idx += pool[sel].tolist()
        added_y += lab.tolist()
        added_p += p[sel].tolist()
        added_round += [r + 1] * len(sel)
        pool = np.delete(pool, sel)
        model = fit_current()

    pseudo = pd.DataFrame({"unl_index": added_idx, "pseudo_label": added_y,
                           "prob": added_p, "round_added": added_round})
    return model, pd.DataFrame(history), pseudo
