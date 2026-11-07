"""Named model pipelines (used by notebooks and predict.py) and semi-supervised helpers.

Add a final pipeline here (via a small PR to Allen) so ``python predict.py --model <name>``
can rebuild it from scratch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .evaluate import auroc, predict_proba1
from .features import FitWithExtraText, build_features


def _logreg(C=1.0):
    return LogisticRegression(C=C, max_iter=2000, solver="liblinear")


MODELS = {
    # word (1-2) + char_wb (3-5) TF-IDF -> logistic regression. The reference baseline.
    "baseline": lambda: Pipeline([("features", build_features("word+char")), ("clf", _logreg(1.0))]),
    "logreg_all": lambda: Pipeline([("features", build_features("all")), ("clf", _logreg(1.0))]),
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
