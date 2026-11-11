"""A third stress test: how good is a model on a KIND of AI text it has never trained on?

``domain_eval.py`` hides a topic. Here we hide a kind of AI writing instead (human texts
stay in training). We have no generator labels, so we make our own groups: k-means on the
GPT-2 features of the AI training texts. Texts in one group were probably written by
similar generators or with similar settings (for example "very predictable", or "avoids
repeating words").

Nothing in ``src/`` is changed; this file only imports from it.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import ID_COL, LABEL_COL, SEED, TEXT_COL
from src.evaluate import auroc, predict_proba1, rank_average

from domain_eval import flatten_layout


class AIKinds:
    """Unsupervised groups of AI texts. Fit on the AI rows of the training split only."""

    def __init__(self, k: int = 4, seed: int = SEED):
        self.k, self.seed = k, seed

    def fit(self, features: pd.DataFrame):
        self.scaler_ = StandardScaler().fit(features)
        self.km_ = KMeans(self.k, n_init=10, random_state=self.seed).fit(self.scaler_.transform(features))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.km_.predict(self.scaler_.transform(features))

    def assign(self, features: pd.DataFrame, labels) -> np.ndarray:
        """Kind number for AI rows, -1 for human rows."""
        labels = np.asarray(labels)
        out = np.full(len(labels), -1)
        out[labels == 1] = self.predict(features[labels == 1])
        return out


def _kind_auroc(y, p, kind, c) -> float:
    """AUROC of 'AI texts of kind c' against ALL human texts."""
    m = (kind == c) | (y == 0)
    return auroc(y[m], np.asarray(p)[m])


def kind_stress_test(make_model, train_df: pd.DataFrame, val_df: pd.DataFrame, kind_train, kind_val,
                     verbose: bool = True):
    """Score one model configuration with each AI kind held out of training in turn.

    Validation texts have their layout removed (the strict setting from notebook ``01``).
    Returns ``(summary, per_kind, preds)``; ``preds`` has one column of validation scores
    per held-out kind (``p_without_kind0`` ...) plus ``p_seen`` from the model trained on
    everything.
    """
    X_tr, y_tr = train_df[TEXT_COL].reset_index(drop=True), train_df[LABEL_COL].to_numpy()
    y_va = val_df[LABEL_COL].to_numpy()
    X_va = pd.Series(flatten_layout(val_df[TEXT_COL]))
    kind_train, kind_val = np.asarray(kind_train), np.asarray(kind_val)
    kinds = sorted(k for k in np.unique(kind_train) if k >= 0)

    t0 = time.time()
    p_seen = predict_proba1(make_model().fit(X_tr, y_tr), X_va)
    preds = pd.DataFrame({ID_COL: val_df[ID_COL].to_numpy(), LABEL_COL: y_va, "kind": kind_val, "p_seen": p_seen})
    rows = []
    for c in kinds:
        keep = kind_train != c                      # every human text + AI texts of the other kinds
        p = predict_proba1(make_model().fit(X_tr[keep], y_tr[keep]), X_va)
        preds[f"p_without_kind{c}"] = p
        rows.append({"kind": int(c), "n_val_ai": int((kind_val == c).sum()),
                     "seen": _kind_auroc(y_va, p_seen, kind_val, c), "unseen": _kind_auroc(y_va, p, kind_val, c)})
        rows[-1]["drop"] = rows[-1]["seen"] - rows[-1]["unseen"]
        if verbose:
            print(f"  kind {c}: seen {rows[-1]['seen']:.4f} -> unseen {rows[-1]['unseen']:.4f}")
    per_kind = pd.DataFrame(rows)
    summary = {"seen_kind_mean": float(per_kind["seen"].mean()), "unseen_kind_mean": float(per_kind["unseen"].mean()),
               "hardest_unseen_kind": float(per_kind["unseen"].min()), "seconds": round(time.time() - t0, 1)}
    return summary, per_kind, preds


def blend_unseen_kind(preds_text: pd.DataFrame, preds_lm: pd.DataFrame, lm_weight: float) -> pd.Series:
    """Unseen-kind AUROC of the rank blend, per kind, from two ``kind_stress_test`` outputs."""
    y, kind = preds_text[LABEL_COL].to_numpy(), preds_text["kind"].to_numpy()
    out = {}
    for c in sorted(k for k in np.unique(kind) if k >= 0):
        col = f"p_without_kind{c}"
        out[c] = _kind_auroc(y, rank_average([preds_text[col], preds_lm[col]], [1 - lm_weight, lm_weight]), kind, c)
    return pd.Series(out)


def kind_weight_curve(make_text, make_lm, train_df: pd.DataFrame, kind_train, weights, seed: int = SEED,
                      verbose: bool = True) -> pd.DataFrame:
    """Blend quality for each weight under 'unseen AI kind', using TRAINING rows only.

    For each kind c the human texts are split in two halves. We train on one half of the
    humans plus the AI texts of the other kinds, and test on the other half of the humans
    plus the AI texts of kind c (then swap the halves). No validation row is touched, so
    the weight chosen from this curve does not peek at validation labels.
    """
    X, y = train_df[TEXT_COL].reset_index(drop=True), train_df[LABEL_COL].to_numpy()
    X_flat = pd.Series(flatten_layout(X))
    kind_train = np.asarray(kind_train)
    human = y == 0
    half = np.full(len(y), -1)
    half[human] = np.random.default_rng(seed).integers(0, 2, human.sum())
    rows = []
    for c in sorted(k for k in np.unique(kind_train) if k >= 0):
        for h in (0, 1):
            train = (human & (half != h)) | (~human & (kind_train != c))
            test = (human & (half == h)) | (kind_train == c)
            s = predict_proba1(make_text().fit(X[train], y[train]), X_flat[test])
            p = predict_proba1(make_lm().fit(X[train], y[train]), X_flat[test])
            rows.append({"kind": int(c), "half": h,
                         **{float(w): auroc(y[test], rank_average([s, p], [1 - w, w])) for w in weights}})
            if verbose:
                print(f"  kind {c}, half {h} done")
    return pd.DataFrame(rows)
