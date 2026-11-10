"""Tools for asking: "how good is a model on a kind of text it has never trained on?"

Two pieces:

1. ``TopicDomains`` groups texts into topics (recipes, film reviews, biographies, ...)
   without using any labels. We call each group a "domain".
2. ``domain_stress_test`` scores a model on the team's shared 2,000-row validation set
   in four ways:

   * standard          - trained on all 8,000 training rows (the number the team reports)
   * layout removed    - same model, but every validation text has its line breaks and
                         extra spaces collapsed, so layout cannot help
   * unseen domain     - each validation text is scored by a model that was trained
                         WITHOUT that text's domain
   * unseen + layout removed - both at once

Nothing in ``src/`` is changed; this file only imports from it.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import ID_COL, LABEL_COL, SEED, TEXT_COL
from src.evaluate import auroc, predict_proba1
from src.features import normalize_style, normalize_ws


def _topic_text(text: str) -> str:
    """Lowercase, layout-free text: topics should depend on the words, not the formatting."""
    return normalize_style(text).lower()


class TopicDomains:
    """Unsupervised topic groups: content-word TF-IDF -> 100 SVD dimensions -> k-means.

    Labels are never used. Fit on the training texts only, then ``predict`` assigns any
    other text (validation, unlabeled, test) to its nearest topic.
    """

    def __init__(self, k: int = 8, n_components: int = 100, seed: int = SEED):
        self.k, self.n_components, self.seed = k, n_components, seed

    def _embed(self, texts, fit: bool = False):
        X = self.vec_.fit_transform(texts) if fit else self.vec_.transform(texts)
        Z = self.svd_.fit_transform(X) if fit else self.svd_.transform(X)
        return X, normalize(Z)  # unit length, so k-means compares direction (topic), not length

    def fit(self, texts):
        texts = list(texts)
        # stop_words="english" drops "the", "of", ... so groups form around content words.
        self.vec_ = TfidfVectorizer(preprocessor=_topic_text, stop_words="english", min_df=5,
                                    max_df=0.5, sublinear_tf=True)
        self.svd_ = TruncatedSVD(self.n_components, random_state=self.seed)
        _, Z = self._embed(texts, fit=True)
        self.km_ = KMeans(self.k, n_init=10, random_state=self.seed).fit(Z)
        self.names_ = self._make_names(texts)
        return self

    def predict(self, texts) -> np.ndarray:
        """Domain number (0 .. k-1) of each text."""
        return self.km_.predict(self._embed(list(texts))[1])

    def fit_quality(self, texts) -> np.ndarray:
        """Cosine similarity of each text to its nearest topic centre (1 = right on the
        centre). Lower values mean the text does not look like any training topic."""
        Z = self._embed(list(texts))[1]
        centres = normalize(self.km_.cluster_centers_)
        return (Z @ centres.T).max(axis=1)

    def top_terms(self, texts, n: int = 8) -> list[list[str]]:
        """Words that are much more common inside each domain than in the whole set."""
        texts = list(texts)
        X, Z = self._embed(texts)
        dom = self.km_.predict(Z)
        vocab = np.asarray(self.vec_.get_feature_names_out())
        overall = np.asarray(X.mean(axis=0)).ravel()
        out = []
        for d in range(self.k):
            inside = np.asarray(X[dom == d].mean(axis=0)).ravel()
            out.append(vocab[np.argsort(overall - inside)[:n]].tolist())
        return out

    def _make_names(self, texts) -> list[str]:
        # Name each domain by its three most typical words, e.g. "d4: film, movie, story".
        return [f"d{d}: " + ", ".join(t[:3]) for d, t in enumerate(self.top_terms(texts, 3))]

    def describe(self, texts, labels=None, n_terms: int = 8) -> pd.DataFrame:
        texts = list(texts)
        dom = self.predict(texts)
        rows = []
        for d, terms in enumerate(self.top_terms(texts, n_terms)):
            m = dom == d
            row = {"domain": d, "n": int(m.sum()), "share": m.mean()}
            if labels is not None:
                row["ai_share"] = float(np.asarray(labels)[m].mean())
            row["typical_words"] = ", ".join(terms)
            rows.append(row)
        return pd.DataFrame(rows)


def flatten_layout(texts) -> list[str]:
    """Collapse line breaks, tabs and repeated spaces into single spaces."""
    return [normalize_ws(t) for t in texts]


def _safe_auroc(y, p) -> float:
    # AUROC needs both classes; inside a small group one class could be missing.
    return auroc(y, p) if len(np.unique(y)) == 2 else float("nan")


def _mean_by_domain(y, p, dom) -> float:
    return float(np.nanmean([_safe_auroc(y[dom == d], p[dom == d]) for d in np.unique(dom)]))


def domain_stress_test(make_model, train_df: pd.DataFrame, val_df: pd.DataFrame,
                       dom_train, dom_val, size_control: bool = False, seed: int = SEED,
                       verbose: bool = True):
    """Score one model configuration the four ways described at the top of this file.

    ``make_model``: a function with no arguments that returns a fresh, unfitted model
    (anything with ``fit(texts, labels)`` and ``predict_proba`` or ``decision_function``).

    ``size_control=True`` adds a fairness check. Holding out a domain also makes the
    training set smaller, so for each domain we additionally train a model with the same
    number of rows removed AT RANDOM (same human/AI counts). If that model is still good,
    the drop comes from the missing domain and not from having less data.

    Returns ``(summary, per_domain, preds)``:
    ``summary`` - dict of headline numbers; ``per_domain`` - one row per domain;
    ``preds`` - one row per validation text with all four scores.
    """
    X_tr, y_tr = train_df[TEXT_COL].reset_index(drop=True), train_df[LABEL_COL].to_numpy()
    X_va, y_va = val_df[TEXT_COL].reset_index(drop=True), val_df[LABEL_COL].to_numpy()
    X_va_flat = pd.Series(flatten_layout(X_va))
    dom_train, dom_val = np.asarray(dom_train), np.asarray(dom_val)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    full = make_model().fit(X_tr, y_tr)
    fit_s = time.time() - t0
    p_std, p_std_flat = predict_proba1(full, X_va), predict_proba1(full, X_va_flat)

    n = len(y_va)
    p_unseen, p_unseen_flat, p_control = np.zeros(n), np.zeros(n), np.full(n, np.nan)
    rows = []
    for d in np.unique(dom_train):
        in_val = dom_val == d
        keep = dom_train != d                       # training rows from every OTHER domain
        model = make_model().fit(X_tr[keep], y_tr[keep])
        p_unseen[in_val] = predict_proba1(model, X_va[in_val])
        p_unseen_flat[in_val] = predict_proba1(model, X_va_flat[in_val])
        row = {"domain": int(d), "n_train_removed": int((~keep).sum()), "n_val": int(in_val.sum()),
               "val_ai_share": float(y_va[in_val].mean()),
               "seen": _safe_auroc(y_va[in_val], p_std[in_val])}
        if size_control:
            drop = np.zeros(len(y_tr), dtype=bool)
            for lab in (0, 1):                      # remove as many humans / AIs as the domain had
                k = int(((~keep) & (y_tr == lab)).sum())
                drop[rng.choice(np.where(y_tr == lab)[0], k, replace=False)] = True
            control = make_model().fit(X_tr[~drop], y_tr[~drop])
            p_control[in_val] = predict_proba1(control, X_va[in_val])
            row["seen_same_size"] = _safe_auroc(y_va[in_val], p_control[in_val])
        row["unseen"] = _safe_auroc(y_va[in_val], p_unseen[in_val])
        row["drop"] = row["seen"] - row["unseen"]
        rows.append(row)
        if verbose:
            print(f"  domain {d}: seen {row['seen']:.4f} -> unseen {row['unseen']:.4f}")

    summary = {
        "val_auroc": auroc(y_va, p_std),                                   # the team's usual number
        "val_layout_removed": auroc(y_va, p_std_flat),
        "seen_domain_mean": _mean_by_domain(y_va, p_std, dom_val),
        "unseen_domain_mean": _mean_by_domain(y_va, p_unseen, dom_val),
        "unseen_layout_removed_mean": _mean_by_domain(y_va, p_unseen_flat, dom_val),
        "unseen_pooled": auroc(y_va, p_unseen),
        "fit_seconds": round(fit_s, 1),
    }
    if size_control:
        summary["seen_same_size_mean"] = _mean_by_domain(y_va, p_control, dom_val)
    preds = pd.DataFrame({ID_COL: val_df[ID_COL].to_numpy(), LABEL_COL: y_va, "domain": dom_val,
                          "p_standard": p_std, "p_layout_removed": p_std_flat,
                          "p_unseen": p_unseen, "p_unseen_layout_removed": p_unseen_flat})
    return summary, pd.DataFrame(rows), preds
