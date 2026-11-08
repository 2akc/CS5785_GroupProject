"""Metrics, plots, result logging, and prediction files shared across the team."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve

from .config import (
    EXPECTED_TEST_ROWS,
    FIGURES_DIR,
    ID_COL,
    LABEL_COL,
    N_FOLDS,
    RESULTS_DIR,
    ROOT,
    SEED,
    SUBMISSIONS_DIR,
    ensure_dirs,
)
from .data import get_cv_folds

# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def auroc(y, p) -> float:
    return float(roc_auc_score(np.asarray(y), np.asarray(p)))


def predict_proba1(model, X) -> np.ndarray:
    """P(label = 1) for any fitted sklearn model (falls back to decision_function)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def cross_val_auroc(pipeline, X, y, k: int = N_FOLDS, seed: int = SEED,
                    return_oof: bool = False, verbose: bool = False) -> dict:
    """Stratified K-fold AUROC (exact duplicates share a fold). Returns mean/std/per-fold."""
    X = pd.Series(list(X)).reset_index(drop=True)
    y = np.asarray(y)
    folds = get_cv_folds(pd.DataFrame({"text": X, LABEL_COL: y}), k=k, seed=seed)
    oof = np.zeros(len(y), dtype=float)
    scores = []
    for i, (tr, va) in enumerate(folds):
        m = clone(pipeline).fit(X.iloc[tr], y[tr])
        oof[va] = predict_proba1(m, X.iloc[va])
        scores.append(auroc(y[va], oof[va]))
        if verbose:
            print(f"  fold {i}: AUROC={scores[-1]:.4f}")
    out = {"cv_auroc_mean": float(np.mean(scores)), "cv_auroc_std": float(np.std(scores)),
           "cv_auroc_folds": [round(s, 5) for s in scores]}
    if return_oof:
        out["oof"] = oof
    return out


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------


def plot_roc(y, probs, name: str, title: str | None = None):
    """``probs`` is one array or a ``{label: array}`` dict. Saves figures/roc_<name>.png."""
    ensure_dirs()
    if not isinstance(probs, dict):
        probs = {name: probs}
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{label} (AUROC={auroc(y, p):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title or f"ROC: {name}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / f"roc_{name}.png"
    fig.savefig(path, dpi=150)
    return fig, path


def savefig(fig, name: str):
    """Save a matplotlib figure to figures/<name>.png (and .pdf for the report)."""
    ensure_dirs()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    return FIGURES_DIR / f"{name}.png"


# --------------------------------------------------------------------------------------
# Result logging
# --------------------------------------------------------------------------------------


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "predict.py"],
                                        cwd=ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def log_result(experiment: str, config: dict, metrics: dict) -> pd.DataFrame:
    """Append one row to results/<experiment>.csv (timestamp, git commit, config, metrics)."""
    ensure_dirs()
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit()}
    for prefix, d in (("", config), ("", metrics)):
        for k, v in d.items():
            if isinstance(v, (list, tuple, dict)):
                v = json.dumps(v)
            elif isinstance(v, np.generic):
                v = v.item()
            row[prefix + k] = v
    path = RESULTS_DIR / f"{experiment}.csv"
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------------------
# Prediction files (for ensembling)
# --------------------------------------------------------------------------------------


def save_predictions(name: str, val_df: pd.DataFrame, val_probs, test_df: pd.DataFrame, test_probs):
    """Write results/preds_<name>.csv with columns id, split, label, prob.

    * ``val_probs``: from a model trained on the shared *train* split only.
    * ``test_probs``: from the same configuration refit on *all* training data.
    Any set of these files can be ensembled, using the val rows to choose weights.
    """
    ensure_dirs()
    val_probs, test_probs = np.asarray(val_probs, float), np.asarray(test_probs, float)
    if len(val_probs) != len(val_df) or len(test_probs) != len(test_df):
        raise ValueError("prediction lengths do not match their dataframes")
    if not (np.isfinite(val_probs).all() and np.isfinite(test_probs).all()):
        raise ValueError("predictions contain NaN/inf")
    val = pd.DataFrame({ID_COL: val_df[ID_COL].values, "split": "val",
                        LABEL_COL: val_df[LABEL_COL].values, "prob": val_probs})
    test = pd.DataFrame({ID_COL: test_df[ID_COL].values, "split": "test",
                         LABEL_COL: np.nan, "prob": test_probs})
    path = RESULTS_DIR / f"preds_{name}.csv"
    pd.concat([val, test], ignore_index=True).to_csv(path, index=False, float_format="%.6f")
    print(f"saved {path.relative_to(ROOT)}")
    return path


def load_predictions(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"preds_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; available: {list_predictions()}")
    return pd.read_csv(path)


def list_predictions() -> list[str]:
    return sorted(p.stem.removeprefix("preds_") for p in RESULTS_DIR.glob("preds_*.csv"))


def rank_average(prob_arrays, weights=None) -> np.ndarray:
    """Weighted mean of per-model ranks, rescaled to [0, 1]. Scale-free, so models whose
    outputs are not calibrated (SVM margins, transformer logits) can be mixed safely."""
    arrs = [rankdata(np.asarray(p)) / len(p) for p in prob_arrays]
    w = np.ones(len(arrs)) if weights is None else np.asarray(weights, float)
    r = np.average(np.vstack(arrs), axis=0, weights=w)
    return (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else np.full_like(r, 0.5)


# --------------------------------------------------------------------------------------
# Submissions
# --------------------------------------------------------------------------------------


def validate_submission(sub: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Fail loudly unless ``sub`` is a well-formed Kaggle submission for ``test_df``."""
    errors = []
    if list(sub.columns) != [ID_COL, LABEL_COL]:
        errors.append(f"columns must be ['{ID_COL}', '{LABEL_COL}'], got {list(sub.columns)}")
    if len(sub) != EXPECTED_TEST_ROWS:
        errors.append(f"expected {EXPECTED_TEST_ROWS} rows, got {len(sub)}")
    if len(sub) != len(test_df):
        errors.append(f"test.csv has {len(test_df)} rows but submission has {len(sub)}")
    if ID_COL in sub:
        if sub[ID_COL].duplicated().any():
            errors.append(f"{sub[ID_COL].duplicated().sum()} duplicated ids")
        missing = set(test_df[ID_COL]) - set(sub[ID_COL])
        extra = set(sub[ID_COL]) - set(test_df[ID_COL])
        if missing:
            errors.append(f"{len(missing)} test ids missing (e.g. {list(missing)[:3]})")
        if extra:
            errors.append(f"{len(extra)} ids not in test.csv (e.g. {list(extra)[:3]})")
    if LABEL_COL in sub:
        lab = pd.to_numeric(sub[LABEL_COL], errors="coerce")
        if lab.isna().any():
            errors.append(f"{lab.isna().sum()} NaN / non-numeric labels")
        if ((lab < 0) | (lab > 1)).any():
            errors.append("labels must be probabilities in [0, 1]")
        if lab.nunique() <= 2:
            errors.append("labels look like hard 0/1 predictions; AUROC needs probabilities")
    if errors:
        raise ValueError("Invalid submission:\n  - " + "\n  - ".join(errors))


def write_submission(test_df: pd.DataFrame, probs, tag: str = "") -> Path:
    """Validate and write submissions/submission_<timestamp>[_tag].csv."""
    ensure_dirs()
    sub = pd.DataFrame({ID_COL: test_df[ID_COL].values, LABEL_COL: np.asarray(probs, float)})
    validate_submission(sub, test_df)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SUBMISSIONS_DIR / f"submission_{stamp}{'_' + tag if tag else ''}.csv"
    sub.to_csv(path, index=False, float_format="%.6f")
    print(f"wrote {path.relative_to(ROOT)} ({len(sub)} rows) - validated OK")
    return path
