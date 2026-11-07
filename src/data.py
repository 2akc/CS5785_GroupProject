"""Data loading, light cleaning, and the ONE shared train/val split.

Design rules
------------
* Cleaning is deliberately light (see ``clean_text``). Capitalization, punctuation, spacing,
  and typos are likely what separates human from AI text, so we never lowercase, strip
  punctuation, remove stopwords, or spell-correct here. Feature extractors may choose to
  lowercase internally; the data layer does not.
* ``clean=False`` returns the raw text so "raw vs. cleaned" can be compared as an experiment.
* The train/val split is computed once from the raw ``train.csv`` ids (seed 42), saved to
  ``data/split_ids.json``, and reused by every notebook. Exact-duplicate texts are always
  placed on the same side of the split, so raw mode cannot leak duplicates from train to val.
  The split does not depend on ``clean`` or ``quick``. QUICK mode just restricts both sides
  to the ids in ``train_starter_2000.csv``.
* ``load_test`` never drops rows: every test id needs a prediction.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import ftfy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .config import (
    DATA_DIR,
    FILES,
    ID_COL,
    LABEL_COL,
    N_FOLDS,
    SEED,
    SPLIT_FILE,
    TEXT_COL,
    VAL_FRAC,
)

# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------

# ftfy defaults also "uncurl" quotes, undo ligatures, and normalise full-width characters.
# Those choices could be signal (e.g. curly vs straight quotes), so we switch them off and
# only repair genuine encoding damage (mojibake such as "donâ€™t" -> "don’t").
_FTFY_KW = dict(
    unescape_html=False,
    fix_latin_ligatures=False,
    fix_character_width=False,
    uncurl_quotes=False,
    fix_line_breaks=False,
    normalization="NFC",
)


def clean_text(text) -> str:
    """Light cleaning: repair mojibake, Unicode NFC, strip outer whitespace. Nothing else."""
    if not isinstance(text, str):
        return ""
    return ftfy.fix_text(text, **_FTFY_KW).strip()


def _as_text(series: pd.Series, clean: bool) -> pd.Series:
    if clean:
        return series.map(clean_text)
    return series.where(series.map(lambda x: isinstance(x, str)), "")


def group_keys(texts) -> np.ndarray:
    """Integer group id per text. Texts that are identical after cleaning share an id."""
    keys = pd.Series(list(texts)).map(clean_text)
    return pd.factorize(keys)[0]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


@dataclass
class CleaningReport:
    name: str
    clean: bool
    n_raw: int
    n_empty: int = 0
    n_duplicate_rows: int = 0          # rows whose text already appeared earlier
    n_conflicting_label_rows: int = 0  # rows in duplicate groups with mixed labels
    n_dropped: int = 0
    n_final: int = 0
    notes: list = field(default_factory=list)

    def __str__(self) -> str:
        s = (f"[{self.name}] raw={self.n_raw} empty={self.n_empty} "
             f"dup_rows={self.n_duplicate_rows} conflicting_label_rows={self.n_conflicting_label_rows} "
             f"dropped={self.n_dropped} final={self.n_final} (clean={self.clean})")
        return s + "".join(f"\n    note: {n}" for n in self.notes)


def _read_csv(key: str, required: tuple[str, ...]) -> pd.DataFrame:
    path = DATA_DIR / FILES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the competition files from Kaggle into "
            f"{DATA_DIR}/ (see data/README.md), or set AIHUMAN_DATA_DIR."
        )
    df = pd.read_csv(path, keep_default_na=False, dtype={TEXT_COL: str})
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}; found {list(df.columns)}")
    if df[ID_COL].duplicated().any():
        raise ValueError(f"{path.name} has {df[ID_COL].duplicated().sum()} duplicated ids")
    return df


def _prepare(df: pd.DataFrame, name: str, clean: bool, drop: bool, verbose: bool) -> pd.DataFrame:
    df = df.copy()
    rep = CleaningReport(name=name, clean=clean, n_raw=len(df))
    df[TEXT_COL] = _as_text(df[TEXT_COL], clean)

    empty = df[TEXT_COL].str.strip() == ""
    rep.n_empty = int(empty.sum())
    dup = df[TEXT_COL].duplicated(keep="first") & ~empty
    rep.n_duplicate_rows = int(dup.sum())

    conflict = pd.Series(False, index=df.index)
    if LABEL_COL in df.columns:
        n_labels = df.groupby(TEXT_COL)[LABEL_COL].transform("nunique")
        conflict = (n_labels > 1) & ~empty
        rep.n_conflicting_label_rows = int(conflict.sum())

    if drop:
        # Exact duplicates: keep the first copy. If the copies disagree on the label,
        # drop every copy because we cannot tell which label is right.
        to_drop = empty | dup | conflict
        rep.n_dropped = int(to_drop.sum())
        df = df[~to_drop]
    elif rep.n_empty or rep.n_duplicate_rows:
        rep.notes.append("rows kept (drop=False)")

    df = df.reset_index(drop=True)
    rep.n_final = len(df)
    df.attrs["cleaning_report"] = rep
    if verbose:
        print(rep)
    return df


def load_train(clean: bool = True, quick: bool = False, drop: bool = True,
               verbose: bool = True) -> pd.DataFrame:
    """Labelled training data. ``quick=True`` loads train_starter_2000.csv instead."""
    key = "starter" if quick else "train"
    df = _read_csv(key, (ID_COL, TEXT_COL, LABEL_COL))
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return _prepare(df, key, clean, drop, verbose)


def load_unlabeled(clean: bool = True, drop: bool = True, verbose: bool = True) -> pd.DataFrame:
    df = _read_csv("unlabeled", (ID_COL, TEXT_COL))
    return _prepare(df, "unlabeled", clean, drop, verbose)


def load_test(clean: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Test data. Rows are never dropped; empty/duplicate counts are only reported."""
    df = _read_csv("test", (ID_COL, TEXT_COL))
    return _prepare(df, "test", clean, drop=False, verbose=verbose)


def overlap_report(clean: bool = True, verbose: bool = True) -> dict:
    """Exact-text overlap between train / unlabeled / test (after cleaning if ``clean``)."""
    sets = {
        "train": load_train(clean=clean, drop=False, verbose=False),
        "unlabeled": load_unlabeled(clean=clean, drop=False, verbose=False),
        "test": load_test(clean=clean, verbose=False),
    }
    texts = {k: set(v[TEXT_COL]) - {""} for k, v in sets.items()}
    out = {f"dup_rows_within_{k}": int(v[TEXT_COL].duplicated().sum()) for k, v in sets.items()}
    for a, b in [("train", "test"), ("train", "unlabeled"), ("unlabeled", "test")]:
        out[f"overlap_{a}_{b}"] = len(texts[a] & texts[b])
    if verbose:
        for k, v in out.items():
            print(f"{k:32s} {v}")
    return out


# --------------------------------------------------------------------------------------
# The shared split
# --------------------------------------------------------------------------------------


def _fingerprint(ids) -> str:
    joined = "\n".join(sorted(map(str, ids)))
    return hashlib.sha1(joined.encode()).hexdigest()[:12]


def _create_split(source_key: str) -> dict:
    raw = _read_csv(source_key, (ID_COL, TEXT_COL, LABEL_COL))
    groups = group_keys(raw[TEXT_COL])
    g = pd.DataFrame({"group": groups, "label": raw[LABEL_COL].astype(int).values})
    group_label = g.groupby("group")["label"].first()
    tr_g, va_g = train_test_split(
        group_label.index.values, test_size=VAL_FRAC, random_state=SEED,
        stratify=group_label.values,
    )
    va_set = set(va_g)
    is_val = np.array([gr in va_set for gr in groups])
    return {
        "source": FILES[source_key],
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "fingerprint": _fingerprint(raw[ID_COL]),
        "train_ids": raw.loc[~is_val, ID_COL].tolist(),
        "val_ids": raw.loc[is_val, ID_COL].tolist(),
    }


def load_split_ids(verbose: bool = True) -> dict:
    """Load data/split_ids.json, creating it on first use."""
    # Normally split on the full train.csv. If only the starter file is present, split that.
    source_key = "train" if (DATA_DIR / FILES["train"]).exists() else "starter"
    path = SPLIT_FILE if source_key == "train" else SPLIT_FILE.with_name("split_ids_starter.json")
    current = _read_csv(source_key, (ID_COL,))[ID_COL]
    if path.exists():
        split = json.loads(path.read_text())
        if split.get("fingerprint") != _fingerprint(current) or split.get("seed") != SEED:
            raise RuntimeError(
                f"{path} was made from a different {FILES[source_key]} or seed. "
                "Delete it to regenerate (and tell the team)."
            )
    else:
        split = _create_split(source_key)
        path.write_text(json.dumps(split))
        if verbose:
            print(f"Created {path} ({len(split['train_ids'])} train / {len(split['val_ids'])} val ids)")
    return split


def get_split(quick: bool = False, clean: bool = True, drop: bool = True,
              verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_df, val_df)`` for the shared, fixed split.

    Use this in every notebook so validation numbers are comparable across the team.
    """
    split = load_split_ids(verbose=verbose)
    df = load_train(clean=clean, quick=quick, drop=drop, verbose=verbose)
    tr_ids, va_ids = set(split["train_ids"]), set(split["val_ids"])
    unknown = ~df[ID_COL].isin(tr_ids | va_ids)
    if unknown.any():
        raise RuntimeError(f"{unknown.sum()} ids are not in the split file; is the starter file a subset of train?")
    tr = df[df[ID_COL].isin(tr_ids)].reset_index(drop=True)
    va = df[df[ID_COL].isin(va_ids)].reset_index(drop=True)
    if verbose:
        print(f"split: train={len(tr)} (AI={tr[LABEL_COL].mean():.3f})  "
              f"val={len(va)} (AI={va[LABEL_COL].mean():.3f})  quick={quick} clean={clean}")
    return tr, va


def get_cv_folds(df: pd.DataFrame, k: int = N_FOLDS, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold over ``df`` (positional indices). Exact duplicates share a fold."""
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    groups = group_keys(df[TEXT_COL])
    return list(skf.split(df[TEXT_COL], df[LABEL_COL], groups))


def xy(df: pd.DataFrame):
    """Return ``(texts, labels)``; labels is None for unlabeled/test frames."""
    y = df[LABEL_COL].to_numpy() if LABEL_COL in df.columns else None
    return df[TEXT_COL], y


__all__ = [
    "clean_text", "group_keys", "load_train", "load_unlabeled", "load_test",
    "overlap_report", "load_split_ids", "get_split", "get_cv_folds", "xy", "CleaningReport",
]
