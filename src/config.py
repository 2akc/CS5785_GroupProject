"""Project-wide paths and constants.

All paths are absolute and derived from this file's location, so imports work the same
whether code runs from the repo root, from ``notebooks/``, or from a Kaggle/Colab clone.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Override with AIHUMAN_DATA_DIR=/kaggle/input/<slug> (or any folder) when data lives elsewhere.
DATA_DIR = Path(os.environ.get("AIHUMAN_DATA_DIR", ROOT / "data")).resolve()
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SUBMISSIONS_DIR = ROOT / "submissions"

SEED = 42
VAL_FRAC = 0.2
N_FOLDS = 5

ID_COL = "id"
TEXT_COL = "text"
LABEL_COL = "label"  # 0 = human, 1 = AI-generated

FILES = {
    "train": "train.csv",
    "unlabeled": "unlabeled.csv",
    "test": "test.csv",
    "starter": "train_starter_2000.csv",
    "sample_submission": "sample_submission.csv",
}
SPLIT_FILE = DATA_DIR / "split_ids.json"

EXPECTED_TEST_ROWS = 3000


def set_seed(seed: int = SEED) -> None:
    """Seed python and numpy for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, FIGURES_DIR, SUBMISSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
