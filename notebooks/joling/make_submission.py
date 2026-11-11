"""Rebuild the final Kaggle submission from scratch (about 2 minutes on a laptop).

    python notebooks/joling/make_submission.py                 # the selected submission (public 0.963)
    python notebooks/joling/make_submission.py --variant smol  # runner-up: SmolLM2 features only (public 0.958)
    python notebooks/joling/make_submission.py --variant tf    # + Aidan's DistilRoBERTa at weight 0.2 (public 0.961)

Pipeline (see notebooks 02-05 for how each choice was made):
  1. Classifier: logistic regression + 10 bagged boosted-tree models on cached language-model
     features (GPT-2 and SmolLM2-360M predictability / repetition features + word-count repetition).
  2. Text model: Allen's ``robust_svm_tuned`` with the blank-line boost.
  3. Weighted rank average of the two. The weight was chosen on training rows in notebook 05.
Both models are trained on all 10,000 labelled rows; seed 42 everywhere.

Needs the feature caches in ``results/`` (committed). They are rebuilt by ``src/lm_features.py``,
``src/repetition_features.py`` and ``src/lm_onepass.py``.

The selected submission can also be rebuilt with the team's entry point:

    python predict.py --model lm_svm_blend --split-order

This script is kept for the two other variants below.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import RESULTS_DIR, SEED, set_seed  # noqa: E402
from src.data import get_split, load_test  # noqa: E402
from src.evaluate import predict_proba1, rank_average, write_submission  # noqa: E402
from src.features import has_blank_line_breaks  # noqa: E402
from src.models import make_model  # noqa: E402

from src.lm_features import FEATURES  # noqa: E402
from src.lm_model import SMOL_FEATURES, blend, lm_linear_plus_trees  # noqa: E402
from src.repetition_features import RP_FEATURES, WORD_FEATURES  # noqa: E402

FEATURE_SETS = {
    "final": FEATURES + RP_FEATURES + WORD_FEATURES + SMOL_FEATURES,   # GPT-2 + SmolLM2 (87 features)
    "smol": SMOL_FEATURES + WORD_FEATURES,                             # SmolLM2 only (52 features)
}
TRANSFORMER_WEIGHT = 0.2


def blank_push(p, texts, push: float = 0.1):
    """Allen's BlankLineBoost: nudge up texts that use blank-line paragraph breaks."""
    b = np.array([has_blank_line_breaks(t) for t in texts], dtype=float)
    return (np.asarray(p) + push * b) / (1 + push)


def main(variant: str = "final") -> Path:
    set_seed(SEED)
    weight = float(pd.read_csv(RESULTS_DIR / "joling_blend_weights_05.csv")
                   .sort_values("mean_of_both").iloc[-1]["lm_weight"])       # 0.8, chosen in notebook 05
    # All 10,000 labelled rows, in the same order as the notebooks (training split, then validation
    # split). The bagged trees draw bootstrap samples by row position, so the order matters for
    # reproducing the exact file.
    train = pd.concat(get_split(verbose=False), ignore_index=True)
    test = load_test(verbose=False)
    columns = FEATURE_SETS["smol" if variant == "smol" else "final"]

    print(f"variant={variant}: {len(columns)} features, classifier weight {weight}")
    classifier = lm_linear_plus_trees(columns=columns).fit(train["text"], train["label"])
    svm = make_model("robust_svm_tuned").fit(train["text"], train["label"])
    p_lm = predict_proba1(classifier, test["text"])
    p_svm = blank_push(predict_proba1(svm, test["text"]), test["text"])
    scores = blend(p_svm, p_lm, weight)

    tag = {"final": "joling_final", "smol": "joling_final_smol", "tf": "joling_final_tf"}[variant]
    if variant == "tf":                         # add Aidan's DistilRoBERTa test predictions as a third model
        tf = pd.read_csv(RESULTS_DIR / "preds_transformer.csv")
        tf = tf[tf["split"] == "test"].set_index("id").loc[test["id"], "prob"].to_numpy()
        scores = rank_average([scores, tf], [1 - TRANSFORMER_WEIGHT, TRANSFORMER_WEIGHT])
    return write_submission(test, scores, tag=tag)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rebuild the final Kaggle submission.")
    ap.add_argument("--variant", choices=["final", "smol", "tf"], default="final")
    main(ap.parse_args().variant)
