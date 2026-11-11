#!/usr/bin/env python
"""One command -> a validated Kaggle submission in submissions/.

Examples
--------
    python predict.py --model baseline
    python predict.py --model baseline --vocab-unlabeled --pseudo-threshold 0.9 --pseudo-rounds 2
    python predict.py --ensemble baseline logreg_char             # rank-average saved preds
    python predict.py --ensemble                                 # all results/preds_*.csv
    python predict.py --model lm_svm_blend --split-order         # FINAL MODEL (Kaggle public 0.963)

``--model`` retrains the named pipeline from src/models.py on ALL of train.csv (optionally
plus pseudo-labelled unlabeled.csv) and predicts test.csv. ``--ensemble`` needs no training:
it rank-averages the test rows of results/preds_<name>.csv files.
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from src.config import ID_COL, LABEL_COL, set_seed
from src.data import get_split, load_test, load_train, load_unlabeled
from src.evaluate import (
    auroc,
    list_predictions,
    load_predictions,
    predict_proba1,
    rank_average,
    write_submission,
)
from src.models import MODELS, make_model, self_train


def run_model(args, test):
    if args.split_order:
        # Same rows, in the order the notebooks use: shared training split, then validation split.
        # Models with bootstrap sampling (lm_classifier, lm_svm_blend) need this to rebuild the
        # uploaded file exactly; for the others the order makes no practical difference.
        train = pd.concat(get_split(quick=args.quick, clean=not args.no_clean, verbose=False), ignore_index=True)
    else:
        train = load_train(clean=not args.no_clean, quick=args.quick)
    extra = None
    unl = None
    if args.vocab_unlabeled or args.pseudo_threshold:
        unl = load_unlabeled(clean=not args.no_clean)["text"].to_numpy()
    if args.vocab_unlabeled:
        extra = unl

    t0 = time.time()
    model = make_model(args.model, extra_texts=extra)
    if args.pseudo_threshold:
        model, hist, pseudo = self_train(model, train["text"], train[LABEL_COL], unl,
                                         threshold=args.pseudo_threshold,
                                         rounds=args.pseudo_rounds, balance=args.pseudo_balance)
        print(hist.to_string(index=False))
        print(f"pseudo-labels used: {len(pseudo)}")
    else:
        model.fit(train["text"], train[LABEL_COL])
    print(f"trained {args.model} on {len(train)} rows in {time.time() - t0:.1f}s")
    return predict_proba1(model, test["text"])


def run_ensemble(args, test):
    names = args.ensemble or list_predictions()
    if not names:
        sys.exit("no results/preds_*.csv files found")
    weights = args.weights
    if weights is not None and len(weights) != len(names):
        sys.exit(f"--weights has {len(weights)} values for {len(names)} models")
    test_ids = test[ID_COL]
    test_cols, val_frames = [], []
    for n in names:
        p = load_predictions(n)
        t = p[p["split"] == "test"].set_index(ID_COL)["prob"]
        missing = set(test_ids) - set(t.index)
        if missing or t.index.duplicated().any():
            sys.exit(f"preds_{n}.csv: {len(missing)} test ids missing / duplicated ids")
        test_cols.append(t.loc[test_ids].to_numpy())
        v = p[p["split"] == "val"].set_index(ID_COL)
        val_frames.append(v)
        print(f"  {n:30s} val AUROC = {auroc(v[LABEL_COL], v['prob']):.4f}  (n={len(v)})")
    common = sorted(set.intersection(*(set(v.index) for v in val_frames)))
    if common:
        y = val_frames[0].loc[common, LABEL_COL]
        ens_val = rank_average([v.loc[common, "prob"] for v in val_frames], weights)
        print(f"  {'ENSEMBLE':30s} val AUROC = {auroc(y, ens_val):.4f}  (n={len(common)} shared val ids)")
    return rank_average(test_cols, weights)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--model", default="baseline", choices=sorted(MODELS))
    g.add_argument("--ensemble", nargs="*", metavar="NAME",
                   help="rank-average results/preds_<NAME>.csv (all files if no names given)")
    ap.add_argument("--weights", nargs="+", type=float, help="ensemble weights, same order as names")
    ap.add_argument("--vocab-unlabeled", action="store_true", help="fit TF-IDF vocab on train + unlabeled")
    ap.add_argument("--pseudo-threshold", type=float, help="enable self-training, e.g. 0.9")
    ap.add_argument("--pseudo-rounds", type=int, default=1)
    ap.add_argument("--pseudo-balance", action="store_true", help="add equal numbers of each class")
    ap.add_argument("--no-clean", action="store_true", help="use raw text (no light cleaning)")
    ap.add_argument("--quick", action="store_true", help="train on train_starter_2000.csv (debug)")
    ap.add_argument("--split-order", action="store_true",
                    help="order the training rows as train split + val split (as in the notebooks)")
    ap.add_argument("--tag", default="", help="suffix for the submission filename")
    args = ap.parse_args(argv)

    set_seed()
    test = load_test(clean=not args.no_clean)
    if args.ensemble is not None:
        probs = run_ensemble(args, test)
        tag = args.tag or "ensemble"
    else:
        probs = run_model(args, test)
        tag = args.tag or args.model
    return write_submission(test, probs, tag=tag)


if __name__ == "__main__":
    main()
