"""Predictability and repetition features from ANY small causal language model, in one pass.

``lm_features.py`` and ``repetition_features.py`` were written for GPT-2 and each run the
model once. This file computes the same per-token numbers with a single run of the model,
and works for other models with a GPT-style tokenizer, e.g. SmolLM2 (2024).

Output columns: the 22 predictability features that need only one model (everything in
``lm_features.FEATURES`` except the two Binoculars-style ones) plus the 11
``repetition_features.RP_FEATURES``.

    python notebooks/joling/lm_onepass.py --model HuggingFaceTB/SmolLM2-360M
    # writes results/joling_onepass_features_SmolLM2-360M.csv  (about 50 minutes, resumable)

The model runs on this laptop with its published weights; nothing is fine-tuned.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import ID_COL, RESULTS_DIR, TEXT_COL  # noqa: E402
from src.features import normalize_style  # noqa: E402

from lm_features import FEATURES, MAX_TOKENS, _batches, _device, _summarise  # noqa: E402
from repetition_features import RP_FEATURES  # noqa: E402

ONEPASS_FEATURES = [f for f in FEATURES if f not in ("xent_mean", "binoculars")] + RP_FEATURES
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"


def short_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def load_model(model_name: str = DEFAULT_MODEL):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(_device()).eval()
    return tok, model


def onepass_features(texts, model_name: str = DEFAULT_MODEL, token_budget: int = 768, model=None) -> pd.DataFrame:
    """One row of ``ONEPASS_FEATURES`` (plus ``n_tokens``) per text, in the order of ``texts``."""
    import torch

    tok, lm = model or load_model(model_name)
    dev = next(lm.parameters()).device
    start = tok.bos_token or tok.eos_token                     # a start token, so the first real token is predicted too
    start_id = tok.convert_tokens_to_ids(start)
    ids = [tok(start + normalize_style(t), truncation=True, max_length=MAX_TOKENS).input_ids for t in texts]
    lengths = [len(i) for i in ids]
    rows: list[dict | None] = [None] * len(ids)
    with torch.no_grad():
        for idx in _batches(np.argsort(lengths), lengths, token_budget):
            enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(dev)
            x = enc.input_ids
            logp = torch.log_softmax(lm(**enc).logits[:, :-1].float(), dim=-1)      # row k predicts token k+1
            target = x[:, 1:]
            ll = logp.gather(2, target[..., None])[..., 0]
            rank = (logp > ll[..., None]).sum(-1)
            # repetition: which tokens are already in the context at each position?
            seen = torch.zeros_like(logp)
            seen.scatter_(2, x[:, :-1, None], 1.0)
            seen[:, :, start_id] = 0
            seen = seen.cumsum(1).clamp_(max=1.0)
            p = logp.exp()
            expected = (p * seen).sum(-1)
            actual = seen.gather(2, target[..., None])[..., 0]
            top1 = logp.argmax(-1)
            top1_seen = seen.gather(2, top1[..., None])[..., 0]
            hit = (top1 == target).cpu().numpy()
            del seen
            t = p.mul_(logp)                                                         # p * log p (in place)
            ent = -t.sum(-1)
            var = t.mul_(logp).sum(-1) - ent ** 2
            del t, p, logp
            LL, EN, VA, RK, E, A, T1 = (v.cpu().numpy().astype(np.float64)
                                        for v in (ll, ent, var, rank, expected, actual, top1_seen))
            for b, i in enumerate(idx):
                n = lengths[i] - 1
                l, e, a, t1, h = LL[b, :n], E[b, :n], A[b, :n], T1[b, :n] > 0, hit[b, :n]
                avoided = t1 & ~h
                row = {"n_tokens": n, **_summarise(l, EN[b, :n], VA[b, :n], RK[b, :n], None)}
                row.update({
                    "rp_expected": e.mean(), "rp_actual": a.mean(), "rp_deficit": e.mean() - a.mean(),
                    "rp_ratio": a.mean() / max(e.mean(), 1e-6),
                    "rp_z": (e.sum() - a.sum()) / np.sqrt(max((e * (1 - e)).sum(), 1e-6)),
                    "rp_top1_seen": t1.mean(), "rp_top1_seen_avoided": avoided.mean(),
                    "rp_avoid_given_seen": avoided.sum() / max(t1.sum(), 1),
                    "rp_ll_repeat": l[a > 0].mean() if (a > 0).any() else l.mean(),
                    "rp_ll_new": l[a == 0].mean() if (a == 0).any() else l.mean(),
                    "rp_deficit_late": (e[n // 2:].mean() - a[n // 2:].mean()) if n > 4 else 0.0,
                })
                rows[i] = row
            if dev.type == "mps":
                torch.mps.empty_cache()
    return pd.DataFrame(rows)


def cache_path(model_name: str = DEFAULT_MODEL) -> Path:
    return RESULTS_DIR / f"joling_onepass_features_{short_name(model_name)}.csv"


def load_cached(model_name: str = DEFAULT_MODEL, prefix: str | None = None) -> pd.DataFrame:
    """Cached features indexed by ``id``. ``prefix`` is put in front of every feature name,
    so features from two language models can sit in one table."""
    path = cache_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run:  python notebooks/joling/lm_onepass.py --model {model_name}")
    df = pd.read_csv(path).set_index(ID_COL).drop(columns="file")
    return df.add_prefix(prefix) if prefix else df


def build_cache(model_name: str = DEFAULT_MODEL, limit: int | None = None, chunk: int = 500):
    """Features for train / unlabeled / test, saved every ``chunk`` texts (resumable)."""
    from src.data import load_test, load_train, load_unlabeled

    path = cache_path(model_name) if not limit else cache_path(model_name).with_suffix(".sample.csv")
    done_ids = set(pd.read_csv(path, usecols=[ID_COL])[ID_COL]) if path.exists() else set()
    if done_ids:
        print(f"resuming: {len(done_ids):,} texts already in {path.name}", flush=True)
    frames = {"train": load_train(verbose=False), "unlabeled": load_unlabeled(verbose=False),
              "test": load_test(verbose=False)}
    model = load_model(model_name)
    for name, df in frames.items():
        df = df.head(limit) if limit else df
        df = df[~df[ID_COL].isin(done_ids)]
        print(f"{name}: {len(df):,} texts to do", flush=True)
        for s in range(0, len(df), chunk):
            part = df.iloc[s:s + chunk]
            feats = onepass_features(part[TEXT_COL].tolist(), model_name, model=model)
            feats.insert(0, "file", name)
            feats.insert(0, ID_COL, part[ID_COL].to_numpy())
            feats.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.6f")
            print(f"  {name}: {min(s + chunk, len(df)):,}/{len(df):,} saved  ({time.strftime('%H:%M:%S')})", flush=True)
    out = pd.read_csv(path)
    print(f"finished {path.relative_to(ROOT)}  ({len(out):,} rows)")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="One-pass language-model features for any small causal LM.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="only the first N texts per file (quick test)")
    a = ap.parse_args()
    build_cache(a.model, a.limit)
