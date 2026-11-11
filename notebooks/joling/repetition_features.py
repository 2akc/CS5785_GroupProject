"""Features that look for the fingerprint of a "repetition penalty".

Many text generators are run with a repetition penalty: a setting that pushes the model
away from tokens it has already used. The result reads less smoothly, so the
predictability features in ``lm_features.py`` take it for human writing. But it leaves a
different trace: the text repeats itself LESS than natural writing does.

Two sets of features:

* ``WORD_FEATURES`` (19, no language model): plain counts such as the share of words used
  only once, or how often a word repeats within the last 10 words.
* ``RP_FEATURES`` (11, GPT-2): at every position GPT-2 tells us how much probability it
  puts on "the next token is one that was already used". Averaged over the text this is the
  repetition we should EXPECT. We compare it with how often the writer ACTUALLY repeated.
  Human text roughly matches the expectation; text written under a repetition penalty
  repeats far less than expected.

Run once to compute and cache both sets for every file (about 30 minutes, resumable):

    python notebooks/joling/repetition_features.py   # writes results/joling_repetition_features_gpt2.csv
"""
from __future__ import annotations

import re
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import ID_COL, RESULTS_DIR, TEXT_COL  # noqa: E402
from src.features import normalize_style  # noqa: E402

from lm_features import MAX_TOKENS, _batches, load_models  # noqa: E402

WORD_FEATURES = [
    "ttr", "mattr50", "hapax_share", "repeat_w10", "repeat_w50", "content_ttr", "content_repeat_w50",
    "content_repeat_all", "max_content_count", "top5_content_share", "distinct2", "distinct3",
    "repeated_bigram_share", "stop_share", "stop_ttr", "compress_ratio", "first_word_diversity",
    "the_share", "and_share",
]
RP_FEATURES = [
    "rp_expected",            # repetition GPT-2 expects (mean probability on already-used tokens)
    "rp_actual",              # repetition that happened (share of tokens that were already used)
    "rp_deficit",             # expected - actual
    "rp_ratio",               # actual / expected
    "rp_z",                   # the deficit in standard units (a longer text gives a surer answer)
    "rp_top1_seen",           # how often GPT-2's top guess was an already-used token
    "rp_top1_seen_avoided",   # ... and the writer wrote something else
    "rp_avoid_given_seen",    # of those cases, the share where the writer wrote something else
    "rp_ll_repeat",           # log-probability of the real token when it was a repeat
    "rp_ll_new",              # log-probability of the real token when it was new
    "rp_deficit_late",        # the deficit in the second half of the text (more context = more to avoid)
]

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(ENGLISH_STOP_WORDS)
_SENT = re.compile(r"(?<=[.!?])\s+")


def _repeat_rate(seq, window: int) -> float:
    """Share of items that already appeared among the previous ``window`` items."""
    if len(seq) < 2:
        return 0.0
    return float(np.mean([seq[i] in seq[max(0, i - window):i] for i in range(1, len(seq))]))


def _moving_ttr(seq, window: int = 50) -> float:
    """Type-token ratio averaged over sliding windows, so long and short texts are comparable."""
    if len(seq) <= window:
        return len(set(seq)) / max(len(seq), 1)
    return float(np.mean([len(set(seq[i:i + window])) / window for i in range(0, len(seq) - window + 1, 5)]))


def word_repetition_vector(text: str) -> dict:
    t = normalize_style(text)
    words = _WORD.findall(t.lower())
    n = max(len(words), 1)
    counts = Counter(words)
    content = [w for w in words if w not in _STOP]
    c_counts = Counter(content)
    n_content = max(len(content), 1)
    bigrams, trigrams = list(zip(words, words[1:])), list(zip(words, words[1:], words[2:]))
    b_counts = Counter(bigrams)
    first = [m[0] for m in (_WORD.findall(s.lower())[:1] for s in _SENT.split(t) if s.strip()) if m]
    raw = t.encode("utf-8")
    return {
        "ttr": len(counts) / n,
        "mattr50": _moving_ttr(words),
        "hapax_share": sum(v == 1 for v in counts.values()) / max(len(counts), 1),
        "repeat_w10": _repeat_rate(words, 10),
        "repeat_w50": _repeat_rate(words, 50),
        "content_ttr": len(c_counts) / n_content,
        "content_repeat_w50": _repeat_rate(content, 50),
        "content_repeat_all": 1 - len(c_counts) / n_content,
        "max_content_count": max(c_counts.values()) / n_content if c_counts else 0.0,
        "top5_content_share": sum(v for _, v in c_counts.most_common(5)) / n_content,
        "distinct2": len(set(bigrams)) / max(len(bigrams), 1),
        "distinct3": len(set(trigrams)) / max(len(trigrams), 1),
        "repeated_bigram_share": sum(v for v in b_counts.values() if v > 1) / max(len(bigrams), 1),
        "stop_share": 1 - len(content) / n,
        "stop_ttr": len({w for w in words if w in _STOP}) / max(n - len(content), 1),
        "compress_ratio": len(zlib.compress(raw, 9)) / max(len(raw), 1),
        "first_word_diversity": len(set(first)) / max(len(first), 1),
        "the_share": counts["the"] / n,
        "and_share": counts["and"] / n,
    }


def word_repetition_features(texts) -> pd.DataFrame:
    return pd.DataFrame([word_repetition_vector(t) for t in texts])


def lm_repetition_features(texts, model_name: str = "gpt2", token_budget: int = 768, models=None) -> pd.DataFrame:
    """One row of ``RP_FEATURES`` per text. Same text preparation as ``lm_features.py``."""
    import torch

    tok, model, _ = models or load_models(model_name, None)
    dev = next(model.parameters()).device
    ids = [tok(tok.bos_token + normalize_style(t), truncation=True, max_length=MAX_TOKENS).input_ids for t in texts]
    lengths = [len(i) for i in ids]
    rows: list[dict | None] = [None] * len(ids)
    with torch.no_grad():
        for idx in _batches(np.argsort(lengths), lengths, token_budget):
            enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(dev)
            x = enc.input_ids
            logp = torch.log_softmax(model(**enc).logits[:, :-1].float(), dim=-1)     # row k predicts token k+1
            target = x[:, 1:]
            # seen[b, k, v] = 1 if token v appears in the text up to position k (the context for row k)
            seen = torch.zeros_like(logp)
            seen.scatter_(2, x[:, :-1, None], 1.0)
            seen[:, :, tok.bos_token_id] = 0                                           # the start token does not count
            seen = seen.cumsum(1).clamp_(max=1.0)
            expected = (logp.exp() * seen).sum(-1)                                     # P(next token is an already-used one)
            actual = seen.gather(2, target[..., None])[..., 0]                         # was the real token already used?
            top1 = logp.argmax(-1)
            top1_seen = seen.gather(2, top1[..., None])[..., 0]
            ll = logp.gather(2, target[..., None])[..., 0]
            hit = (top1 == target).cpu().numpy()
            del seen, logp
            E, A, T, LL = (v.cpu().numpy().astype(np.float64) for v in (expected, actual, top1_seen, ll))
            for b, i in enumerate(idx):
                n = lengths[i] - 1
                e, a, t1, l, h = E[b, :n], A[b, :n], T[b, :n] > 0, LL[b, :n], hit[b, :n]
                avoided = t1 & ~h
                rows[i] = {
                    "rp_expected": e.mean(), "rp_actual": a.mean(), "rp_deficit": e.mean() - a.mean(),
                    "rp_ratio": a.mean() / max(e.mean(), 1e-6),
                    "rp_z": (e.sum() - a.sum()) / np.sqrt(max((e * (1 - e)).sum(), 1e-6)),
                    "rp_top1_seen": t1.mean(), "rp_top1_seen_avoided": avoided.mean(),
                    "rp_avoid_given_seen": avoided.sum() / max(t1.sum(), 1),
                    "rp_ll_repeat": l[a > 0].mean() if (a > 0).any() else l.mean(),
                    "rp_ll_new": l[a == 0].mean() if (a == 0).any() else l.mean(),
                    "rp_deficit_late": (e[n // 2:].mean() - a[n // 2:].mean()) if n > 4 else 0.0,
                }
            if dev.type == "mps":
                torch.mps.empty_cache()
    return pd.DataFrame(rows)


def cache_path(model_name: str = "gpt2") -> Path:
    return RESULTS_DIR / f"joling_repetition_features_{model_name.split('/')[-1]}.csv"


def load_cached(model_name: str = "gpt2") -> pd.DataFrame:
    path = cache_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run:  python notebooks/joling/repetition_features.py")
    return pd.read_csv(path).set_index(ID_COL)


def build_cache(model_name: str = "gpt2", limit: int | None = None, chunk: int = 1000):
    """Both feature sets for train / unlabeled / test, saved every ``chunk`` texts (resumable)."""
    from src.data import load_test, load_train, load_unlabeled

    path = cache_path(model_name) if not limit else cache_path(model_name).with_suffix(".sample.csv")
    done_ids = set(pd.read_csv(path, usecols=[ID_COL])[ID_COL]) if path.exists() else set()
    if done_ids:
        print(f"resuming: {len(done_ids):,} texts already in {path.name}", flush=True)
    frames = {"train": load_train(verbose=False), "unlabeled": load_unlabeled(verbose=False),
              "test": load_test(verbose=False)}
    models = load_models(model_name, None)
    for name, df in frames.items():
        df = df.head(limit) if limit else df
        df = df[~df[ID_COL].isin(done_ids)]
        print(f"{name}: {len(df):,} texts to do", flush=True)
        for start in range(0, len(df), chunk):
            part = df.iloc[start:start + chunk]
            texts = part[TEXT_COL].tolist()
            feats = pd.concat([lm_repetition_features(texts, model_name, models=models),
                               word_repetition_features(texts)], axis=1)
            feats.insert(0, "file", name)
            feats.insert(0, ID_COL, part[ID_COL].to_numpy())
            feats.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.6f")
            print(f"  {name}: {min(start + chunk, len(df)):,}/{len(df):,} saved  ({time.strftime('%H:%M:%S')})", flush=True)
    out = pd.read_csv(path)
    print(f"finished {path.relative_to(ROOT)}  ({len(out):,} rows)")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute and cache repetition features.")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--limit", type=int, default=None, help="only the first N texts per file (quick test)")
    a = ap.parse_args()
    build_cache(a.model, a.limit)
