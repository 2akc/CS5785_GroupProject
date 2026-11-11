"""Features that measure how PREDICTABLE a text is to a small pretrained language model.

Idea: a language model reads the text one token (word piece) at a time and, before each
token, guesses what comes next. AI-written text tends to use the tokens the model expects;
human text is more surprising. These features do not look at WHICH words are used, only at
how expected they were, so they should depend less on the topic of the text.

The model runs on this laptop (no API calls). Its weights are pretrained and are NOT
trained or fine-tuned here; the only thing we train later is a small classifier on top of
the numbers this file produces.

Run once to compute and cache features for every file (about 30 minutes; it saves as it
goes, so an interrupted run can simply be started again):

    python notebooks/joling/lm_features.py            # writes results/joling_lm_features_gpt2.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ID_COL, RESULTS_DIR, TEXT_COL  # noqa: E402
from src.features import normalize_style  # noqa: E402

MAX_TOKENS = 512        # longest texts are about 350 words, roughly 470 tokens
FEATURES = [
    # log-probability the model gave to each real token (higher = more expected)
    "ll_mean", "ll_std", "ll_min", "ll_p10", "ll_p25", "ll_p50", "ll_p75", "ll_p90",
    # entropy = how unsure the model was before seeing each token
    "ent_mean", "ent_std",
    # surprise compared with what the model itself expected (Fast-DetectGPT style)
    "curvature", "z_mean", "z_std",
    # rank of the real token among the model's guesses (GLTR style); rank 1 = top guess
    "top1", "top5", "top10", "top100", "beyond1000", "log_rank_mean", "ll_per_log_rank",
    # does predictability change along the text?
    "ll_half_diff", "ll_autocorr",
    # second model as a reference point (Binoculars style)
    "xent_mean", "binoculars",
]


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def _summarise(ll, ent, var, rank, xent) -> dict:
    """Turn the per-token numbers of ONE text into the fixed list of features."""
    z = (ll + ent) / np.sqrt(np.maximum(var, 1e-8))      # per-token surprise in standard units
    log_rank = np.log1p(rank)
    half = len(ll) // 2
    f = {
        "ll_mean": ll.mean(), "ll_std": ll.std(), "ll_min": ll.min(),
        **{f"ll_p{q}": np.percentile(ll, q) for q in (10, 25, 50, 75, 90)},
        "ent_mean": ent.mean(), "ent_std": ent.std(),
        "curvature": (ll + ent).sum() / np.sqrt(max(var.sum(), 1e-8)),
        "z_mean": z.mean(), "z_std": z.std(),
        "top1": (rank < 1).mean(), "top5": (rank < 5).mean(), "top10": (rank < 10).mean(),
        "top100": (rank < 100).mean(), "beyond1000": (rank >= 1000).mean(),
        "log_rank_mean": log_rank.mean(),
        "ll_per_log_rank": -ll.mean() / max(log_rank.mean(), 1e-8),
        "ll_half_diff": ll[half:].mean() - ll[:half].mean() if half else 0.0,
        "ll_autocorr": float(np.corrcoef(ll[:-1], ll[1:])[0, 1]) if len(ll) > 2 and ll.std() > 0 else 0.0,
    }
    if xent is not None:
        f["xent_mean"] = xent.mean()
        f["binoculars"] = -ll.mean() / max(xent.mean(), 1e-8)
    return f


def _batches(order, lengths, token_budget: int):
    """Group texts (already sorted by length) so each batch holds at most ``token_budget``
    tokens after padding: many short texts together, long texts in small groups."""
    batch = []
    for i in order:
        if batch and (len(batch) + 1) * lengths[i] > token_budget:
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


def lm_features(texts, model_name: str = "gpt2", reference_name: str | None = "distilgpt2",
                token_budget: int = 1024, progress: bool = True, models=None) -> pd.DataFrame:
    """One row of ``FEATURES`` (plus ``n_tokens``) per text, in the same order as ``texts``.

    ``model_name``: the model whose surprise we measure. ``reference_name``: a second,
    smaller model with the same vocabulary, used only for the two Binoculars-style
    features (pass ``None`` to skip them). ``token_budget`` caps the tokens per batch:
    every token needs a table of ~50,000 probabilities, so 1,024 tokens is about 200 MB per
    table, which keeps an 8 GB laptop out of trouble. ``models`` lets a caller pass
    already-loaded ``(tokenizer, model, reference)`` so they are not loaded again.
    """
    import torch

    dev = _device()
    tok, model, ref = models or load_models(model_name, reference_name)

    # Layout is removed first (same normalisation as the team's robust_* models), so these
    # features cannot pick up line-break habits. The start token gives the first real
    # token something to be predicted from.
    prepared = [tok.bos_token + normalize_style(t) for t in texts]
    ids = [tok(t, truncation=True, max_length=MAX_TOKENS).input_ids for t in prepared]
    lengths = [len(i) for i in ids]
    order = np.argsort(lengths)                          # similar lengths together = little padding
    rows: list[dict | None] = [None] * len(ids)
    t0, done = time.time(), 0
    with torch.no_grad():
        for idx in _batches(order, lengths, token_budget):
            enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(dev)
            logp = torch.log_softmax(model(**enc).logits[:, :-1].float(), dim=-1)
            target = enc.input_ids[:, 1:]
            ll = logp.gather(-1, target[..., None])[..., 0]          # log-prob of the real token
            rank = (logp > ll[..., None]).sum(-1)                    # how many guesses ranked higher
            xent = None
            if ref is not None:
                q = torch.softmax(ref(**enc).logits[:, :-1].float(), dim=-1)
                xent = -q.mul_(logp).sum(-1)                         # reference model's guesses, scored by the main model
                del q
            t = logp.exp().mul_(logp)                                # p * log p   (in place: one big table, not three)
            ent = -t.sum(-1)
            var = t.mul_(logp).sum(-1) - ent ** 2                    # sum p * (log p)^2  -  entropy^2
            del t, logp
            ll, ent, var, rank = (x.cpu().numpy().astype(np.float64) for x in (ll, ent, var, rank))
            xent = xent.cpu().numpy().astype(np.float64) if xent is not None else None
            for b, i in enumerate(idx):
                n = lengths[i] - 1                                   # real tokens (padding sits after them)
                rows[i] = {"n_tokens": n, **_summarise(ll[b, :n], ent[b, :n], var[b, :n], rank[b, :n],
                                                       xent[b, :n] if xent is not None else None)}
            if dev == "mps":
                torch.mps.empty_cache()                              # hand memory back between batches
            before, done = done, done + len(idx)
            if progress and (done // 500 > before // 500 or done == len(order)):
                rate = done / (time.time() - t0)
                print(f"  {done:>6,}/{len(order):,} texts  ({rate:.1f}/s, about {(len(order) - done) / rate / 60:.0f} min left)",
                      flush=True)
    return pd.DataFrame(rows)


def load_models(model_name: str = "gpt2", reference_name: str | None = "distilgpt2"):
    """Load the tokenizer, the main model and (optionally) the reference model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(dev).eval()
    ref = AutoModelForCausalLM.from_pretrained(reference_name).to(dev).eval() if reference_name else None
    return tok, model, ref


def cache_path(model_name: str = "gpt2") -> Path:
    return RESULTS_DIR / f"joling_lm_features_{model_name.split('/')[-1]}.csv"


def load_cached(model_name: str = "gpt2") -> pd.DataFrame:
    """Cached features for every text in train / unlabeled / test, indexed by ``id``."""
    path = cache_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run:  python notebooks/joling/lm_features.py")
    return pd.read_csv(path).set_index(ID_COL)


def build_cache(model_name: str = "gpt2", reference_name: str | None = "distilgpt2",
                limit: int | None = None, chunk: int = 1000):
    """Compute features for all three data files and save them as one CSV.

    Work is saved after every ``chunk`` texts. If the run is interrupted, running the same
    command again continues from where it stopped.
    """
    from src.data import load_test, load_train, load_unlabeled

    path = cache_path(model_name) if not limit else cache_path(model_name).with_suffix(".sample.csv")
    done_ids = set(pd.read_csv(path, usecols=[ID_COL])[ID_COL]) if path.exists() else set()
    if done_ids:
        print(f"resuming: {len(done_ids):,} texts already in {path.name}", flush=True)
    frames = {"train": load_train(verbose=False), "unlabeled": load_unlabeled(verbose=False),
              "test": load_test(verbose=False)}
    models = load_models(model_name, reference_name)
    for name, df in frames.items():
        df = df.head(limit) if limit else df
        df = df[~df[ID_COL].isin(done_ids)]
        print(f"{name}: {len(df):,} texts to do", flush=True)
        for start in range(0, len(df), chunk):
            part = df.iloc[start:start + chunk]
            feats = lm_features(part[TEXT_COL].tolist(), model_name, reference_name, models=models, progress=False)
            feats.insert(0, "file", name)
            feats.insert(0, ID_COL, part[ID_COL].to_numpy())
            feats.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.6f")
            print(f"  {name}: {min(start + chunk, len(df)):,}/{len(df):,} saved  ({time.strftime('%H:%M:%S')})", flush=True)
    out = pd.read_csv(path)
    print(f"finished {path.relative_to(ROOT)}  ({len(out):,} rows)")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute and cache language-model features.")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--reference", default="distilgpt2", help='second model, or "none"')
    ap.add_argument("--limit", type=int, default=None, help="only the first N texts per file (quick test)")
    a = ap.parse_args()
    build_cache(a.model, None if a.reference == "none" else a.reference, a.limit)
