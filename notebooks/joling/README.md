# notebooks/joling

Jo-Ling's experiments. Goal: a detector that still works on writing domains
that are not in `train.csv`.

Rules followed here:
- Same shared split as the rest of the team (`src.data.get_split()`, 8,000 train / 2,000 val, seed 42).
- Nothing in `src/` is modified; all new code lives in this folder.

| File | What it is |
|---|---|
| `01_domain_yardstick.ipynb` | Scores the team's models on topics they did not train on, and with layout removed. About 20 minutes on a laptop. |
| `02_lm_predictability.ipynb` | Classifier on GPT-2 "predictability" features, blended with `robust_svm_tuned`. Writes three Kaggle files. About 12 minutes (uses the cached features). |
| `03_repetition_fingerprint.ipynb` | Finds the weak spot of `02` (AI text that avoids repeating itself), adds repetition features, re-blends. Writes three more Kaggle files. About 3 minutes (uses cached features and scores). |
| `repetition_features.py` | Word-count and GPT-2 repetition features. `python notebooks/joling/repetition_features.py` rebuilds the cache (about 30 minutes, resumable). |
| `lm_features.py` | Computes the 24 language-model features. `python notebooks/joling/lm_features.py` rebuilds the cache (about 30 minutes, resumable). |
| `lm_model.py` | Classifiers on the cached features and rank blending. |
| `domain_eval.py` | `TopicDomains` (unsupervised topic groups) and `domain_stress_test` (the four validation numbers). |
| `plots.py` | Chart helpers. |
| `requirements-extra.txt` | `torch` and `transformers`, needed only to rebuild the feature cache. |

## Reproduce
```bash
pip install -r requirements.txt
pip install -r notebooks/joling/requirements-extra.txt -c requirements.txt   # only to rebuild the cache
python notebooks/joling/lm_features.py                                       # optional: results/joling_lm_features_gpt2.csv is committed
jupyter nbconvert --to notebook --execute --inplace notebooks/joling/01_domain_yardstick.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/joling/02_lm_predictability.ipynb
python notebooks/joling/repetition_features.py                               # optional: the cache is committed
jupyter nbconvert --to notebook --execute --inplace notebooks/joling/03_repetition_fingerprint.ipynb   # needs the outputs of 02
```
Notebook `02` writes `submissions/submission_<time>_joling_{lm,blend,blend_blank}.csv`; notebook `03` writes
`submissions/submission_<time>_joling_{lm_rp,blend_rp,blend_rp_blank}.csv`.
Other outputs go to `results/joling_*.csv`, `results/preds_joling_*.csv` and `figures/joling_*.png`. Seed: 42 everywhere.

## Result so far
| Model | Validation | Unseen domain + layout removed | Kaggle public |
|---|---|---|---|
| `robust_svm_tuned` | 0.979 | 0.933 | not recorded |
| `robust_svm_tuned_blank` (team's previous best) | 0.991 | 0.933 | 0.913 |
| Language-model classifier | 0.979 | 0.976 | 0.914 |
| Blend: SVM + language-model classifier (weight 0.6) | 0.993 | 0.987 | 0.936 |
| Blend: SVM + blank-line boost + language-model classifier | 0.995 | 0.987 | **0.941** |
| `03`: classifier with repetition features | 0.991 | 0.988 | 0.937 |
| `03`: blend, SVM + new classifier (weight 0.8) | 0.995 | 0.991 | not uploaded yet |
| `03`: blend, SVM + blank-line boost + new classifier | 0.996 | 0.991 | not uploaded yet |

Validation columns use the shared 2,000-row set. Kaggle public scores (900 test rows) were read from the
submissions page on 2026-11-10 and are listed in `results/joling_kaggle_public.csv`.

The pretrained GPT-2 and distilgpt2 weights are used as fixed feature extractors on the laptop; no API is called
and no weights are fine-tuned.
