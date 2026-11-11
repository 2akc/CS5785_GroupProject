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
| `04_unseen_ai_kind.ipynb` | A third stress test: hide one kind of AI text. Compares classifier designs on it, re-chooses the blend weight, writes the current best Kaggle files. About 12 minutes. |
| `05_newer_language_model.ipynb` | Recomputes the features with SmolLM2-360M (2024) and compares GPT-2, SmolLM2 and both. Writes the current best Kaggle files. About 10 minutes. |
| `lm_onepass.py` | The same features in one pass, for any small GPT-style model. `python notebooks/joling/lm_onepass.py` rebuilds the SmolLM2 cache (about 50 minutes, resumable). |
| `kind_eval.py` | `AIKinds` (unsupervised groups of AI texts) and the unseen-kind stress test. |
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
jupyter nbconvert --to notebook --execute --inplace notebooks/joling/04_unseen_ai_kind.ipynb          # needs the outputs of 02
python notebooks/joling/lm_onepass.py                                        # optional: the cache is committed
jupyter nbconvert --to notebook --execute --inplace notebooks/joling/05_newer_language_model.ipynb   # needs the outputs of 02 and 04
```
Notebook `02` writes `submissions/submission_<time>_joling_{lm,blend,blend_blank}.csv`; notebook `03` writes
`submissions/submission_<time>_joling_{lm_rp,blend_rp,blend_rp_blank}.csv`.
Other outputs go to `results/joling_*.csv`, `results/preds_joling_*.csv` and `figures/joling_*.png`. Seed: 42 everywhere.

## Result so far
All validation columns use the shared 2,000-row set with layout removed. "Unseen" means the model was trained
without the topic (domain) or without the kind of AI text it is tested on.

| Model | Validation | Unseen domain | Unseen AI kind | Kaggle public |
|---|---|---|---|---|
| `robust_svm_tuned` | 0.979 | 0.933 | 0.961 | not recorded (0.913 with the blank-line boost) |
| `02` classifier: trees on 24 predictability features | 0.979 | 0.976 | 0.925 | 0.914 |
| `03` classifier: + repetition features (54) | 0.991 | 0.988 | 0.971 | 0.937 |
| `04` classifier: linear + bagged trees (54) | 0.989 | 0.987 | 0.982 | not uploaded |
| `02` blend: SVM + `02` classifier (weight 0.6) | 0.993 | 0.987 | 0.975 | 0.936 (0.941 with the blank-line boost) |
| `04` blend: SVM + `04` classifier (weight 0.8) | 0.993 | 0.990 | 0.988 | not uploaded |
| `05` classifier: GPT-2 + SmolLM2 features (87) | 0.994 | 0.992 | 0.988 | 0.955 |
| `05` blend: SVM + `05` classifier (weight 0.8) | 0.997 | 0.993 | 0.992 | **0.963** with the blank-line boost |
| `05` runner-up blend: SVM + SmolLM2-only classifier (weight 0.8) | 0.996 | not run | not run | 0.958 with the blank-line boost |
| `05` blend + Aidan's DistilRoBERTa (weight 0.2) | 0.998 | not run | not run | 0.961 |

Kaggle public scores (900 test rows) are listed in `results/joling_kaggle_public.csv`.
Current best files: `submissions/submission_<time>_joling_{lm_05,blend_05_blank}.csv`, written by notebook `05`.

The pretrained GPT-2, distilgpt2 and SmolLM2-360M weights are used as fixed feature extractors on the laptop;
no API is called and no weights are fine-tuned.
