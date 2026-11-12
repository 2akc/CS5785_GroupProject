# AI vs. human text detection

Classifies ~200-word passages as human-written (`0`) or AI-generated (`1`) for the
CS5785 (Fall 2026) Kaggle competition. Scored by **AUROC**, so predictions are probabilities.

**Baseline** (word + char TF-IDF → logistic regression): val AUROC **0.956**, 5-fold CV **0.954**.

**Final model** (`lm_svm_blend`): a classifier on features that measure how predictable a text is to small
pretrained language models (GPT-2, SmolLM2-360M) and how much it repeats itself, rank-averaged with the tuned SVM +
blank-line boost. Val AUROC **0.997**, Kaggle public **0.963**. How it was built: [notebooks/joling/](notebooks/joling/README.md).

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # macOS: brew install libomp (for LightGBM)
```
Put the Kaggle CSVs in `data/` (see [data/README.md](data/README.md)).

## Reproduce the submission
```bash
python predict.py --model lm_svm_blend --split-order  # FINAL: rebuilds the selected Kaggle file exactly (~1 min)
python predict.py --model baseline                    # train on all labelled data, predict test
python predict.py --ensemble skeleton_only baseline   # rank-average saved predictions
```
The final model reads cached language-model features from `results/joling_*features*.csv` (committed), so it needs
no GPU and no extra packages. To recompute the caches (about 110 minutes on a laptop):
```bash
pip install -r notebooks/joling/requirements-extra.txt   # torch, transformers
python -m src.lm_features && python -m src.repetition_features && python -m src.lm_onepass
```
`--split-order` trains on the same rows in the order the notebooks use (train split, then validation split); the final
model's bagged trees draw bootstrap samples by row position, so this is what makes the file match bit for bit.
Writes a validated `submissions/submission_<timestamp>.csv` (3,000 rows, `id,label`, values in [0, 1]).

## Structure
| Path | Contents |
|---|---|
| `src/data.py` | Loading, light cleaning (encoding fix + NFC only), fixed 80/20 train/val split |
| `src/features.py` | Word / char TF-IDF, function-word skeleton n-grams, stylometric features |
| `src/models.py` | Named pipelines (LogReg, NB-SVM, MNB, SVMs, `lm_classifier`, `lm_svm_blend`), pseudo-labelling |
| `src/lm_features.py`, `src/repetition_features.py`, `src/lm_onepass.py` | Language-model features: token predictability (GPT-2), repetition expected vs. actual, and the same from SmolLM2-360M. Each file is also a script that builds its cache |
| `src/lm_model.py` | Classifiers on the cached features (logistic regression + bagged trees) and the rank blend with a text model |
| `src/evaluate.py` | AUROC, cross-validation, result logging, prediction files, submission checks |
| `notebooks/` | Experiments: baseline, EDA, unlabeled data, model comparison, training size, advanced classical models, generalization, ensemble |
| `notebooks/joling/` | Stress tests for unseen topics / unseen kinds of AI text, the language-model classifier and blend, report experiments (notebooks `01`-`06`) |
| `results/`, `figures/` | Experiment outputs used in the report |
| `report/` | LaTeX report (`cd report && latexmk -pdf main.tex`) |

