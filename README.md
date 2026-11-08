# AI vs. human text detection

Classifies ~200-word passages as human-written (`0`) or AI-generated (`1`) for the
CS5785 (Fall 2026) Kaggle competition. Scored by **AUROC**, so predictions are probabilities.

**Baseline** (word + char TF-IDF → logistic regression): val AUROC **0.956**, 5-fold CV **0.954**.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # macOS: brew install libomp (for LightGBM)
```
Put the Kaggle CSVs in `data/` (see [data/README.md](data/README.md)).

## Reproduce the submission
```bash
python predict.py --model baseline                  # train on all labelled data, predict test
python predict.py --ensemble skeleton_only baseline # rank-average saved predictions
```
Writes a validated `submissions/submission_<timestamp>.csv` (3,000 rows, `id,label`, values in [0, 1]).

## Structure
| Path | Contents |
|---|---|
| `src/data.py` | Loading, light cleaning (encoding fix + NFC only), fixed 80/20 train/val split |
| `src/features.py` | Word / char TF-IDF, function-word skeleton n-grams, stylometric features |
| `src/models.py` | Named classical pipelines (LogReg, NB-SVM, MNB), pseudo-labelling |
| `src/evaluate.py` | AUROC, cross-validation, result logging, prediction files, submission checks |
| `notebooks/` | Experiments: baseline, EDA, unlabeled data, model comparison, training size, advanced classical models, generalization, ensemble |
| `results/`, `figures/` | Experiment outputs used in the report |
| `report/` | LaTeX report (`cd report && latexmk -pdf main.tex`) |

## AI tool usage
TODO before submission.
