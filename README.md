# CS5785 Kaggle project: AI vs. human text

Classify ~200-word passages as human-written (`0`) or AI-generated (`1`). Kaggle scores
**AUROC**, so every submission contains **probabilities**, not hard labels.
CS5785 Applied ML, Cornell Tech, Fall 2026. Deadline **Nov 11, 2026** (treat Nov 11 as buffer).

## Setup (CPU, laptop)

```bash
git clone https://github.com/2akc/CS5785_GroupProject.git && cd CS5785_GroupProject
python3.12 -m venv .venv            # Python 3.11 or 3.12 recommended
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name cs5785-aihuman --display-name "CS5785 (aihuman)"
```

macOS only: LightGBM needs OpenMP, so run `brew install libomp`.

Optional GPU notebook (Kaggle / Colab): `pip install -r requirements.txt -r requirements-gpu.txt`.

## Data

Download the competition files from the Kaggle **Data** tab into `data/` (it is gitignored,
so data is never committed). See [data/README.md](data/README.md) for the expected files
and a Kaggle CLI one-liner. On the first run, `src.data.get_split()` creates
`data/split_ids.json`. The split is deterministic (seed 42, 80/20 stratified, exact duplicates
kept on the same side), so every teammate gets the identical split.

Set `AIHUMAN_DATA_DIR=/some/path` to read data from elsewhere (e.g. `/kaggle/input/...`).

## Reproduce the final submission

```bash
python predict.py --model baseline                       # retrain on all of train.csv -> submissions/
python predict.py --ensemble baseline deberta_small      # or rank-average saved predictions
```

`predict.py` validates the output before writing it: exactly 3,000 rows, columns `id,label`,
every test id exactly once, values in [0, 1], no NaNs. It fails loudly otherwise.
Other options: `--vocab-unlabeled`, `--pseudo-threshold 0.9 --pseudo-rounds 2`,
`--pseudo-balance`, `--no-clean`, `--quick`. Run `python predict.py -h` for the full list.

> TODO (Tue Nov 10): replace the commands above with the exact command for the final submission.

## Repository layout

| Path | What | Owner |
|---|---|---|
| `src/config.py` | repo-relative paths, `SEED = 42`, `VAL_FRAC = 0.2`, `set_seed()` | Allen |
| `src/data.py` | loaders, light `clean_text`, duplicate/overlap report, shared `get_split()`, `get_cv_folds()` | Allen |
| `src/features.py` | word / char TF-IDF, stylometric features, `build_features(name)` | Allen |
| `src/models.py` | named pipelines for `predict.py`, pseudo-labelling / self-training | Allen |
| `src/evaluate.py` | `auroc`, `cross_val_auroc`, `plot_roc`, `log_result`, `save_predictions`, `rank_average`, submission validation | Allen |
| `predict.py` | one command → validated `submissions/submission_<timestamp>.csv` | Allen |
| `results/` | small CSV/JSON outputs + `preds_<model>.csv` (committed) | everyone |
| `figures/` | report figures (committed) | everyone |
| `report/` | LaTeX report (`main.tex` + `sections/`) | everyone |

### Notebooks

Every notebook has a `QUICK = True/False` flag at the top (`True` uses the 2,000-row starter
file for debugging). Each one uses the shared split, writes to `results/` and `figures/`,
and runs top to bottom with *Restart & Run All*. Only commit numbers from `QUICK = False` runs.
QUICK-mode files get a `_quick` suffix.

| Notebook | Purpose | Owner | Report section |
|---|---|---|---|
| `00_baseline` | word+char TF-IDF + LogReg: val AUROC, 5-fold CV, predictions, first submission | Allen | Intro / model comparison |
| `01_data_cleaning_eda` | class balance, lengths, duplicates/overlap, train-vs-test shift, raw vs. cleaned | Allen | Intro, generalization |
| `02_unlabeled_pseudolabel` | vocab on train+unlabeled, pseudo-label thresholds, self-training rounds | Allen | Unlabeled data |
| `10_model_comparison` | feature × model grid, hyperparameter sweep | Teammate A | Model comparison |
| `11_training_size` | learning curves, 7 fractions × 3 seeds | Teammate A | Training size |
| `20_transformer_finetune` | DeBERTa-v3-small fine-tune (GPU, Kaggle/Colab) | Teammate B | Improvements |
| `21_generalization` | adversarial validation, test-like val set, AUROC by length | Teammate B | Generalization |
| `22_error_analysis_ensemble` | FP/FN, top features, calibration, rank-average ensemble | Teammate B | Error analysis, improvements |

### Shared conventions

```python
from src.data import get_split, load_test
from src.evaluate import cross_val_auroc, log_result, save_predictions

train_df, val_df = get_split(quick=QUICK)                  # always this split
log_result("model_comparison", {"model": "..."}, {"val_auroc": 0.97})   # -> results/model_comparison.csv
save_predictions("my_model", val_df, val_probs, test_df, test_probs)    # -> results/preds_my_model.csv
```

For `save_predictions`, `val_probs` comes from a model fit on the train split only, and
`test_probs` from the same configuration refit on all labelled data. That way anyone's model
can be ensembled and compared on the same validation rows.

Cleaning policy: `clean_text` only repairs encoding damage, applies Unicode NFC, and strips
outer whitespace. We never lowercase, strip punctuation, remove stopwords, or fix typos in the
data layer, because those are likely the signal. `clean=False` gives raw text for comparison.

## Report

```bash
cd report && latexmk -pdf main.tex      # output: report/main.pdf (gitignored)
```

Each `report/sections/*.tex` file starts with its owner and the `results/` and `figures/`
files it should use. Figures are referenced relative to `figures/`
(e.g. `\includegraphics{roc_baseline.png}`).

## Team workflow

- **Branches:** `main` holds only working, reviewed code. Each person has one branch
  (`allen`, `teammate-a`, `teammate-b`) and opens a PR into `main` when something works.
  Merge main into your branch regularly (`git pull origin main`).
- **Ownership:** only Allen edits `src/` and `predict.py`. Teammates import from `src/` and
  never modify it; if you need a change, ask Allen and it lands as a small PR.
  Each person only edits their own notebooks, so there are no notebook merge conflicts.
- **Outputs:** each experiment writes its own `results/<experiment>.csv` and its own figures.
- **Kaggle:** only Teammate B submits (5/day limit). Others send B their `results/preds_*.csv`
  (or a PR with them).
- **Rules:** CPU-runnable; GPU notebook is optional. No commercial LLM APIs (OpenAI,
  Anthropic, ...) to classify examples or generate predictions. Self-hosted open-source models
  (DeBERTa, GPT-2) are fine.

| Date | Milestone |
|---|---|
| Sat Nov 7 | scaffold, data pipeline, baseline merged; everyone sets up env + data |
| Sun–Mon Nov 8–9 | experiments in parallel on personal branches |
| Tue Nov 10 | ensemble, final submissions, report written |
| Wed Nov 11 | buffer only |

## AI tool usage

> TODO: fill in before submission (keep consistent with `report/sections/ai_disclosure.tex`).
> List each tool, what it was used for, and by whom. Confirm that no commercial LLM API was used
> to classify examples or generate predictions.
