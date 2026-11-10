# notebooks/joling

Jo-Ling's experiments. Goal: a detector that still works on writing domains
that are not in `train.csv`.

Rules followed here:
- Same shared split as the rest of the team (`src.data.get_split()`, 8,000 train / 2,000 val, seed 42).
- Nothing in `src/` is modified; all new code lives in this folder.

| File | What it is |
|---|---|
| `01_domain_yardstick.ipynb` | Scores the team's models on topics they did not train on, and with layout removed. About 20 minutes on a laptop. |
| `domain_eval.py` | `TopicDomains` (unsupervised topic groups) and `domain_stress_test` (the four validation numbers). |
| `plots.py` | Chart helpers. |
| `requirements-extra.txt` | Extra packages for the language-model notebook (`02`, in progress). |

Outputs are written to `results/joling_*.csv` and `figures/joling_*.png`.
