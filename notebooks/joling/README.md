# notebooks/joling

Jo-Ling's experiments. Goal: a detector that still works on writing domains
that are not in `train.csv`.

Rules followed here:
- Same shared split as the rest of the team (`src.data.get_split()`, 8,000 train / 2,000 val, seed 42).
- Nothing in `src/` is modified; all new code lives in this folder.
