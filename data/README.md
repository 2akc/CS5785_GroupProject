# data/

This folder is gitignored. Download the competition files from Kaggle into it:

```
data/
├── train.csv               # 10,000 rows: id, text, label (0 = human, 1 = AI)
├── unlabeled.csv           # 3,000 rows: id, text
├── test.csv                # 3,000 rows: id, text
├── train_starter_2000.csv  # subset of train.csv (used by QUICK mode)
└── sample_submission.csv   # id, label
```

Either download them from the competition's **Data** tab in the browser, or use the Kaggle CLI
(after joining the competition and accepting its rules):

```bash
kaggle competitions download -c <competition-slug> -p data/
unzip data/*.zip -d data/ && rm data/*.zip
```

`data/split_ids.json` is created automatically the first time `src.data.get_split()` runs.
It is deterministic (seed 42), so every teammate gets the identical split.

To read data from somewhere else (e.g. `/kaggle/input/...` on a Kaggle GPU notebook),
set the environment variable `AIHUMAN_DATA_DIR=/path/to/folder`.
