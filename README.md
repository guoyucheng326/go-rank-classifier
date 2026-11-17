# Go Rank Classifier

Classify a Go game record (SGF) into one of 28 Fox server rank IDs using only
the moves in the game.

The model is trained on
[featurecat/go-dataset](https://github.com/featurecat/go-dataset), which contains
games from ranks `Pro`, `9d` through `1d`, and `1k` through `18k`.

## Data policy

- Only SGF games whose main line contains 100 through 350 moves are retained.
- Every retained game is truncated to its first 300 moves.
- Games below 100 moves or above 350 moves are discarded.
- SGF metadata such as player names and stated rank is never used as a feature.
- Training applies one of the eight board symmetries uniformly at random:
  identity, 90/180/270 degree rotations, and their mirror images.
- Training also samples a random cutoff from move 100 through the game's stored
  end (at most move 300). Validation and test use `min(game_length, 300)`.

## Rank IDs

The output ID is based on the official Fox rank ordering:

| ID | Rank | ID | Rank | ID | Rank | ID | Rank |
|---:|---|---:|---|---:|---|---:|---|
| 0 | Pro | 1 | 9d | 2 | 8d | 3 | 7d |
| 4 | 6d | 5 | 5d | 6 | 4d | 7 | 3d |
| 8 | 2d | 9 | 1d | 10 | 1k | 11 | 2k |
| 12 | 3k | 13 | 4k | 14 | 5k | 15 | 6k |
| 16 | 7k | 17 | 8k | 18 | 9k | 19 | 10k |
| 20 | 11k | 21 | 12k | 22 | 13k | 23 | 14k |
| 24 | 15k | 25 | 16k | 26 | 17k | 27 | 18k |

## Installation

Python 3.10+ and PyTorch are required. The preprocessing script also requires
the `7z` command.

```bash
python -m pip install -r requirements.txt
```

## Prepare the training data

Point `--archive-root` at the downloaded `go-dataset` repository. Archives are
streamed through `7z`, so the expander does not create tens of millions of
intermediate SGF files.

```bash
python -m go_rank_classifier.prepare \
  --archive-root /home/amax/datasets/go-dataset \
  --output data/prepared \
  --per-class 0 \
  --workers 4
```

`--per-class 0` keeps every game that passes the main-line length and board-size
filters. Small ranks such as `Pro`, `8d`, and `9d` therefore use all of their
available records, while larger ranks contribute more data. Set a positive
cap only when a smaller experiment is desired.

Prepared moves are stored as memory-mapped `.npy` arrays. This avoids loading
the full corpus into RAM during a daily training run.

The output contains:

```text
data/prepared/
  train_moves.npy
  train_lengths.npy
  train_labels.npy
  val_*.npy
  test_*.npy
  train.shuffled
  val.shuffled
  test.shuffled
  labels.json
  manifest.json
```

Each game is stored as a compact sequence of board coordinates. CNN board
features are built on demand, which keeps the prepared dataset relatively small.

Prepare converts the grouped cache into a globally shuffled, memory-mapped
cache. This allows training to read batches almost sequentially instead of
causing one random disk read per game. To retrofit an existing cache without
re-reading the source archives:

```bash
python -m go_rank_classifier.shuffle_data --data-dir data/prepared
```

## Train

```bash
python -m go_rank_classifier.train \
  --data data/prepared \
  --output runs/go_rank \
  --epochs 20 \
  --batch-size 512 \
  --workers 12
```

The best checkpoint is saved as `runs/go_rank/best.pt`. `history.csv` and
`metrics.json` contain the training and evaluation results.

After a normal run, the best checkpoint is evaluated automatically on the test
split. `test_metrics.json`, `test_metrics_per_class.csv`, and
`test_metrics_confusion.npy` contain overall, per-rank, and confusion-matrix
results.

The default `--class-weight-power 0.5` partially compensates for uneven class
counts without throwing away games from large ranks. Set it to `0` for a
natural-frequency loss, or `1` for fully inverse-frequency weighting.

Resume from the latest checkpoint after an interruption or for a daily
training run:

```bash
python -m go_rank_classifier.train \
  --data data/prepared \
  --output runs/go_rank \
  --epochs 5 \
  --resume runs/go_rank/last.pt
```

### Daily training command

`scripts/train_daily.sh` prepares the cache when it is missing, resumes from
`last.pt` when available, and writes a timestamped log. It uses a fixed
learning rate by default so a one-epoch daily schedule does not collapse to the
cosine scheduler's minimum learning rate.

```bash
chmod +x scripts/train_daily.sh
scripts/train_daily.sh
```

Useful environment variables:

| Variable | Default |
|---|---|
| `PYTHON` | `/opt/anaconda3/bin/python` |
| `DATASET_ARCHIVES` | `/home/amax/datasets/go-dataset` |
| `DATA_DIR` | `data/prepared` |
| `RUN_DIR` | `runs/go_rank` |
| `PER_CLASS` | `0` (all accepted games) |
| `INITIAL_EPOCHS` | `5` |
| `EPOCHS` | `1` |
| `BATCH_SIZE` | `512` |
| `WORKERS` | `12` |
| `LEARNING_RATE` | `0.001` for a new run |
| `RESUME_LR` | `0.0003` for resumed runs |
| `CLASS_WEIGHT_POWER` | `0.5` |
| `SCHEDULER` | `fixed` |
| `MAX_SECONDS` | unset |
| `SHUFFLE_DATA` | `1` |
| `FORCE_PREPARE` | `0` |

On the first run the script trains `INITIAL_EPOCHS`. Subsequent daily runs
resume from `last.pt` and train `EPOCHS` more epochs. Example crontab entry:

```cron
15 2 * * * cd /path/to/go_ranker && scripts/train_daily.sh
```

If `MAX_SECONDS` or `--max-seconds` stops an epoch partway through, `last.pt`
stores the exact batch offset. The next run resumes that epoch and its sampler
order rather than skipping the unprocessed samples.

The daily script uses a lock file to prevent overlapping training jobs. Do not
run `shuffle_data` while another training process is using the cache.

`SIGTERM` and `SIGINT` are handled after the current batch so the model,
optimizer, and batch progress are saved before exit.

For a quick smoke test:

```bash
python -m go_rank_classifier.train \
  --data data/prepared \
  --output runs/smoke \
  --epochs 1 \
  --limit-train 2048 \
  --limit-val 512
```

## Predict

Print only the numeric rank ID:

```bash
python -m go_rank_classifier.predict \
  --checkpoint runs/go_rank/best.pt \
  game.sgf
```

Print JSON with the rank name and class probabilities:

```bash
python -m go_rank_classifier.predict \
  --checkpoint runs/go_rank/best.pt \
  --json \
  game.sgf
```

Python API:

```python
from go_rank_classifier.predict import RankPredictor

predictor = RankPredictor("runs/go_rank/best.pt")
rank_id, rank_name = predictor.predict_file("game.sgf")
print(rank_id, rank_name)
```

## Evaluate and audit

Evaluate a checkpoint and write overall metrics, per-rank metrics, and the
confusion matrix:

```bash
python -m go_rank_classifier.evaluate \
  --checkpoint runs/go_rank/best.pt \
  --data data/prepared \
  --split test
```

Audit exact duplicate main-line move sequences across train, validation, and
test:

```bash
python -m go_rank_classifier.audit \
  --data-dir data/prepared \
  --output reports/duplicate_audit.json
```

The audit reports duplicate rows within each split, cross-split overlap, and
duplicate move sequences assigned to different ranks.

## Input representation

For each selected cutoff, the model receives a `22 x 19 x 19` tensor:

- Black/white stones accumulated over the whole game (`2` channels).
- Black/white stones played in the segment from move 100 to the cutoff (`2`).
- Black/white stones in the last 32 moves (`2`).
- One-hot black/white positions for each of the last eight moves (`16`).

The SGF main line is parsed directly. Variations are ignored, pass moves are
counted but do not add a board plane, and non-19x19 games are rejected.

## Tests

```bash
python -m unittest discover -s tests -v
```

