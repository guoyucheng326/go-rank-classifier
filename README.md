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

## Prepare a balanced dataset

Point `--archive-root` at the downloaded `go-dataset` repository. Archives are
streamed through `7z`, so the expander does not create tens of millions of
intermediate SGF files.

```bash
python -m go_rank_classifier.prepare \
  --archive-root /home/amax/datasets/go-dataset \
  --output data/prepared \
  --per-class 20000 \
  --workers 8
```

The output contains:

```text
data/prepared/
  train.npz
  val.npz
  test.npz
  labels.json
  manifest.json
```

Each game is stored as a compact sequence of board coordinates. CNN board
features are built on demand, which keeps the prepared dataset relatively small.

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

