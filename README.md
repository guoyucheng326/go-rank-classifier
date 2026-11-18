# Go Rank Classifier

Train a dual-player Go rank classifier from SGF records. One SGF contains
separate black and white rank labels, so the model has two 28-class heads. To
return one rank ID for a game, select the player being evaluated.

The model predicts:

- black and white integer rank IDs
- black and white continuous strength estimates
- per-class probability distributions and confidence

`Pro` is mapped to continuous `10d`. It remains one class, so the task still has
28 classes:

| ID | Rank | Strength |
|---:|---|---:|
| 0 | Pro | 10 |
| 1 | 9d | 9 |
| 2 | 8d | 8 |
| ... | ... | ... |
| 9 | 1d | 1 |
| 10 | 1k | 0 |
| 11 | 2k | -1 |
| ... | ... | ... |
| 27 | 18k | -17 |

## Data policy

- Keep games whose SGF main line has 100 through 350 moves.
- Truncate model input to the first 300 moves.
- Require valid and recognizable SGF `BR` and `WR` values.
- Require `HA[0]` and no `AB`/`AW` setup stones.
- Reject non-19x19 games.
- Deduplicate exact main-line move sequences before splitting.
- Split by game after deduplication.

`BR` and `WR` are used only as labels. They are never model inputs.

## Prepare SGF data

```bash
python -m go_rank_classifier.prepare \
  --archive-root /home/amax/datasets/go-dataset \
  --output data/prepared_v2 \
  --workers 4
```

The output contains memory-mapped dual-label data:

```text
train_moves.npy
train_lengths.npy
train_black_labels.npy
train_white_labels.npy
train_hashes.npy
...same files for val and test...
manifest.json
```

The manifest records the data fingerprint and separate black/white class
counts.

## Build KataGo features

Set the paths to the exact Transformer KataGo binary, model, and analysis
configuration used on the RTX 5080:

```bash
export KATAGO_BIN=/path/to/katago
export KATAGO_MODEL=/path/to/transformer-model.bin.gz
export KATAGO_CONFIG=/path/to/analysis.cfg

DATA_DIR=data/prepared_v2 \
OUTPUT_DIR=data/katago_v1 \
MAX_VISITS=500 \
POSITIONS_PER_PLAYER=0 \
GAMES_PER_CLASS=0 \
JOB_SIZE=500 \
scripts/build_katago_features.sh
```

The current RTX 5080 environment defaults to:

```text
KataGo v1.17.1 (git 5246793f)
binary: /home/amax/guojianren/.tools/katago-opencl/katago
model:  /home/amax/guojianren/.tools/katago-models/current-transformer.bin.gz
config: /home/amax/guojianren/.tools/katago-opencl/analysis_config.cfg
```

The generated manifest records SHA256 hashes for all three files.

`POSITIONS_PER_PLAYER=0` analyzes every move for both players.
`GAMES_PER_CLASS=0` analyzes every game.

The fixed analysis defaults are:

```text
rules       = chinese
komi        = 7.5
maxVisits   = 500
ownership   = disabled
PV length   = 15
perspective = side to move
```

The executable, model, and configuration SHA256 values are recorded in each
feature manifest.

For a pilot, use:

```bash
GAMES_PER_CLASS=500 POSITIONS_PER_PLAYER=8 scripts/build_katago_features.sh
```

The generator saves:

- compact per-move metrics in sharded `.npz` files
- separate black/white aggregated feature matrices
- a mask identifying games with complete requested analysis
- feature names and fixed engine metadata

The aggregated features include:

- Top-1/Top-3/Top-5 recommendation agreement
- played-move candidate rank
- visit-weighted recommendation agreement
- winrate loss and score-lead loss
- blunder and severe-blunder rates
- policy entropy
- layout, middle-game, and late-game summaries

Training never invokes KataGo. It reads the cached feature matrices.

## Train

```bash
scripts/train_daily.sh
```

The model uses:

- shared 96/192/384-channel CNN
- independent black and white 28-class heads
- optional 128-dimensional KataGo feature branch
- distance-aware ordinal soft labels
- class weighting `power=0.75`
- warmup plus cosine learning-rate decay
- Macro-F1/adjacent/composite early stopping
- midepoch checkpoint recovery

Default paths:

| Variable | Default |
|---|---|
| `DATA_DIR` | `data/prepared_v2` |
| `KATAGO_DATA` | `data/katago_v1` |
| `RUN_DIR` | `runs/go_rank_v2` |
| `MAX_VISITS` | `500` |
| `CLASS_WEIGHT_POWER` | `0.75` |
| `INITIAL_EPOCHS` | `12` |
| `EPOCHS` | `12` |
| `TEST_SYMMETRIES` | `8` |
| `TEST_CUTOFFS` | `200 250 300` |
| `DECODE` | `expected` |

To resume an existing model into a new experiment directory:

```bash
RUN_DIR=runs/go_rank_v2 \
RESUME_FROM=runs/go_rank/best.pt \
scripts/train_daily.sh
```

Because the model architecture and label definition changed, old 2.08M
single-head checkpoints are not directly compatible with the dual-head model.

## Evaluation

```bash
python -m go_rank_classifier.evaluate \
  --checkpoint runs/go_rank_v2/best.pt \
  --data data/prepared_v2 \
  --katago-data data/katago_v1 \
  --split test \
  --symmetries 8 \
  --cutoffs 200 250 300 \
  --decode expected
```

Reports include:

- black/white Top-1 and Top-3
- Macro-F1
- expected-strength MAE/RMSE
- `+-0.5` and `+-1.0` rank accuracy
- per-class precision/recall/F1
- confusion matrices

## Predict one player

```bash
python -m go_rank_classifier.predict \
  --checkpoint runs/go_rank_v2/best.pt \
  --player white \
  --katago-black black_features.npy \
  --katago-white white_features.npy \
  game.sgf
```

The two feature files each contain one row of features for the corresponding
player. `--player` is required because one SGF has two different rank targets.

## Duplicate audit

```bash
python -m go_rank_classifier.audit \
  --data-dir data/prepared_v2 \
  --output reports/duplicate_audit.json
```

## Tests

```bash
python -m unittest discover -s tests -v
```
