#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/opt/anaconda3/bin/python}"
DATASET_ARCHIVES="${DATASET_ARCHIVES:-/home/amax/datasets/go-dataset}"
DATA_DIR="${DATA_DIR:-data/prepared}"
RUN_DIR="${RUN_DIR:-runs/go_rank_v2}"
RESUME_FROM="${RESUME_FROM:-}"
PER_CLASS="${PER_CLASS:-0}"
INITIAL_EPOCHS="${INITIAL_EPOCHS:-12}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-512}"
WORKERS="${WORKERS:-12}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
RESUME_LR="${RESUME_LR:-0.0002}"
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-0.75}"
SCHEDULER="${SCHEDULER:-cosine}"
LOSS="${LOSS:-ordinal-soft}"
ORDINAL_SIGMA="${ORDINAL_SIGMA:-1.0}"
ORDINAL_LOSS_WEIGHT="${ORDINAL_LOSS_WEIGHT:-0.3}"
DISTANCE_LOSS_WEIGHT="${DISTANCE_LOSS_WEIGHT:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-composite}"
PATIENCE="${PATIENCE:-4}"
MIN_DELTA="${MIN_DELTA:-0.0005}"
CHECKPOINT_EVERY_BATCHES="${CHECKPOINT_EVERY_BATCHES:-500}"
TEST_SYMMETRIES="${TEST_SYMMETRIES:-8}"
TEST_CUTOFFS="${TEST_CUTOFFS:-200 250 300}"
DECODE="${DECODE:-expected}"
MAX_SECONDS="${MAX_SECONDS:-}"
SHUFFLE_DATA="${SHUFFLE_DATA:-1}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"

mkdir -p "$RUN_DIR" logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

LOCK_FILE="$RUN_DIR/.train.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another training process is already using $RUN_DIR"
  exit 0
fi

if [[ "$FORCE_PREPARE" == "1" || ! -f "$DATA_DIR/manifest.json" || ! -f "$DATA_DIR/train_moves.npy" ]]; then
  "$PYTHON" -m go_rank_classifier.prepare \
    --archive-root "$DATASET_ARCHIVES" \
    --output "$DATA_DIR" \
    --per-class "$PER_CLASS" \
    --workers 4
fi

if [[ "$SHUFFLE_DATA" == "1" && ! -f "$DATA_DIR/train.shuffled" ]]; then
  "$PYTHON" -m go_rank_classifier.shuffle_data \
    --data-dir "$DATA_DIR" \
    --splits train val test
fi

RESUME_ARGS=()
if [[ -f "$RUN_DIR/last.pt" ]]; then
  RESUME_ARGS=(--resume "$RUN_DIR/last.pt")
  RUN_EPOCHS="$EPOCHS"
  ACTIVE_LR="$RESUME_LR"
elif [[ -n "$RESUME_FROM" && -f "$RESUME_FROM" ]]; then
  RESUME_ARGS=(--resume "$RESUME_FROM")
  RUN_EPOCHS="$EPOCHS"
  ACTIVE_LR="$RESUME_LR"
else
  RUN_EPOCHS="$INITIAL_EPOCHS"
  ACTIVE_LR="$LEARNING_RATE"
fi

TIME_ARGS=()
if [[ -n "$MAX_SECONDS" ]]; then
  TIME_ARGS=(--max-seconds "$MAX_SECONDS")
fi
read -r -a TEST_CUTOFF_ARGS <<< "$TEST_CUTOFFS"

"$PYTHON" -m go_rank_classifier.train \
  --data "$DATA_DIR" \
  --output "$RUN_DIR" \
  --epochs "$RUN_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --lr "$ACTIVE_LR" \
  --class-weight-power "$CLASS_WEIGHT_POWER" \
  --scheduler "$SCHEDULER" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --loss "$LOSS" \
  --ordinal-sigma "$ORDINAL_SIGMA" \
  --ordinal-loss-weight "$ORDINAL_LOSS_WEIGHT" \
  --distance-loss-weight "$DISTANCE_LOSS_WEIGHT" \
  --early-stop-metric "$EARLY_STOP_METRIC" \
  --patience "$PATIENCE" \
  --min-delta "$MIN_DELTA" \
  --checkpoint-every-batches "$CHECKPOINT_EVERY_BATCHES" \
  --test-symmetries "$TEST_SYMMETRIES" \
  --test-cutoffs "${TEST_CUTOFF_ARGS[@]}" \
  --decode "$DECODE" \
  "${RESUME_ARGS[@]}" \
  "${TIME_ARGS[@]}"

ln -sfn "$(basename "$LOG_FILE")" logs/latest.log
