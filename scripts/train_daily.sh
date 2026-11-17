#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/opt/anaconda3/bin/python}"
DATASET_ARCHIVES="${DATASET_ARCHIVES:-/home/amax/datasets/go-dataset}"
DATA_DIR="${DATA_DIR:-data/prepared}"
RUN_DIR="${RUN_DIR:-runs/go_rank}"
PER_CLASS="${PER_CLASS:-0}"
INITIAL_EPOCHS="${INITIAL_EPOCHS:-5}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
WORKERS="${WORKERS:-12}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
RESUME_LR="${RESUME_LR:-0.0003}"
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-0.5}"
SCHEDULER="${SCHEDULER:-fixed}"
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
else
  RUN_EPOCHS="$INITIAL_EPOCHS"
  ACTIVE_LR="$LEARNING_RATE"
fi

TIME_ARGS=()
if [[ -n "$MAX_SECONDS" ]]; then
  TIME_ARGS=(--max-seconds "$MAX_SECONDS")
fi

"$PYTHON" -m go_rank_classifier.train \
  --data "$DATA_DIR" \
  --output "$RUN_DIR" \
  --epochs "$RUN_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --lr "$ACTIVE_LR" \
  --class-weight-power "$CLASS_WEIGHT_POWER" \
  --scheduler "$SCHEDULER" \
  "${RESUME_ARGS[@]}" \
  "${TIME_ARGS[@]}"

ln -sfn "$(basename "$LOG_FILE")" logs/latest.log
