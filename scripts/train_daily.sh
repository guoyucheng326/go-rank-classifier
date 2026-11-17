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
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-0.5}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"

mkdir -p "$RUN_DIR" logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ "$FORCE_PREPARE" == "1" || ! -f "$DATA_DIR/manifest.json" || ! -f "$DATA_DIR/train_moves.npy" ]]; then
  "$PYTHON" -m go_rank_classifier.prepare \
    --archive-root "$DATASET_ARCHIVES" \
    --output "$DATA_DIR" \
    --per-class "$PER_CLASS" \
    --workers 4
fi

RESUME_ARGS=()
if [[ -f "$RUN_DIR/last.pt" ]]; then
  RESUME_ARGS=(--resume "$RUN_DIR/last.pt")
  RUN_EPOCHS="$EPOCHS"
else
  RUN_EPOCHS="$INITIAL_EPOCHS"
fi

"$PYTHON" -m go_rank_classifier.train \
  --data "$DATA_DIR" \
  --output "$RUN_DIR" \
  --epochs "$RUN_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --lr "$LEARNING_RATE" \
  --class-weight-power "$CLASS_WEIGHT_POWER" \
  "${RESUME_ARGS[@]}"

ln -sfn "$(basename "$LOG_FILE")" logs/latest.log
