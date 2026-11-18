#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/opt/anaconda3/bin/python}"
DATA_DIR="${DATA_DIR:-data/prepared_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-data/katago_v1}"
KATAGO_BIN="${KATAGO_BIN:-/home/amax/guojianren/.tools/katago-opencl/katago}"
KATAGO_MODEL="${KATAGO_MODEL:-/home/amax/guojianren/.tools/katago-models/current-transformer.bin.gz}"
KATAGO_CONFIG="${KATAGO_CONFIG:-/home/amax/guojianren/.tools/katago-opencl/analysis_config.cfg}"
MAX_VISITS="${MAX_VISITS:-500}"
POSITIONS_PER_PLAYER="${POSITIONS_PER_PLAYER:-0}"
GAMES_PER_CLASS="${GAMES_PER_CLASS:-0}"
JOB_SIZE="${JOB_SIZE:-500}"
SPLITS="${SPLITS:-train val test}"
SEED="${SEED:-2029}"

mkdir -p "$OUTPUT_DIR" logs
read -r -a SPLIT_ARGS <<< "$SPLITS"

for split in "${SPLIT_ARGS[@]}"; do
  "$PYTHON" -m go_rank_classifier.katago_features \
    --data-dir "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --split "$split" \
    --katago-bin "$KATAGO_BIN" \
    --model "$KATAGO_MODEL" \
    --config "$KATAGO_CONFIG" \
    --rules chinese \
    --komi 7.5 \
    --max-visits "$MAX_VISITS" \
    --positions-per-player "$POSITIONS_PER_PLAYER" \
    --games-per-class "$GAMES_PER_CLASS" \
    --job-size "$JOB_SIZE" \
    --seed "$SEED" \
    2>&1 | tee "logs/katago_${split}.log"
done
