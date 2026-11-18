#!/usr/bin/env bash
# Run a time-boxed training session, then publish its logs to the repo.
#
#   MAX_SECONDS=28800 EPOCHS=60 scripts/train_and_publish.sh
#
# Designed to survive a detached shell: nohup/setsid friendly, no TTY needed.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MAX_SECONDS="${MAX_SECONDS:-28800}"
export EPOCHS="${EPOCHS:-60}"
export SHUFFLE_DATA="${SHUFFLE_DATA:-1}"
export RESUME_LR="${RESUME_LR:-0.0003}"
export SCHEDULER="${SCHEDULER:-fixed}"

echo "[train_and_publish] start $(date -Is) max_seconds=$MAX_SECONDS epochs=$EPOCHS"
scripts/train_daily.sh
STATUS=$?
echo "[train_and_publish] training exited with status $STATUS at $(date -Is)"

TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}" scripts/publish_reports.sh
echo "[train_and_publish] done $(date -Is)"
