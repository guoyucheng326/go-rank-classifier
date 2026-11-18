#!/usr/bin/env bash
# Snapshot a finished training run into reports/, write a README summary, commit and push.
#
# Usage:
#   TAG=20261118-162000 scripts/publish_reports.sh
# Env:
#   RUN_DIR      run directory to snapshot (default: runs/go_rank)
#   LOG_FILE     training log to copy (default: logs/latest.log)
#   GITEE_TOKEN  optional push credential, only ever read from the environment
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/opt/anaconda3/bin/python}"
RUN_DIR="${RUN_DIR:-runs/go_rank_v2}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
LOG_FILE="${LOG_FILE:-logs/latest.log}"
BRANCH="${BRANCH:-main}"

mkdir -p reports

if [[ -f "$LOG_FILE" ]]; then
  cp "$LOG_FILE" "reports/train_resume_${TAG}.log"
  echo "report: copied $LOG_FILE"
else
  echo "warning: $LOG_FILE not found, skipping log copy" >&2
fi

copy_artifact() {
  local name="$1" extension="$2"
  if [[ -f "$RUN_DIR/$name.$extension" ]]; then
    cp "$RUN_DIR/$name.$extension" "reports/${name}_${TAG}.${extension}"
    echo "report: copied $RUN_DIR/$name.$extension"
  fi
}

copy_artifact history csv
copy_artifact metrics json
copy_artifact test_metrics json
copy_artifact test_black_metrics json
copy_artifact test_black_metrics_per_class csv
copy_artifact test_black_metrics_confusion npy
copy_artifact test_white_metrics json
copy_artifact test_white_metrics_per_class csv
copy_artifact test_white_metrics_confusion npy

"$PYTHON" - "$TAG" "$RUN_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

tag, run_dir = sys.argv[1], Path(sys.argv[2])
history_path = run_dir / "history.csv"
rows = list(csv.DictReader(history_path.open(encoding="utf-8"))) if history_path.exists() else []

lines = [
    f"# go_ranker 续训结果 ({tag})",
    "",
    "由 `scripts/publish_reports.sh` 在训练结束后自动生成。",
    "",
    "## 训练配置",
    "",
    "| 项 | 值 |",
    "| --- | --- |",
]
args = {}
last_path = run_dir / "last.pt"
if last_path.exists():
    try:
        import torch

        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        args = checkpoint.get("args") or {}
    except Exception as error:  # noqa: BLE001 - reporting must never fail
        lines.append(f"| checkpoint | 读取失败: {error} |")
for key in ("data", "output", "epochs", "batch_size", "workers", "lr", "scheduler",
            "class_weight_power", "label_smoothing", "max_seconds", "seed", "amp"):
    if key in args and args[key] is not None:
        lines.append(f"| {key} | {args[key]} |")

lines += ["", "## 逐轮结果", ""]
if rows:
    def fmt(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
        if number == int(number) and abs(number) < 1_000_000:
            return str(int(number))
        return f"{number:.6f}".rstrip("0").rstrip(".")

    header = [
        "epoch", "train_loss",
        "black_accuracy", "white_accuracy",
        "black_top3_accuracy", "white_top3_accuracy",
        "black_macro_f1", "white_macro_f1",
        "black_within_1", "white_within_1",
        "monitor_score", "lr", "seconds", "partial", "stop_reason",
    ]
    header = [name for name in header if name in rows[0]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(name, "")) for name in header) + " |")
    total = sum(float(row.get("seconds", 0) or 0) for row in rows)
    lines += ["", f"- 本次记录 {len(rows)} 轮，合计 {total:.1f} s（{total / 3600:.2f} h）"]
    best = max(
        rows,
        key=lambda row: (
            float(row.get("black_accuracy", 0) or 0)
            + float(row.get("white_accuracy", 0) or 0)
        )
        / 2,
    )
    lines.append(
        f"- 本次最佳：epoch {best['epoch']}，black_acc {best.get('black_accuracy', 'n/a')}，"
        f"white_acc {best.get('white_accuracy', 'n/a')}"
    )
else:
    lines.append("（没有 history.csv）")

metrics_path = run_dir / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    lines += ["", "## 最佳 checkpoint", "", "```json", json.dumps(metrics, indent=2), "```"]

test_path = run_dir / "test_metrics.json"
if test_path.exists():
    test_metrics = json.loads(test_path.read_text(encoding="utf-8"))
    lines += ["", "## test 集指标", "", "```json", json.dumps(test_metrics, indent=2), "```"]
    lines += ["", "逐类指标见 `test_metrics_per_class_" + tag + ".csv`。"]

lines += ["", "## 继续训练", "", "```bash",
          "EPOCHS=12 MAX_SECONDS=28800 scripts/train_daily.sh",
          "```", ""]
(Path("reports") / f"README_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
print(f"report: wrote reports/README_{tag}.md")
PY

git add -f reports
if git diff --cached --quiet; then
  echo "report: nothing to commit"
  exit 0
fi
git commit -m "Add training logs ${TAG}" >/dev/null
echo "report: committed $(git rev-parse --short HEAD)"

# Never fall back to an interactive credential prompt: under VS Code the
# GIT_ASKPASS helper blocks forever on "Username for 'https://gitee.com'".
export GIT_TERMINAL_PROMPT=0
export GIT_ASKPASS=/bin/true
export GIT_USER="${GIT_USER:-guojianren123}"

if [[ -n "${GITEE_TOKEN:-}" ]]; then
  export GITEE_TOKEN
  helper='!f() { echo username=$GIT_USER; echo password=$GITEE_TOKEN; }; f'
  if timeout 180 git -c "credential.helper=$helper" push origin "$BRANCH"; then
    echo "report: pushed to origin/$BRANCH"
  else
    echo "report: push failed via GITEE_TOKEN" >&2
    exit 1
  fi
elif timeout 120 git push origin "$BRANCH"; then
  echo "report: pushed to origin/$BRANCH"
else
  echo "report: push failed and GITEE_TOKEN is not set" >&2
  exit 1
fi
