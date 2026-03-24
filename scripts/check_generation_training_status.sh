#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/runpod-volume/hair_swap_generation}"
STATE_ROOT="${STATE_ROOT:-$WORK_ROOT/state}"
SESSION_NAME="${SESSION_NAME:-hairgen_train}"
SESSION_META="$STATE_ROOT/${SESSION_NAME}.json"
export SESSION_META

if [[ ! -f "$SESSION_META" ]]; then
  echo "[status] no session metadata found: $SESSION_META"
  exit 1
fi

echo "[status] metadata"
cat "$SESSION_META"
echo

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[status] tmux session is alive: $SESSION_NAME"
  echo "[status] attach with: tmux attach -t $SESSION_NAME"
else
  echo "[status] tmux session not found (may be using nohup or may have finished)"
fi

STDOUT_LOG="$(python - <<'PY'
import json, os
from pathlib import Path
meta = json.loads(Path(os.environ["SESSION_META"]).read_text(encoding="utf-8"))
print(meta["stdout_log"])
PY
)"

if [[ -f "$STDOUT_LOG" ]]; then
  echo "[status] last log lines"
  tail -n 40 "$STDOUT_LOG"
fi
