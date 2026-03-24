#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/runpod-volume/hair_swap_generation}"
STATE_ROOT="${STATE_ROOT:-$WORK_ROOT/state}"
SESSION_NAME="${SESSION_NAME:-hairgen_train}"
SESSION_META="$STATE_ROOT/${SESSION_NAME}.json"
export SESSION_META

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
  echo "[stop] tmux session killed: $SESSION_NAME"
fi

if [[ -f "$SESSION_META" ]]; then
  PID="$(python - <<'PY'
import json, os
from pathlib import Path
meta = json.loads(Path(os.environ["SESSION_META"]).read_text(encoding="utf-8"))
print(meta.get("pid", ""))
PY
)"
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" || true
    echo "[stop] pid terminated: $PID"
  fi
fi
