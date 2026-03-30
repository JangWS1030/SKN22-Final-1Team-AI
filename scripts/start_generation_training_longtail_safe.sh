#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-/workspace/hair_swap_generation}"
STATE_ROOT="${STATE_ROOT:-$WORK_ROOT/state_longtail}"
LOG_ROOT="${LOG_ROOT:-$WORK_ROOT/logs_longtail}"
SESSION_NAME="${SESSION_NAME:-hairgen_longtail_train}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"
mkdir -p "$STATE_ROOT" "$LOG_ROOT"

RUN_SCRIPT="$REPO_ROOT/scripts/run_generation_training_longtail.sh"
WRAPPER_SCRIPT="$STATE_ROOT/launch_generation_training_longtail.sh"
STDOUT_LOG="$LOG_ROOT/${SESSION_NAME}.stdout.log"
SESSION_META="$STATE_ROOT/${SESSION_NAME}.json"

cat > "$WRAPPER_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
bash "$RUN_SCRIPT" >> "$STDOUT_LOG" 2>&1
EOF
chmod +x "$WRAPPER_SCRIPT"

if [[ "$LAUNCH_MODE" == "tmux" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[safe-launch] tmux is not installed. Falling back to nohup mode." >&2
    LAUNCH_MODE="nohup"
  fi
fi

if [[ "$LAUNCH_MODE" == "tmux" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[safe-launch] tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION_NAME" "bash '$WRAPPER_SCRIPT'"
  PID="$(tmux list-panes -t "$SESSION_NAME" -F '#{pane_pid}' | head -n 1)"
  MODE="tmux"
else
  nohup bash "$WRAPPER_SCRIPT" >/dev/null 2>&1 &
  PID="$!"
  MODE="nohup"
fi

cat > "$SESSION_META" <<EOF
{
  "session_name": "$SESSION_NAME",
  "launch_mode": "$MODE",
  "pid": "$PID",
  "stdout_log": "$STDOUT_LOG",
  "wrapper_script": "$WRAPPER_SCRIPT",
  "repo_root": "$REPO_ROOT",
  "work_root": "$WORK_ROOT"
}
EOF

echo "[safe-launch] started"
echo "[safe-launch] mode: $MODE"
echo "[safe-launch] session: $SESSION_NAME"
echo "[safe-launch] pid: $PID"
echo "[safe-launch] log: $STDOUT_LOG"
