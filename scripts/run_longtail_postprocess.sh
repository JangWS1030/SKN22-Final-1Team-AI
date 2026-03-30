#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-/workspace/hair_swap_generation}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/datasets}"
LONGTAIL_DATA_ROOT="${LONGTAIL_DATA_ROOT:-$DATASET_ROOT/longtail_training}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-$WORK_ROOT/output/training}"
BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-$WORK_ROOT/output/benchmarks}"
DOC_OUTPUT_ROOT="${DOC_OUTPUT_ROOT:-$WORK_ROOT/output/docs_longtail}"
STATE_ROOT="${STATE_ROOT:-$WORK_ROOT/state_longtail}"
LOG_ROOT="${LOG_ROOT:-$WORK_ROOT/logs_longtail}"
SESSION_NAME="${SESSION_NAME:-hairgen_longtail_train}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-180}"
RUN_STAGE4_EXPECTED="${RUN_STAGE4_EXPECTED:-1}"
POSTPROCESS_STATE_PATH="${POSTPROCESS_STATE_PATH:-$STATE_ROOT/postprocess_longtail.json}"

mkdir -p "$BENCHMARK_OUTPUT_ROOT" "$DOC_OUTPUT_ROOT" "$STATE_ROOT" "$LOG_ROOT"
export PYTHONUNBUFFERED=1

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_with_log() {
  local log_name="$1"
  shift
  local log_path="$LOG_ROOT/$log_name"
  log "Logging to $log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

resolve_lora_dir() {
  local root="$1"
  if [[ -d "$root/best" ]]; then
    echo "$root/best"
    return
  fi
  if [[ -d "$root/final" ]]; then
    echo "$root/final"
    return
  fi
  return 1
}

session_pid() {
  local meta="$STATE_ROOT/${SESSION_NAME}.json"
  if [[ ! -f "$meta" ]]; then
    return 1
  fi
  python - "$meta" <<'PY'
import json, sys
from pathlib import Path
meta = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(meta.get("pid", ""))
PY
}

process_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

training_process_alive() {
  pgrep -f "scripts/train_hair_lora.py" >/dev/null 2>&1
}

wait_for_training_completion() {
  local stage3_root="$TRAIN_OUTPUT_ROOT/generation_lora_stage3_longtail_4090"
  local stage4_root="$TRAIN_OUTPUT_ROOT/generation_lora_stage4_garment_reveal_4090"
  local pid=""

  while true; do
    if [[ "$RUN_STAGE4_EXPECTED" == "1" ]]; then
      if [[ -f "$stage4_root/training_summary.json" ]] && resolve_lora_dir "$stage4_root" >/dev/null 2>&1; then
        log "Detected completed stage4 output."
        return 0
      fi
    else
      if [[ -f "$stage3_root/training_summary.json" ]] && resolve_lora_dir "$stage3_root" >/dev/null 2>&1; then
        log "Detected completed stage3 output."
        return 0
      fi
    fi

    pid="$(session_pid || true)"
    if [[ -n "$pid" ]] && process_alive "$pid"; then
      log "Training still running (pid=$pid). Waiting ${WAIT_INTERVAL_SECONDS}s."
      sleep "$WAIT_INTERVAL_SECONDS"
      continue
    fi
    if training_process_alive; then
      log "Detected active train_hair_lora.py process. Waiting ${WAIT_INTERVAL_SECONDS}s."
      sleep "$WAIT_INTERVAL_SECONDS"
      continue
    fi

    if [[ -f "$stage4_root/training_summary.json" ]] && resolve_lora_dir "$stage4_root" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -f "$stage3_root/training_summary.json" ]] && resolve_lora_dir "$stage3_root" >/dev/null 2>&1; then
      return 0
    fi

    log "No completed summary yet and no active training process detected. Waiting ${WAIT_INTERVAL_SECONDS}s for a restart or late summary."
    sleep "$WAIT_INTERVAL_SECONDS"
  done
}

benchmark_done() {
  local report_dir="$1"
  [[ -f "$report_dir/summary.json" ]] && [[ -f "$report_dir/README.md" ]]
}

STAGE3_ROOT="$TRAIN_OUTPUT_ROOT/generation_lora_stage3_longtail_4090"
STAGE4_ROOT="$TRAIN_OUTPUT_ROOT/generation_lora_stage4_garment_reveal_4090"
SMOKE_OUTPUT="$BENCHMARK_OUTPUT_ROOT/longtail_smoke"
RARE_OUTPUT="$BENCHMARK_OUTPUT_ROOT/rare_style_showcase"
GARMENT_OUTPUT="$BENCHMARK_OUTPUT_ROOT/garment_reveal_showcase"

wait_for_training_completion

run_with_log "postprocess_bootstrap.log" python -m pip install python-docx

PROMOTED_LORA="$(resolve_lora_dir "$STAGE4_ROOT" || true)"
if [[ -z "$PROMOTED_LORA" ]]; then
  PROMOTED_LORA="$(resolve_lora_dir "$STAGE3_ROOT")"
fi

if ! benchmark_done "$SMOKE_OUTPUT/report"; then
  run_with_log "postprocess_smoke.log" python scripts/run_generation_benchmark.py \
    --benchmark-manifest "$LONGTAIL_DATA_ROOT/benchmarks/smoke_longtail_eval_quick.jsonl" \
    --output-dir "$SMOKE_OUTPUT" \
    --lora-path "$PROMOTED_LORA" \
    --return-intermediates
  run_with_log "postprocess_smoke.log" python scripts/evaluate_generation_benchmark.py \
    --predictions-manifest "$SMOKE_OUTPUT/predictions.jsonl" \
    --report-dir "$SMOKE_OUTPUT/report"
fi

if ! benchmark_done "$RARE_OUTPUT/report"; then
  run_with_log "postprocess_rare.log" python scripts/run_generation_benchmark.py \
    --benchmark-manifest "$LONGTAIL_DATA_ROOT/benchmarks/rare_style_showcase_eval.jsonl" \
    --output-dir "$RARE_OUTPUT" \
    --lora-path "$PROMOTED_LORA"
  run_with_log "postprocess_rare.log" python scripts/evaluate_generation_benchmark.py \
    --predictions-manifest "$RARE_OUTPUT/predictions.jsonl" \
    --report-dir "$RARE_OUTPUT/report"
fi

if ! benchmark_done "$GARMENT_OUTPUT/report"; then
  run_with_log "postprocess_garment.log" python scripts/run_generation_benchmark.py \
    --benchmark-manifest "$LONGTAIL_DATA_ROOT/benchmarks/garment_reveal_showcase_eval.jsonl" \
    --output-dir "$GARMENT_OUTPUT" \
    --lora-path "$PROMOTED_LORA"
  run_with_log "postprocess_garment.log" python scripts/evaluate_generation_benchmark.py \
    --predictions-manifest "$GARMENT_OUTPUT/predictions.jsonl" \
    --report-dir "$GARMENT_OUTPUT/report"
fi

run_with_log "postprocess_docs.log" python scripts/build_longtail_result_reports.py \
  --stage3-summary "$STAGE3_ROOT/training_summary.json" \
  --stage4-summary "$STAGE4_ROOT/training_summary.json" \
  --smoke-summary "$SMOKE_OUTPUT/report/summary.json" \
  --rare-summary "$RARE_OUTPUT/report/summary.json" \
  --garment-summary "$GARMENT_OUTPUT/report/summary.json" \
  --smoke-predictions "$SMOKE_OUTPUT/predictions.jsonl" \
  --rare-predictions "$RARE_OUTPUT/predictions.jsonl" \
  --garment-predictions "$GARMENT_OUTPUT/predictions.jsonl" \
  --work-root "$WORK_ROOT" \
  --output-dir "$DOC_OUTPUT_ROOT"

cat > "$POSTPROCESS_STATE_PATH" <<EOF
{
  "promoted_lora": "$PROMOTED_LORA",
  "smoke_output": "$SMOKE_OUTPUT",
  "rare_output": "$RARE_OUTPUT",
  "garment_output": "$GARMENT_OUTPUT",
  "doc_output_dir": "$DOC_OUTPUT_ROOT"
}
EOF

log "Longtail postprocess complete"
