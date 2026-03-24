#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-/runpod-volume/hair_swap_generation}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/dataset_build/processed/celeba_dialog_hq_generation}"
HF_HOME="${HF_HOME:-$WORK_ROOT/hf-cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_ROOT/pip-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/output/training}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$DATA_ROOT/benchmarks}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$REPO_ROOT/configs/accelerate/single_gpu_4090.yaml}"
STAGE1_CONFIG="${STAGE1_CONFIG:-$REPO_ROOT/configs/generation_train/stage1_budget_4090.json}"
STAGE2_CONFIG="${STAGE2_CONFIG:-$REPO_ROOT/configs/generation_train/stage2_highconf_4090.json}"
RUN_SMOKE_EVAL="${RUN_SMOKE_EVAL:-1}"
RUN_FULL_RECON_EVAL="${RUN_FULL_RECON_EVAL:-0}"
AUTO_RESUME="${AUTO_RESUME:-1}"
SKIP_STAGE1_IF_DONE="${SKIP_STAGE1_IF_DONE:-1}"
SKIP_STAGE2_IF_DONE="${SKIP_STAGE2_IF_DONE:-1}"
STATE_ROOT="${STATE_ROOT:-$WORK_ROOT/state}"
LOG_ROOT="${LOG_ROOT:-$WORK_ROOT/logs}"
LOCK_DIR="$STATE_ROOT/generation_training.lock"
RUN_METADATA_PATH="$STATE_ROOT/current_run.json"

export HF_HOME
export PIP_CACHE_DIR
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$WORK_ROOT" "$HF_HOME" "$PIP_CACHE_DIR" "$OUTPUT_ROOT" "$STATE_ROOT" "$LOG_ROOT"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[train] Another training run appears to be active: $LOCK_DIR" >&2
  exit 1
fi

cleanup() {
  rm -rf "$LOCK_DIR"
}
trap cleanup EXIT

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

write_run_metadata() {
  cat > "$RUN_METADATA_PATH" <<EOF
{
  "repo_root": "$REPO_ROOT",
  "work_root": "$WORK_ROOT",
  "data_root": "$DATA_ROOT",
  "output_root": "$OUTPUT_ROOT",
  "log_root": "$LOG_ROOT",
  "stage1_output": "$OUTPUT_ROOT/generation_lora_stage1_budget_4090",
  "stage2_output": "$OUTPUT_ROOT/generation_lora_stage2_highconf_4090"
}
EOF
}

stage_resume_args() {
  local stage_output="$1"
  if [[ "$AUTO_RESUME" == "1" ]] && compgen -G "$stage_output/checkpoint-*" > /dev/null; then
    echo "--resume-from-checkpoint latest"
  fi
}

stage_is_done() {
  local stage_output="$1"
  [[ -d "$stage_output/best" || -d "$stage_output/final" ]]
}

run_with_log() {
  local log_name="$1"
  shift
  local log_path="$LOG_ROOT/$log_name"
  log "Logging to $log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

write_run_metadata

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[train] DATA_ROOT not found: $DATA_ROOT" >&2
  echo "[train] Put the processed dataset on the mounted volume or override DATA_ROOT." >&2
  exit 1
fi

cd "$REPO_ROOT"

run_with_log "bootstrap.log" python -m pip install --upgrade pip
run_with_log "bootstrap.log" python -m pip install -r requirements-train.txt

run_with_log "bootstrap.log" python scripts/build_generation_benchmarks.py \
  --manifest "$DATA_ROOT/manifests/recon_hair_style_enriched.jsonl" \
  --out-dir "$BENCHMARK_ROOT" \
  --summary-output "$DATA_ROOT/reports/benchmark_summary.json"

STAGE1_OUTPUT="$OUTPUT_ROOT/generation_lora_stage1_budget_4090"
STAGE2_OUTPUT="$OUTPUT_ROOT/generation_lora_stage2_highconf_4090"

if [[ "$SKIP_STAGE1_IF_DONE" == "1" ]] && stage_is_done "$STAGE1_OUTPUT"; then
  log "Stage 1 already completed, skipping."
else
  STAGE1_RESUME_ARGS="$(stage_resume_args "$STAGE1_OUTPUT")"
  log "Stage 1 starting"
  # shellcheck disable=SC2086
  run_with_log "stage1.log" accelerate launch --config_file "$ACCELERATE_CONFIG" scripts/train_hair_lora.py \
    --config "$STAGE1_CONFIG" \
    --manifest "$DATA_ROOT/manifests/style_train_all.jsonl" \
    --validation-manifest "$BENCHMARK_ROOT/recon_eval_stratified.jsonl" \
    --output-dir "$STAGE1_OUTPUT" \
    --cache-dir "$HF_HOME" \
    $STAGE1_RESUME_ARGS
fi

STAGE1_LORA="$STAGE1_OUTPUT/best"
if [[ ! -d "$STAGE1_LORA" ]]; then
  STAGE1_LORA="$STAGE1_OUTPUT/final"
fi

if [[ "$RUN_SMOKE_EVAL" == "1" ]]; then
  log "Stage 1 smoke benchmark"
  run_with_log "stage1_smoke_eval.log" python scripts/run_generation_benchmark.py \
    --benchmark-manifest "$BENCHMARK_ROOT/smoke_eval_quick.jsonl" \
    --output-dir "$OUTPUT_ROOT/eval_stage1_smoke" \
    --lora-path "$STAGE1_LORA"

  run_with_log "stage1_smoke_eval.log" python scripts/evaluate_generation_benchmark.py \
    --predictions-manifest "$OUTPUT_ROOT/eval_stage1_smoke/predictions.jsonl" \
    --report-dir "$OUTPUT_ROOT/eval_stage1_smoke/report"
fi

if [[ "$SKIP_STAGE2_IF_DONE" == "1" ]] && stage_is_done "$STAGE2_OUTPUT"; then
  log "Stage 2 already completed, skipping."
else
  STAGE2_RESUME_ARGS="$(stage_resume_args "$STAGE2_OUTPUT")"
  log "Stage 2 starting"
  # shellcheck disable=SC2086
  run_with_log "stage2.log" accelerate launch --config_file "$ACCELERATE_CONFIG" scripts/train_hair_lora.py \
    --config "$STAGE2_CONFIG" \
    --manifest "$DATA_ROOT/manifests/style_train_highconf.jsonl" \
    --validation-manifest "$BENCHMARK_ROOT/recon_eval_stratified.jsonl" \
    --output-dir "$STAGE2_OUTPUT" \
    --cache-dir "$HF_HOME" \
    --initial-lora-path "$STAGE1_LORA" \
    $STAGE2_RESUME_ARGS
fi

STAGE2_LORA="$STAGE2_OUTPUT/best"
if [[ ! -d "$STAGE2_LORA" ]]; then
  STAGE2_LORA="$STAGE2_OUTPUT/final"
fi

log "Stage 2 smoke benchmark"
run_with_log "stage2_smoke_eval.log" python scripts/run_generation_benchmark.py \
  --benchmark-manifest "$BENCHMARK_ROOT/smoke_eval_quick.jsonl" \
  --output-dir "$OUTPUT_ROOT/eval_stage2_smoke" \
  --lora-path "$STAGE2_LORA"

run_with_log "stage2_smoke_eval.log" python scripts/evaluate_generation_benchmark.py \
  --predictions-manifest "$OUTPUT_ROOT/eval_stage2_smoke/predictions.jsonl" \
  --report-dir "$OUTPUT_ROOT/eval_stage2_smoke/report"

if [[ "$RUN_FULL_RECON_EVAL" == "1" ]]; then
  log "Stage 2 full recon benchmark"
  run_with_log "stage2_recon_eval.log" python scripts/run_generation_benchmark.py \
    --benchmark-manifest "$BENCHMARK_ROOT/recon_eval_stratified.jsonl" \
    --output-dir "$OUTPUT_ROOT/eval_stage2_recon" \
    --lora-path "$STAGE2_LORA"

  run_with_log "stage2_recon_eval.log" python scripts/evaluate_generation_benchmark.py \
    --predictions-manifest "$OUTPUT_ROOT/eval_stage2_recon/predictions.jsonl" \
    --report-dir "$OUTPUT_ROOT/eval_stage2_recon/report"
fi

log "Stage 1/2 training pipeline complete"
