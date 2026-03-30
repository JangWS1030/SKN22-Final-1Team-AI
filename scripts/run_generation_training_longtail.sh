#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-/workspace/hair_swap_generation}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/datasets}"
BASE_DATA_ROOT="${BASE_DATA_ROOT:-$DATASET_ROOT/celeba_dialog_hq_generation}"
LONGTAIL_DATA_ROOT="${LONGTAIL_DATA_ROOT:-$DATASET_ROOT/longtail_training}"
HF_HOME="${HF_HOME:-$WORK_ROOT/hf-cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_ROOT/pip-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/output/training}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$REPO_ROOT/configs/accelerate/single_gpu_4090.yaml}"
STAGE3_CONFIG="${STAGE3_CONFIG:-$REPO_ROOT/configs/generation_train/stage3_longtail_4090.json}"
STAGE4_CONFIG="${STAGE4_CONFIG:-$REPO_ROOT/configs/generation_train/stage4_garment_reveal_4090.json}"
RUN_STAGE4="${RUN_STAGE4:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
SKIP_STAGE3_IF_DONE="${SKIP_STAGE3_IF_DONE:-1}"
SKIP_STAGE4_IF_DONE="${SKIP_STAGE4_IF_DONE:-1}"
STATE_ROOT="${STATE_ROOT:-$WORK_ROOT/state_longtail}"
LOG_ROOT="${LOG_ROOT:-$WORK_ROOT/logs_longtail}"
LOCK_DIR="$STATE_ROOT/generation_longtail_training.lock"
RUN_METADATA_PATH="$STATE_ROOT/current_run.json"
INITIAL_LONGTAIL_LORA="${INITIAL_LONGTAIL_LORA:-}"

export HF_HOME
export PIP_CACHE_DIR
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$WORK_ROOT" "$HF_HOME" "$PIP_CACHE_DIR" "$OUTPUT_ROOT" "$STATE_ROOT" "$LOG_ROOT"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[train-longtail] Another run appears active: $LOCK_DIR" >&2
  exit 1
fi

cleanup() {
  rm -rf "$LOCK_DIR"
}
trap cleanup EXIT

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

config_uses_8bit_adam() {
  local config_path="$1"
  python - "$config_path" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("1" if bool(cfg.get("use_8bit_adam")) else "0")
PY
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

resolve_base_lora() {
  if [[ -n "$INITIAL_LONGTAIL_LORA" && -d "$INITIAL_LONGTAIL_LORA" ]]; then
    echo "$INITIAL_LONGTAIL_LORA"
    return
  fi
  if [[ -d "$WORK_ROOT/output/training/generation_lora_stage2_highconf_4090/best" ]]; then
    echo "$WORK_ROOT/output/training/generation_lora_stage2_highconf_4090/best"
    return
  fi
  if [[ -d "$WORK_ROOT/output/training/generation_lora_stage2_highconf_4090/final" ]]; then
    echo "$WORK_ROOT/output/training/generation_lora_stage2_highconf_4090/final"
    return
  fi
  if [[ -d "$REPO_ROOT/pretrained_models/generation_lora_stage2_best" ]]; then
    echo "$REPO_ROOT/pretrained_models/generation_lora_stage2_best"
    return
  fi
  return 1
}

cat > "$RUN_METADATA_PATH" <<EOF
{
  "repo_root": "$REPO_ROOT",
  "work_root": "$WORK_ROOT",
  "dataset_root": "$DATASET_ROOT",
  "base_data_root": "$BASE_DATA_ROOT",
  "longtail_data_root": "$LONGTAIL_DATA_ROOT",
  "output_root": "$OUTPUT_ROOT",
  "log_root": "$LOG_ROOT",
  "stage3_output": "$OUTPUT_ROOT/generation_lora_stage3_longtail_4090",
  "stage4_output": "$OUTPUT_ROOT/generation_lora_stage4_garment_reveal_4090"
}
EOF

if [[ ! -d "$BASE_DATA_ROOT" ]]; then
  echo "[train-longtail] BASE_DATA_ROOT not found: $BASE_DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$LONGTAIL_DATA_ROOT" ]]; then
  echo "[train-longtail] LONGTAIL_DATA_ROOT not found: $LONGTAIL_DATA_ROOT" >&2
  exit 1
fi
if [[ ! -f "$LONGTAIL_DATA_ROOT/manifests/style_train_stage3_longtail_mix.jsonl" ]]; then
  echo "[train-longtail] Missing stage3 manifest under $LONGTAIL_DATA_ROOT/manifests" >&2
  exit 1
fi

BASE_LORA="$(resolve_base_lora || true)"
if [[ -z "$BASE_LORA" ]]; then
  echo "[train-longtail] Could not resolve base stage2 LoRA. Set INITIAL_LONGTAIL_LORA." >&2
  exit 1
fi

cd "$REPO_ROOT"

run_with_log "bootstrap_longtail.log" python -m pip install --upgrade pip
run_with_log "bootstrap_longtail.log" python -m pip install -r requirements-train.txt

STAGE3_USE_8BIT="$(config_uses_8bit_adam "$STAGE3_CONFIG")"
STAGE4_USE_8BIT="$(config_uses_8bit_adam "$STAGE4_CONFIG")"
if [[ "$STAGE3_USE_8BIT" != "1" && "$STAGE4_USE_8BIT" != "1" ]]; then
  log "Both stage configs disable 8-bit Adam. Removing bitsandbytes to avoid incompatible triton imports."
  run_with_log "bootstrap_longtail.log" python -m pip uninstall -y bitsandbytes || true
fi

STAGE3_OUTPUT="$OUTPUT_ROOT/generation_lora_stage3_longtail_4090"
STAGE4_OUTPUT="$OUTPUT_ROOT/generation_lora_stage4_garment_reveal_4090"

if [[ "$SKIP_STAGE3_IF_DONE" == "1" ]] && stage_is_done "$STAGE3_OUTPUT"; then
  log "Stage 3 already completed, skipping."
else
  STAGE3_RESUME_ARGS="$(stage_resume_args "$STAGE3_OUTPUT")"
  log "Stage 3 starting from base LoRA: $BASE_LORA"
  # shellcheck disable=SC2086
  run_with_log "stage3_longtail.log" accelerate launch --config_file "$ACCELERATE_CONFIG" scripts/train_hair_lora.py \
    --config "$STAGE3_CONFIG" \
    --manifest "$LONGTAIL_DATA_ROOT/manifests/style_train_stage3_longtail_mix.jsonl" \
    --validation-manifest "$LONGTAIL_DATA_ROOT/manifests/style_eval_stage3_longtail_ood.jsonl" \
    --output-dir "$STAGE3_OUTPUT" \
    --cache-dir "$HF_HOME" \
    --initial-lora-path "$BASE_LORA" \
    $STAGE3_RESUME_ARGS
fi

STAGE3_LORA="$STAGE3_OUTPUT/best"
if [[ ! -d "$STAGE3_LORA" ]]; then
  STAGE3_LORA="$STAGE3_OUTPUT/final"
fi

if [[ "$RUN_STAGE4" != "1" ]]; then
  log "Stage 4 disabled; longtail training complete after stage 3."
  exit 0
fi

if [[ "$SKIP_STAGE4_IF_DONE" == "1" ]] && stage_is_done "$STAGE4_OUTPUT"; then
  log "Stage 4 already completed, skipping."
else
  STAGE4_RESUME_ARGS="$(stage_resume_args "$STAGE4_OUTPUT")"
  log "Stage 4 starting from stage3 LoRA: $STAGE3_LORA"
  # shellcheck disable=SC2086
  run_with_log "stage4_garment.log" accelerate launch --config_file "$ACCELERATE_CONFIG" scripts/train_hair_lora.py \
    --config "$STAGE4_CONFIG" \
    --manifest "$LONGTAIL_DATA_ROOT/manifests/style_train_stage4_garment_reveal.jsonl" \
    --validation-manifest "$LONGTAIL_DATA_ROOT/manifests/style_eval_stage4_garment_reveal.jsonl" \
    --output-dir "$STAGE4_OUTPUT" \
    --cache-dir "$HF_HOME" \
    --initial-lora-path "$STAGE3_LORA" \
    $STAGE4_RESUME_ARGS
fi

log "Stage 3/4 longtail training pipeline complete"
