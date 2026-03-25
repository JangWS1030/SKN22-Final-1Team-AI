#!/bin/bash
set -euo pipefail

normalize_runpod_webhook_env() {
  local name="$1"
  local value="${!name:-}"
  local updated="$value"

  if [[ -z "$value" ]]; then
    return
  fi

  if [[ "$name" == "RUNPOD_WEBHOOK_GET_JOB" && "$updated" == *'$RUNPOD_POD_ID'* ]]; then
    updated="${updated//\$RUNPOD_POD_ID/\$ID}"
    echo "[entrypoint_sd] normalized ${name} placeholder: \$RUNPOD_POD_ID -> \$ID"
  fi

  if [[ -n "${RUNPOD_GPU_TYPE_ID:-}" && "$updated" == *'$RUNPOD_GPU_TYPE_ID'* ]]; then
    updated="${updated//\$RUNPOD_GPU_TYPE_ID/${RUNPOD_GPU_TYPE_ID}}"
    echo "[entrypoint_sd] expanded ${name} placeholder: \$RUNPOD_GPU_TYPE_ID"
  fi

  export "${name}=${updated}"
}

if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "[entrypoint_sd] RUNPOD_POD_ID is present."
else
  echo "[entrypoint_sd] RUNPOD_POD_ID is missing."
fi

normalize_runpod_webhook_env RUNPOD_WEBHOOK_GET_JOB
normalize_runpod_webhook_env RUNPOD_WEBHOOK_PING
normalize_runpod_webhook_env RUNPOD_WEBHOOK_POST_OUTPUT
normalize_runpod_webhook_env RUNPOD_WEBHOOK_POST_STREAM

echo "[entrypoint_sd] Starting SD Inpainting handler..."
exec python handler_sd.py
