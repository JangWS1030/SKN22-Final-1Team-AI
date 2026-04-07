#!/bin/bash
set -euo pipefail

normalize_runpod_webhook_env() {
  local name="$1"
  local value="${!name:-}"
  local updated="$value"

  if [[ -z "$value" ]]; then
    return
  fi

  if [[ -n "${RUNPOD_POD_ID:-}" && "$updated" == *'$RUNPOD_POD_ID'* ]]; then
    updated="${updated//\$RUNPOD_POD_ID/${RUNPOD_POD_ID}}"
    echo "[entrypoint_sd] expanded ${name} placeholder: \$RUNPOD_POD_ID"
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

HANDLER_FILE="${RUNPOD_HANDLER_FILE:-handler_sd.py}"

echo "[entrypoint_sd] Starting handler: ${HANDLER_FILE}"
exec python "${HANDLER_FILE}" "$@"
