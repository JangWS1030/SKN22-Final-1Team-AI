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

HANDLER_FILE="${RUNPOD_HANDLER_FILE:-handler_sd.py}"
SERVICE_MODE="${MIRRAI_SERVICE_MODE:-serverless}"

if [[ "${SERVICE_MODE}" == "http" ]]; then
  HTTP_APP="${MIRRAI_HTTP_APP:-internal_api_app:app}"
  HTTP_HOST="${MIRRAI_HTTP_HOST:-0.0.0.0}"
  HTTP_PORT="${MIRRAI_HTTP_PORT:-8000}"
  echo "[entrypoint_sd] Starting internal HTTP API: ${HTTP_APP} (${HTTP_HOST}:${HTTP_PORT})"
  exec python -m uvicorn "${HTTP_APP}" --host "${HTTP_HOST}" --port "${HTTP_PORT}"
fi

echo "[entrypoint_sd] Starting handler: ${HANDLER_FILE}"
exec python "${HANDLER_FILE}" "$@"
