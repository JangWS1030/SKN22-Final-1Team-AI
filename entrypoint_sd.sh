#!/bin/bash
set -e

HANDLER_FILE="${RUNPOD_HANDLER_FILE:-handler_sd.py}"

echo "[entrypoint_sd] Starting handler: ${HANDLER_FILE}"
exec python "${HANDLER_FILE}" "$@"
