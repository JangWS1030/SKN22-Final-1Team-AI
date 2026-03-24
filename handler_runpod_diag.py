from __future__ import annotations

import os
import socket
import time
import uuid
from typing import Any


def _normalize_runpod_env() -> None:
    if not os.environ.get("RUNPOD_ENDPOINT_ID"):
        return

    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        pod_id = os.environ.get("HOSTNAME") or f"diag-{uuid.uuid4().hex}"
        os.environ["RUNPOD_POD_ID"] = pod_id

    gpu_type_id = os.environ.get("RUNPOD_GPU_TYPE_ID")
    if not gpu_type_id:
        gpu_size = str(os.environ.get("RUNPOD_GPU_SIZE", "")).strip()
        if gpu_size:
            gpu_type_id = gpu_size.split(",", 1)[0].strip()
            os.environ["RUNPOD_GPU_TYPE_ID"] = gpu_type_id

    replacements = {
        "$RUNPOD_POD_ID": pod_id,
        "$ID": pod_id,
    }
    if gpu_type_id:
        replacements["$RUNPOD_GPU_TYPE_ID"] = gpu_type_id

    for env_key in (
        "RUNPOD_WEBHOOK_GET_JOB",
        "RUNPOD_WEBHOOK_PING",
        "RUNPOD_WEBHOOK_POST_OUTPUT",
        "RUNPOD_WEBHOOK_POST_STREAM",
    ):
        raw = os.environ.get(env_key)
        if not raw:
            continue
        normalized = raw
        for needle, replacement in replacements.items():
            normalized = normalized.replace(needle, replacement)
        os.environ[env_key] = normalized


def handler(job: dict[str, Any]) -> dict[str, Any]:
    inp = (job or {}).get("input") or {}

    sleep_seconds = float(inp.get("sleep_seconds", 0) or 0)
    if sleep_seconds > 0:
        time.sleep(min(sleep_seconds, 30.0))

    if inp.get("force_error"):
        return {"error": "forced diagnostic error"}

    return {
        "status": "ok",
        "mode": "runpod_diag",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "input_keys": sorted(inp.keys()),
        "runpod": {
            "endpoint_id": os.environ.get("RUNPOD_ENDPOINT_ID"),
            "pod_id": os.environ.get("RUNPOD_POD_ID"),
            "gpu_type_id": os.environ.get("RUNPOD_GPU_TYPE_ID"),
            "has_get_job_url": bool(os.environ.get("RUNPOD_WEBHOOK_GET_JOB")),
            "has_ping_url": bool(os.environ.get("RUNPOD_WEBHOOK_PING")),
        },
    }


if __name__ == "__main__":
    _normalize_runpod_env()
    import runpod

    runpod.serverless.start({"handler": handler})
