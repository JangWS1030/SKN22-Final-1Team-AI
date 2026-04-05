#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.runpod_release import (
    build_template_update_payload,
    get_endpoint,
    get_template,
    patch_endpoint,
    refresh_worker_pool,
    update_template,
)
load_dotenv(PROJECT_ROOT / ".env")


def _clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore RunPod serverless cache volume attachment and cache env vars."
    )
    parser.add_argument("--api-key", default=_clean_env_value(os.environ.get("RUNPOD_API_KEY")))
    parser.add_argument("--endpoint-id", default=_clean_env_value(os.environ.get("RUNPOD_ENDPOINT_ID")))
    parser.add_argument("--network-volume-id", required=True)
    parser.add_argument("--hf-home", default="/runpod-volume/huggingface")
    parser.add_argument("--torch-home", default="/runpod-volume/torch")
    parser.add_argument("--preload-on-startup", action="store_true")
    parser.add_argument("--worker-refresh-pause", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _normalize_template_env(env_value: Any) -> dict[str, str]:
    if isinstance(env_value, dict):
        return {
            str(key): str(value)
            for key, value in env_value.items()
            if key is not None and value is not None
        }
    if isinstance(env_value, list):
        normalized: dict[str, str] = {}
        for item in env_value:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            value = item.get("value")
            if key is None or value is None:
                continue
            normalized[str(key)] = str(value)
        return normalized
    return {}


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("RUNPOD_API_KEY or --api-key is required.")
    if not args.endpoint_id:
        raise SystemExit("RUNPOD_ENDPOINT_ID or --endpoint-id is required.")

    endpoint = get_endpoint(args.endpoint_id, args.api_key, include_workers=True)
    template_id = str(endpoint.get("templateId") or "")
    if not template_id:
        raise SystemExit(f"Endpoint {args.endpoint_id} does not expose a templateId.")

    template = get_template(template_id, args.api_key)
    current_image = str(template.get("imageName") or "").strip()
    if not current_image:
        raise SystemExit(f"Template {template_id} does not expose imageName.")

    merged_env = _normalize_template_env(template.get("env"))
    merged_env["HF_HOME"] = args.hf_home
    merged_env["TORCH_HOME"] = args.torch_home
    if args.preload_on_startup:
        merged_env["MIRRAI_PRELOAD_ON_STARTUP"] = "1"

    template_payload = build_template_update_payload(template, current_image)
    template_payload["env"] = merged_env
    endpoint_payload = {"networkVolumeId": args.network_volume_id}

    original_workers_min = int(endpoint.get("workersMin") or 0)
    original_workers_max = int(endpoint.get("workersMax") or 0)

    print(f"[restore] endpoint_id={args.endpoint_id}")
    print(f"[restore] template_id={template_id}")
    print(f"[restore] current_image={current_image}")
    print(f"[restore] target_network_volume_id={args.network_volume_id}")
    print(f"[restore] workers_min_max={original_workers_min}/{original_workers_max}")

    if args.dry_run:
        print("[dry-run] endpoint patch payload:")
        print(json.dumps(endpoint_payload, ensure_ascii=False, indent=2))
        print("[dry-run] template update payload:")
        print(json.dumps(template_payload, ensure_ascii=False, indent=2))
        return 0

    patch_endpoint(args.endpoint_id, args.api_key, endpoint_payload)
    update_template(template_id, args.api_key, template_payload)

    if original_workers_max > 0:
        refresh_worker_pool(
            args.endpoint_id,
            args.api_key,
            restore_workers_min=original_workers_min,
            restore_workers_max=original_workers_max,
            pause_seconds=args.worker_refresh_pause,
        )

    repaired_endpoint = get_endpoint(args.endpoint_id, args.api_key, include_workers=True)
    repaired_template = get_template(template_id, args.api_key)
    print("[restore] completed")
    print(
        json.dumps(
            {
                "endpoint": {
                    "id": repaired_endpoint.get("id"),
                    "version": repaired_endpoint.get("version"),
                    "networkVolumeId": repaired_endpoint.get("networkVolumeId"),
                    "workersMin": repaired_endpoint.get("workersMin"),
                    "workersMax": repaired_endpoint.get("workersMax"),
                },
                "template": {
                    "id": repaired_template.get("id"),
                    "imageName": repaired_template.get("imageName"),
                    "env": repaired_template.get("env"),
                    "volumeMountPath": repaired_template.get("volumeMountPath"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
