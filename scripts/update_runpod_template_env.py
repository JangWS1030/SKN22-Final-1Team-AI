#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

REST_BASE_URL = "https://rest.runpod.io/v1"

LEGACY_ENV_KEYS = (
    "SEG_PTH_GITHUB_OWNER",
    "SEG_PTH_GITHUB_REPO",
    "SEG_PTH_GITHUB_PATH",
    "SEG_PTH_GITHUB_REF",
    "SEG_PTH_PATH",
    "GITHUB_TOKEN",
    "ENABLE_STARTUP_GIT_PULL",
    "RUNPOD_DEBUG_LEVEL",
    "MODEL_DOWNLOAD_TIMEOUT",
    "TORCH_HOME",
)


def clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def request_json(
    method: str,
    url: str,
    *,
    api_key: str,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> Any:
    response = requests.request(
        method,
        url,
        headers=build_headers(api_key),
        params=params,
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(
            f"{method} {url} failed ({response.status_code}): {response.text[:800]}"
        )
    if not response.text.strip():
        return {}
    return response.json()


def get_endpoint(endpoint_id: str, api_key: str) -> dict[str, Any]:
    data = request_json(
        "GET",
        f"{REST_BASE_URL}/endpoints/{endpoint_id}",
        api_key=api_key,
        params={"includeTemplate": "true"},
    )
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected endpoint response shape.")
    return data


def get_template(template_id: str, api_key: str) -> dict[str, Any]:
    data = request_json(
        "GET",
        f"{REST_BASE_URL}/templates/{template_id}",
        api_key=api_key,
        params={"includeEndpointBoundTemplates": "true"},
    )
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected template response shape.")
    return data


def update_template(template_id: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = request_json(
        "POST",
        f"{REST_BASE_URL}/templates/{template_id}/update",
        api_key=api_key,
        payload=payload,
    )
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected template update response shape.")
    return data


def build_template_update_payload(template: dict[str, Any], env_map: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "imageName": template["imageName"],
        "env": env_map,
    }
    passthrough_keys = (
        "name",
        "containerDiskInGb",
        "containerRegistryAuthId",
        "dockerEntrypoint",
        "dockerStartCmd",
        "isPublic",
        "ports",
        "readme",
        "volumeInGb",
        "volumeMountPath",
    )
    for key in passthrough_keys:
        if key in template and template[key] is not None:
            payload[key] = template[key]
    return payload


def parse_set_arg(items: list[str]) -> dict[str, str]:
    env_updates: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        env_updates[key.strip()] = value.strip()
    return env_updates


def redact_env_map(env_map: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in sorted(env_map.items()):
        lowered = key.lower()
        if any(marker in lowered for marker in ("token", "secret", "password", "key")):
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    return redacted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update env vars on the endpoint-bound RunPod template."
    )
    parser.add_argument(
        "--api-key",
        default=clean_env_value(os.environ.get("RUNPOD_API_KEY")),
        help="RunPod API key. Defaults to RUNPOD_API_KEY from environment or .env.",
    )
    parser.add_argument(
        "--endpoint-id",
        default=clean_env_value(os.environ.get("RUNPOD_ENDPOINT_ID")),
        help="RunPod endpoint id. Defaults to RUNPOD_ENDPOINT_ID from environment or .env.",
    )
    parser.add_argument(
        "--set",
        dest="set_items",
        action="append",
        default=[],
        help="Set or override an env var on the bound template using KEY=VALUE.",
    )
    parser.add_argument(
        "--remove",
        dest="remove_items",
        action="append",
        default=[],
        help="Remove an env var from the bound template.",
    )
    parser.add_argument(
        "--prune-legacy",
        action="store_true",
        help="Remove known legacy env vars from the bound template.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved env payload without updating RunPod.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("RUNPOD_API_KEY or --api-key is required.")
    if not args.endpoint_id:
        raise SystemExit("RUNPOD_ENDPOINT_ID or --endpoint-id is required.")

    endpoint = get_endpoint(args.endpoint_id, args.api_key)
    template_id = endpoint.get("templateId")
    if not template_id:
        raise SystemExit(f"Endpoint {args.endpoint_id} does not expose a templateId.")

    template = get_template(str(template_id), args.api_key)
    current_env = dict(template.get("env") or {})
    desired_env = dict(current_env)

    for key, value in parse_set_arg(args.set_items).items():
        desired_env[key] = value
    for key in args.remove_items:
        desired_env.pop(key, None)
    if args.prune_legacy:
        for key in LEGACY_ENV_KEYS:
            desired_env.pop(key, None)

    update_payload = build_template_update_payload(template, desired_env)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "endpointId": args.endpoint_id,
                    "templateId": template_id,
                    "env": redact_env_map(desired_env),
                    "updatePayload": redact_env_map(update_payload["env"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    updated = update_template(str(template_id), args.api_key, update_payload)
    print(
        json.dumps(
            {
                "endpointId": args.endpoint_id,
                "templateId": template_id,
                "updatedEnv": redact_env_map(updated.get("env") or desired_env),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
