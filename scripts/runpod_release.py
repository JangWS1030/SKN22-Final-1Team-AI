#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

REST_BASE_URL = "https://rest.runpod.io/v1"
SERVERLESS_BASE_URL = "https://api.runpod.ai/v2"
VERSIONED_BUILD_TAG_RE = re.compile(r"^v\d+$", re.IGNORECASE)


def clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def normalize_optional_value(value: str | None) -> str | None:
    cleaned = clean_env_value(value)
    if cleaned is None:
        return None
    cleaned = cleaned.strip()
    return cleaned or None


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
    timeout: int = 30,
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
        message = response.text.strip()
        raise RuntimeError(f"{method} {url} failed ({response.status_code}): {message[:800]}")
    if not response.text.strip():
        return {}
    return response.json()


def get_endpoint(endpoint_id: str, api_key: str, *, include_workers: bool = False) -> dict[str, Any]:
    params: dict[str, Any] | None = None
    if include_workers:
        params = {"includeWorkers": "true"}
    data = request_json(
        "GET",
        f"{REST_BASE_URL}/endpoints/{endpoint_id}",
        api_key=api_key,
        params=params,
        timeout=30,
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
        timeout=30,
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
        timeout=60,
    )
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected template update response shape.")
    return data


def patch_endpoint(endpoint_id: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = request_json(
        "PATCH",
        f"{REST_BASE_URL}/endpoints/{endpoint_id}",
        api_key=api_key,
        payload=payload,
        timeout=60,
    )
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected endpoint patch response shape.")
    return data


def submit_health_check(endpoint_id: str, api_key: str) -> str:
    data = request_json(
        "POST",
        f"{SERVERLESS_BASE_URL}/{endpoint_id}/run",
        api_key=api_key,
        payload={"input": {"health_check": True}},
        timeout=30,
    )
    if not isinstance(data, dict) or "id" not in data:
        raise RuntimeError(f"Unexpected health-check submission response: {data!r}")
    job_id = str(data["id"])
    status = data.get("status", "UNKNOWN")
    print(f"[submit] health-check job_id={job_id} status={status}")
    return job_id


def poll_job(endpoint_id: str, api_key: str, job_id: str, *, timeout: int, interval: int) -> dict[str, Any]:
    deadline = time.time() + timeout
    url = f"{SERVERLESS_BASE_URL}/{endpoint_id}/status/{job_id}"

    while time.time() < deadline:
        data = request_json("GET", url, api_key=api_key, timeout=60)
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected job status response: {data!r}")

        status = str(data.get("status", "UNKNOWN"))
        if status == "COMPLETED":
            output_url = data.get("output_url")
            if output_url:
                fetched = requests.get(str(output_url), timeout=120)
                fetched.raise_for_status()
                payload = fetched.json()
                if not isinstance(payload, dict):
                    raise RuntimeError(f"Unexpected output_url payload: {payload!r}")
                return payload

            output = data.get("output", {})
            if not isinstance(output, dict):
                raise RuntimeError(f"Unexpected job output payload: {output!r}")
            return output

        if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            error = data.get("error") or (data.get("output") or {}).get("error", "")
            raise RuntimeError(f"Health-check job {status}: {error}")

        remaining = max(0, int(deadline - time.time()))
        print(f"[poll] status={status} remaining={remaining}s", end="\r", flush=True)
        time.sleep(interval)

    raise TimeoutError(f"Health-check job did not complete within {timeout}s.")


def wait_for_version_change(
    endpoint_id: str,
    api_key: str,
    *,
    previous_version: int | None,
    target_image: str,
    timeout: int,
    interval: int,
) -> tuple[bool, dict[str, Any]]:
    deadline = time.time() + timeout
    last_endpoint: dict[str, Any] = {}

    while time.time() < deadline:
        endpoint = get_endpoint(endpoint_id, api_key, include_workers=True)
        last_endpoint = endpoint

        version_raw = endpoint.get("version")
        current_version = int(version_raw) if isinstance(version_raw, int) else None
        workers = endpoint.get("workers") or []
        worker_images = sorted(
            {
                str(worker.get("imageName"))
                for worker in workers
                if isinstance(worker, dict) and worker.get("imageName")
            }
        )

        if previous_version is not None and current_version is not None and current_version > previous_version:
            print(f"[wait] endpoint version changed: {previous_version} -> {current_version}")
            return True, endpoint

        if target_image in worker_images:
            print(f"[wait] active worker image now includes target image: {target_image}")
            return True, endpoint

        version_label = current_version if current_version is not None else "unknown"
        print(
            f"[wait] endpoint version={version_label} worker_images={worker_images or ['(none)']}",
            flush=True,
        )
        time.sleep(interval)

    return False, last_endpoint


def infer_repository(image_name: str) -> str:
    if "@" in image_name:
        return image_name.split("@", 1)[0]
    if ":" in image_name.rsplit("/", 1)[-1]:
        return image_name.rsplit(":", 1)[0]
    return image_name


def infer_build_tag(image_name: str) -> str | None:
    image_name = str(image_name or "").strip()
    if not image_name or "@" in image_name:
        return None
    tail = image_name.rsplit("/", 1)[-1]
    if ":" not in tail:
        return None
    return tail.rsplit(":", 1)[-1].strip() or None


def build_tag_matches_expected(
    expected_build_tag: str | None,
    actual_build_tag: str | None,
) -> tuple[bool, str | None]:
    expected = str(expected_build_tag or "").strip()
    actual = str(actual_build_tag or "").strip()
    if not expected:
        return True, None
    if actual == expected:
        return True, None
    if expected.lower() == "latest" and VERSIONED_BUILD_TAG_RE.fullmatch(actual):
        return True, (
            "[health] accepted build_tag alias: "
            f"target tag 'latest' resolved to concrete build '{actual}'."
        )
    return False, None


def resolve_target_image(
    *,
    image: str | None,
    image_tag: str | None,
    image_repo: str | None,
    current_image: str,
) -> str:
    image = normalize_optional_value(image)
    image_tag = normalize_optional_value(image_tag)
    image_repo = normalize_optional_value(image_repo)
    if image:
        return image
    if not image_tag:
        raise ValueError("Provide either --image or --image-tag.")
    repository = image_repo or infer_repository(current_image)
    return f"{repository}:{image_tag}"


def build_template_update_payload(template: dict[str, Any], target_image: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"imageName": target_image}
    passthrough_keys = (
        "name",
        "containerDiskInGb",
        "containerRegistryAuthId",
        "dockerEntrypoint",
        "dockerStartCmd",
        "env",
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


def summarize_workers(endpoint: dict[str, Any]) -> list[dict[str, str]]:
    workers = endpoint.get("workers") or []
    summaries: list[dict[str, str]] = []
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        summaries.append(
            {
                "id": str(worker.get("id") or ""),
                "image": str(worker.get("imageName") or ""),
                "version": str(worker.get("slsVersion") or ""),
                "desiredStatus": str(worker.get("desiredStatus") or ""),
                "lastStatusChange": str(worker.get("lastStatusChange") or ""),
            }
        )
    return summaries


def wait_for_worker_bounds(
    endpoint_id: str,
    api_key: str,
    *,
    expected_workers_min: int | None = None,
    expected_workers_max: int | None = None,
    timeout: int = 90,
    interval: int = 3,
) -> None:
    deadline = time.time() + max(5, timeout)
    while time.time() < deadline:
        endpoint = get_endpoint(endpoint_id, api_key, include_workers=False)
        current_min = int(endpoint.get("workersMin") or 0)
        current_max = int(endpoint.get("workersMax") or 0)
        min_ok = expected_workers_min is None or current_min == expected_workers_min
        max_ok = expected_workers_max is None or current_max == expected_workers_max
        if min_ok and max_ok:
            return
        print(
            "[refresh] waiting for bounds "
            f"(current min/max={current_min}/{current_max}, "
            f"expected min/max={expected_workers_min}/{expected_workers_max})"
        )
        time.sleep(max(1, interval))
    raise RuntimeError(
        "Timed out waiting for endpoint worker bounds to settle "
        f"(expected min/max={expected_workers_min}/{expected_workers_max})."
    )


def refresh_worker_pool(
    endpoint_id: str,
    api_key: str,
    *,
    restore_workers_min: int,
    restore_workers_max: int,
    pause_seconds: int,
) -> None:
    normalized_restore_min = max(0, int(restore_workers_min))
    normalized_restore_max = max(normalized_restore_min, int(restore_workers_max))

    if normalized_restore_min > 0:
        print(f"[refresh] scaling workersMin -> 0 for endpoint={endpoint_id}")
        patch_endpoint(endpoint_id, api_key, {"workersMin": 0})
        wait_for_worker_bounds(
            endpoint_id,
            api_key,
            expected_workers_min=0,
            expected_workers_max=normalized_restore_max,
        )

    print(f"[refresh] scaling workersMax -> 0 for endpoint={endpoint_id}")
    patch_endpoint(endpoint_id, api_key, {"workersMax": 0})
    wait_for_worker_bounds(
        endpoint_id,
        api_key,
        expected_workers_min=0 if normalized_restore_min > 0 else None,
        expected_workers_max=0,
    )
    time.sleep(max(1, pause_seconds))
    print(f"[refresh] restoring workersMax -> {normalized_restore_max}")
    patch_endpoint(endpoint_id, api_key, {"workersMax": normalized_restore_max})
    wait_for_worker_bounds(
        endpoint_id,
        api_key,
        expected_workers_min=0 if normalized_restore_min > 0 else None,
        expected_workers_max=normalized_restore_max,
    )

    if normalized_restore_min > 0:
        print(f"[refresh] restoring workersMin -> {normalized_restore_min}")
        patch_endpoint(endpoint_id, api_key, {"workersMin": normalized_restore_min})
        wait_for_worker_bounds(
            endpoint_id,
            api_key,
            expected_workers_min=normalized_restore_min,
            expected_workers_max=normalized_restore_max,
        )


def redact_for_display(value: Any, *, key_hint: str | None = None) -> Any:
    secret_markers = ("token", "secret", "password", "key")

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for item_key, item_value in value.items():
            redacted[item_key] = redact_for_display(item_value, key_hint=str(item_key))
        return redacted

    if isinstance(value, list):
        return [redact_for_display(item, key_hint=key_hint) for item in value]

    if isinstance(value, str) and key_hint:
        lowered = key_hint.lower()
        if any(marker in lowered for marker in secret_markers):
            return "***REDACTED***"
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update a RunPod endpoint-bound template image and wait for rollout."
    )
    parser.add_argument(
        "--api-key",
        default=clean_env_value(os.environ.get("RUNPOD_API_KEY")),
        help="RunPod API key. Defaults to RUNPOD_API_KEY from environment or .env.",
    )
    parser.add_argument(
        "--endpoint-id",
        default=clean_env_value(os.environ.get("RUNPOD_ENDPOINT_ID")),
        help="RunPod serverless endpoint ID. Defaults to RUNPOD_ENDPOINT_ID from environment or .env.",
    )
    parser.add_argument("--image", default=None, help="Full Docker image name to release.")
    parser.add_argument("--image-tag", default=None, help="Docker tag to release with the inferred repository.")
    parser.add_argument("--image-repo", default=None, help="Override Docker repository when using --image-tag.")
    parser.add_argument("--timeout", type=int, default=1200, help="Total timeout in seconds. Default: 1200.")
    parser.add_argument("--poll-interval", type=int, default=5, help="Polling interval in seconds. Default: 5.")
    parser.add_argument("--skip-health-check", action="store_true", help="Skip handler-level health check.")
    parser.add_argument("--skip-version-wait", action="store_true", help="Skip waiting for endpoint version rollout.")
    parser.add_argument("--force", action="store_true", help="Update even if the current image already matches.")
    parser.add_argument(
        "--worker-refresh-pause",
        type=int,
        default=8,
        help="Seconds to keep workersMax=0 when forcing a worker-pool refresh. Default: 8.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show the resolved release plan without changing RunPod.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.api_key:
        raise SystemExit("RUNPOD_API_KEY or --api-key is required.")
    if not args.endpoint_id:
        raise SystemExit("RUNPOD_ENDPOINT_ID or --endpoint-id is required.")

    endpoint = get_endpoint(args.endpoint_id, args.api_key, include_workers=True)
    template_id = endpoint.get("templateId")
    if not template_id:
        raise SystemExit(f"Endpoint {args.endpoint_id} does not expose a templateId.")

    template = get_template(str(template_id), args.api_key)
    current_image = str(template.get("imageName") or "").strip()
    if not current_image:
        raise SystemExit(f"Template {template_id} does not expose imageName.")

    target_image = resolve_target_image(
        image=args.image,
        image_tag=args.image_tag,
        image_repo=args.image_repo,
        current_image=current_image,
    )
    expected_build_tag = infer_build_tag(target_image)
    previous_version = endpoint.get("version") if isinstance(endpoint.get("version"), int) else None
    original_workers_min = int(endpoint.get("workersMin") or 0)
    original_workers_max = int(endpoint.get("workersMax") or 0)

    print(f"[release] endpoint_id={args.endpoint_id}")
    print(f"[release] template_id={template_id}")
    print(f"[release] current_image={current_image}")
    print(f"[release] target_image={target_image}")
    print(f"[release] endpoint_version={previous_version if previous_version is not None else 'unknown'}")

    if current_image == target_image and not args.force:
        print("[release] target image already active on the bound template. Nothing to change.")
        return 0

    update_payload = build_template_update_payload(template, target_image)
    if args.dry_run:
        print("[dry-run] template update payload:")
        print(json.dumps(redact_for_display(update_payload), ensure_ascii=False, indent=2))
        return 0

    updated_template = update_template(str(template_id), args.api_key, update_payload)
    updated_image = updated_template.get("imageName")
    print(f"[release] template updated to image={updated_image}")

    rollout_confirmed = False
    if not args.skip_version_wait:
        wait_timeout = max(args.poll_interval * 2, min(args.timeout, 300))
        rollout_confirmed, _ = wait_for_version_change(
            args.endpoint_id,
            args.api_key,
            previous_version=previous_version,
            target_image=target_image,
            timeout=wait_timeout,
            interval=args.poll_interval,
        )
        if not rollout_confirmed:
            print("[warn] Endpoint version did not change within the short rollout window. Proceeding to health check.")

    if not args.skip_health_check:
        health_timeout = max(60, args.timeout)
        output: dict[str, Any] = {}
        build_tag_confirmed = expected_build_tag is None
        max_health_attempts = 3
        for attempt in range(1, max_health_attempts + 1):
            job_id = submit_health_check(args.endpoint_id, args.api_key)
            output = poll_job(
                args.endpoint_id,
                args.api_key,
                job_id,
                timeout=health_timeout,
                interval=args.poll_interval,
            )
            actual_build_tag = str(output.get("build_tag") or "").strip()
            build_tag_matched, build_tag_note = build_tag_matches_expected(
                expected_build_tag,
                actual_build_tag,
            )
            if build_tag_matched:
                build_tag_confirmed = True
                if build_tag_note:
                    print(build_tag_note)
                break

            pod_id = str(((output.get("runpod") or {}).get("pod_id") or "")).strip()
            print(
                "[warn] health-check build_tag mismatch: "
                f"expected={expected_build_tag} actual={actual_build_tag or '(missing)'} pod_id={pod_id or '(missing)'}"
            )
            worker_endpoint = get_endpoint(args.endpoint_id, args.api_key, include_workers=True)
            worker_summaries = summarize_workers(worker_endpoint)
            if worker_summaries:
                print("[warn] active worker summary:")
                print(json.dumps(worker_summaries, ensure_ascii=False, indent=2))

            mixed_images = {
                summary["image"]
                for summary in worker_summaries
                if summary.get("image")
            }
            can_refresh_workers = (
                attempt == 1
                and original_workers_max > 0
                and bool(mixed_images)
                and mixed_images != {target_image}
            )
            if can_refresh_workers:
                refresh_worker_pool(
                    args.endpoint_id,
                    args.api_key,
                    restore_workers_min=original_workers_min,
                    restore_workers_max=original_workers_max,
                    pause_seconds=args.worker_refresh_pause,
                )
            elif attempt < max_health_attempts:
                time.sleep(args.poll_interval)

        if not build_tag_confirmed:
            raise RuntimeError(
                "Health-check never reached the expected build tag. "
                f"expected={expected_build_tag} last_output={output!r}"
            )
        status = str(output.get("status", "")).lower()
        if status and status != "ok":
            raise RuntimeError(f"Health-check returned an unexpected status payload: {output!r}")
        print()
        print("[health] completed successfully")
        print(json.dumps(output, ensure_ascii=False, indent=2))

    print("[release] RunPod release completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
