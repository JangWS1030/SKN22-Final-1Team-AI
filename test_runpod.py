#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

RUNPOD_BASE_URL = "https://api.runpod.ai/v2"
DEFAULT_TIMEOUT = 1200
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RunPod serverless smoke test for SD handler")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--image", type=Path, default=None, help="Local image path")
    parser.add_argument("--image-url", default="", help="Remote image URL")
    parser.add_argument("--hairstyle", default="", help="Hairstyle text")
    parser.add_argument("--color", default="", help="Color text")
    parser.add_argument("--gender", default="", help="Optional subject gender hint: male/female")
    parser.add_argument(
        "--generation-backend",
        default="",
        help="Generation backend: sd15_controlnet, sdxl_inpaint, flux_fill, powerpaint",
    )
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--health-check", action="store_true")
    mode.add_argument("--analyze-face", action="store_true")
    parser.add_argument("--include-visualization", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def submit_job(endpoint_id: str, api_key: str, payload: dict) -> str:
    response = requests.post(
        f"{RUNPOD_BASE_URL}/{endpoint_id}/run",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"input": payload},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    job_id = str(data["id"])
    print(f"[submit] job_id={job_id} status={data.get('status')}")
    return job_id


def poll_job(endpoint_id: str, api_key: str, job_id: str, timeout: int) -> dict:
    deadline = time.time() + timeout
    status_url = f"{RUNPOD_BASE_URL}/{endpoint_id}/status/{job_id}"

    while time.time() < deadline:
        response = requests.get(status_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
        response.raise_for_status()
        data = response.json()
        status = str(data.get("status", "UNKNOWN"))
        if status == "COMPLETED":
            output_url = data.get("output_url")
            if output_url:
                print(f"\n[poll] Result too large. Fetching from {output_url}...")
                fetched = requests.get(str(output_url), timeout=300)
                fetched.raise_for_status()
                return fetched.json()
            output = data.get("output")
            if output is not None:
                return output if isinstance(output, dict) else {"results": [], "raw_output": output}
            # If COMPLETED but no output key, maybe the results are top-level or it failed silently
            return data
        if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            error = data.get("error") or data.get("output", {}).get("error") if isinstance(data.get("output"), dict) else ""
            raise RuntimeError(f"RunPod job {status}: {error or 'Unknown error'}")
        print(f"[poll] status={status}", end="\r", flush=True)
        time.sleep(5)

    raise TimeoutError(f"RunPod job did not complete within {timeout}s")


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def save_results(output: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for item in output.get("results", []) or []:
        image_base64 = item.get("image_base64")
        if not image_base64:
            continue
        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
        path = out_dir / f"result_rank{item.get('rank', 0)}_seed{item.get('seed', 0)}.jpg"
        image.save(path, format="JPEG", quality=95)
        saved.append(path)

    intermediates = output.get("intermediates", {})
    if isinstance(intermediates, dict):
        for name, b64 in intermediates.items():
            if not b64:
                continue
            if isinstance(b64, str) and "," in b64:
                b64 = b64.split(",", 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            path = out_dir / f"debug_{name}.jpg"
            image.save(path, format="JPEG", quality=95)
            saved.append(path)

    intermediate_data = output.get("intermediate_data", {})
    if intermediate_data:
        import json
        json_path = out_dir / "intermediate_data.json"
        with open(json_path, "w") as f:
            json.dump(intermediate_data, f, indent=2)
        saved.append(json_path)

    return saved


def save_analysis_visualization(output: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    visualization_base64 = output.get("visualization_base64")
    if visualization_base64:
        if isinstance(visualization_base64, str) and "," in visualization_base64:
            visualization_base64 = visualization_base64.split(",", 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(visualization_base64))).convert("RGB")
        path = out_dir / "analyze_face_visualization.jpg"
        image.save(path, format="JPEG", quality=95)
        saved.append(path)
    return saved


def build_payload(args: argparse.Namespace) -> dict:
    if args.health_check:
        return {"health_check": True}
    if args.analyze_face:
        payload = {
            "action": "analyze_face",
            "include_visualization": args.include_visualization,
        }
        if args.image:
            payload["image"] = image_to_base64(args.image)
        elif args.image_url:
            payload["image_url"] = args.image_url
        else:
            raise SystemExit("Provide --image or --image-url for analyze-face test.")
        return payload

    payload = {
        "hairstyle_text": args.hairstyle,
        "color_text": args.color,
        "top_k": args.top_k,
        "return_base64": True,
        "return_intermediates": True,
    }
    if args.gender:
        payload["subject_gender"] = args.gender
    if args.generation_backend:
        payload["generation_backend"] = args.generation_backend
    if args.image:
        payload["image"] = image_to_base64(args.image)
    elif args.image_url:
        payload["image_url"] = args.image_url
    else:
        raise SystemExit("Provide --image or --image-url for inference test.")
    return payload


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("RUNPOD_API_KEY or --api-key is required.")
    if not args.endpoint_id:
        raise SystemExit("RUNPOD_ENDPOINT_ID or --endpoint-id is required.")

    payload = build_payload(args)
    print(json.dumps({"endpoint_id": args.endpoint_id, "payload_keys": sorted(payload.keys())}, ensure_ascii=False))
    job_id = submit_job(args.endpoint_id, args.api_key, payload)
    output = poll_job(args.endpoint_id, args.api_key, job_id, args.timeout)
    with open("raw_output.json", "w") as f:
        json.dump(output, f)

    if args.health_check:
        if str(output.get("status", "")).lower() != "ok":
            raise SystemExit("Health check response did not contain status=ok")
        return 0
    if args.analyze_face:
        if str(output.get("status", "")).lower() != "ok":
            raise SystemExit(f"Analyze-face response did not contain status=ok: {output}")
        saved = save_analysis_visualization(output, args.output_dir)
        print(
            json.dumps(
                {
                    "face_shape": output.get("face_shape"),
                    "golden_ratio_score": output.get("golden_ratio_score"),
                    "saved_images": [str(path) for path in saved],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not output.get("results"):
        print(f"[main] WARNING: 'results' is missing in output. Full output keys: {list(output.keys())}")
        if "error" in output:
            print(f"[main] ERROR in output: {output['error']}")

    saved = save_results(output, args.output_dir)
    print(json.dumps({"saved_images": [str(path) for path in saved]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
