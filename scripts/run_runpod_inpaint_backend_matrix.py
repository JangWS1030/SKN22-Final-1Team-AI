#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
from dotenv import load_dotenv
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

RUNPOD_BASE_URL = "https://api.runpod.ai/v2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

DEFAULT_BACKENDS = ["sd15_controlnet"]
DEFAULT_STYLES: List[Dict[str, Any]] = [
    {
        "name": "short_two_block_open_forehead",
        "hairstyle_text": "male haircut, short soft two-block, open forehead, clean side line, natural salon style",
        "color_text": "natural black",
        "subject_gender": "male",
    },
    {
        "name": "layered_wolf_cut",
        "hairstyle_text": "layered wolf cut, soft volume, face-framing layers, realistic salon hair",
        "color_text": "ash brown",
        "subject_gender": "",
    },
    {
        "name": "short_bob_no_face_damage",
        "hairstyle_text": "short bob cut, compact jaw-length silhouette, natural realistic hair, clean neckline",
        "color_text": "brown",
        "subject_gender": "female",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RunPod serverless inference across local images and inpaint model backends."
    )
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "images")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "runpod_inpaint_backend_matrix")
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--styles-json", type=Path, default=None, help="Optional JSON file containing a list of style payloads.")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--return-intermediates", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    value = value.strip("._-").lower()
    return value or "item"


def iter_images(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def load_styles(path: Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return list(DEFAULT_STYLES)
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("--styles-json must contain a JSON list.")
    styles: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"style item #{idx} must be an object.")
        styles.append(item)
    return styles


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def request_json(method: str, url: str, api_key: str, payload: Dict[str, Any] | None = None, timeout: int = 60) -> Dict[str, Any]:
    response = requests.request(
        method,
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected RunPod response: {data!r}")
    return data


def submit_job(endpoint_id: str, api_key: str, payload: Dict[str, Any]) -> str:
    data = request_json(
        "POST",
        f"{RUNPOD_BASE_URL}/{endpoint_id}/run",
        api_key,
        {"input": payload},
        timeout=30,
    )
    job_id = str(data["id"])
    print(f"[submit] backend={payload.get('generation_backend')} image={payload.get('_image_name')} style={payload.get('_style_name')} job_id={job_id} status={data.get('status')}")
    return job_id


def poll_job(endpoint_id: str, api_key: str, job_id: str, timeout: int) -> Dict[str, Any]:
    deadline = time.time() + timeout
    status_url = f"{RUNPOD_BASE_URL}/{endpoint_id}/status/{job_id}"
    while time.time() < deadline:
        data = request_json("GET", status_url, api_key, timeout=60)
        status = str(data.get("status", "UNKNOWN"))
        if status == "COMPLETED":
            output_url = data.get("output_url")
            if output_url:
                fetched = requests.get(str(output_url), timeout=300)
                fetched.raise_for_status()
                output = fetched.json()
            else:
                output = data.get("output", {})
            if not isinstance(output, dict):
                return {"raw_output": output}
            return output
        if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            error = data.get("error") or (data.get("output") or {}).get("error", "")
            raise RuntimeError(f"RunPod job {status}: {error}")
        print(f"[poll] job_id={job_id} status={status}", end="\r", flush=True)
        time.sleep(5)
    raise TimeoutError(f"RunPod job {job_id} did not complete within {timeout}s")


def save_output_images(output: Dict[str, Any], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for item in output.get("results", []) or []:
        image_base64 = item.get("image_base64")
        if not image_base64:
            continue
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
        rank = int(item.get("rank", 0))
        seed = int(item.get("seed", 0))
        path = out_dir / f"rank{rank}_seed{seed}.jpg"
        image.save(path, format="JPEG", quality=95)
        saved.append(str(path))
    return saved


def build_payload(image_path: Path, backend: str, style: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "image": image_to_base64(image_path),
        "hairstyle_text": str(style.get("hairstyle_text", "")).strip(),
        "color_text": str(style.get("color_text", "")).strip(),
        "subject_gender": str(style.get("subject_gender", "")).strip(),
        "generation_backend": backend,
        "top_k": max(1, min(5, int(args.top_k))),
        "return_base64": True,
        "return_intermediates": bool(args.return_intermediates),
        "_image_name": image_path.name,
        "_style_name": style.get("name", "style"),
    }
    prompt_context = style.get("prompt_context")
    if isinstance(prompt_context, dict):
        payload["prompt_context"] = prompt_context
    sd_prompt_data = style.get("sd_prompt_data")
    if isinstance(sd_prompt_data, dict):
        payload["sd_prompt_data"] = sd_prompt_data
    return payload


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("RUNPOD_API_KEY or --api-key is required.")
    if not args.endpoint_id:
        raise SystemExit("RUNPOD_ENDPOINT_ID or --endpoint-id is required.")
    if not args.images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {args.images_dir}")

    backends = [part.strip() for part in args.backends.split(",") if part.strip()]
    styles = load_styles(args.styles_json)
    images = list(iter_images(args.images_dir))
    if args.limit_images > 0:
        images = images[: args.limit_images]
    if not images:
        raise SystemExit(f"No test images found under {args.images_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    total = len(backends) * len(styles) * len(images)
    completed = 0

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for backend in backends:
            for image_path in images:
                for style in styles:
                    style_name = str(style.get("name") or "style")
                    payload = build_payload(image_path, backend, style, args)
                    record: Dict[str, Any] = {
                        "backend": backend,
                        "image": str(image_path),
                        "style": style_name,
                        "status": "started",
                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    out_dir = args.output_dir / slugify(backend) / slugify(image_path.stem) / slugify(style_name)
                    try:
                        job_id = submit_job(args.endpoint_id, args.api_key, payload)
                        output = poll_job(args.endpoint_id, args.api_key, job_id, args.timeout)
                        saved = save_output_images(output, out_dir)
                        raw_path = out_dir / "output.json"
                        raw_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
                        record.update(
                            {
                                "status": "ok",
                                "job_id": job_id,
                                "elapsed_seconds": output.get("elapsed_seconds"),
                                "generation_backend": output.get("generation_backend"),
                                "saved_images": saved,
                                "raw_output": str(raw_path),
                            }
                        )
                    except Exception as exc:
                        record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                        manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                        manifest.flush()
                        raise
                    completed += 1
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest.flush()
                    print(f"[matrix] completed {completed}/{total}: backend={backend} image={image_path.name} style={style_name}")

    print(
        json.dumps(
            {
                "status": "ok",
                "total_jobs": total,
                "manifest": str(manifest_path),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
