import argparse
import base64
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

RUNPOD_BASE_URL = "https://api.runpod.ai/v2"
REPO_RELATIVE_MARKERS = ("dataset_build", "data", "images", "output", "models")


def log(message: str) -> None:
    print(message, flush=True)


def resolve_repo_local_path(path_value: str | None) -> Path:
    raw_value = str(path_value or "").strip()
    if not raw_value:
        return Path()

    direct_path = Path(raw_value)
    if direct_path.exists():
        return direct_path

    normalized = raw_value.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]

    for marker in REPO_RELATIVE_MARKERS:
        if marker in parts:
            marker_index = parts.index(marker)
            candidate = (PROJECT_ROOT / Path(*parts[marker_index:])).resolve()
            if candidate.exists():
                return candidate

    if normalized.startswith("/workspace/repo/"):
        candidate = (PROJECT_ROOT / Path(normalized.removeprefix("/workspace/repo/"))).resolve()
        if candidate.exists():
            return candidate

    return direct_path


def imread_grayscale(path: Path) -> np.ndarray | None:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RunPod Endpoint using Local Dataset")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "dataset_build/processed/celeba_dialog_hq_generation/manifests/style_eval.jsonl",
    )
    parser.add_argument("--run-name", type=str, required=True, help="Name of this test run (e.g. 'base' or 'finetune')")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "eval_results")
    parser.add_argument("--submit-timeout", type=int, default=30, help="HTTP timeout in seconds for job submission")
    parser.add_argument("--status-timeout", type=int, default=60, help="HTTP timeout in seconds for job status requests")
    parser.add_argument("--output-timeout", type=int, default=120, help="HTTP timeout in seconds for output fetch requests")
    parser.add_argument("--job-timeout", type=int, default=1800, help="Maximum time in seconds to wait for each RunPod job")
    parser.add_argument("--poll-interval", type=int, default=3, help="Polling interval in seconds")
    parser.add_argument("--status-log-interval", type=int, default=30, help="How often to emit waiting status logs in seconds")
    return parser.parse_args()


def submit_job(endpoint_id: str, api_key: str, payload: dict, submit_timeout: int) -> str:
    response = requests.post(
        f"{RUNPOD_BASE_URL}/{endpoint_id}/run",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"input": payload},
        timeout=submit_timeout,
    )
    response.raise_for_status()
    data = response.json()
    job_id = str(data["id"])
    return job_id


def poll_job(
    endpoint_id: str,
    api_key: str,
    job_id: str,
    timeout: int,
    status_timeout: int,
    output_timeout: int,
    poll_interval: int,
    status_log_interval: int,
) -> dict:
    deadline = time.time() + timeout
    status_url = f"{RUNPOD_BASE_URL}/{endpoint_id}/status/{job_id}"
    last_status = None
    last_status_log = 0.0

    while time.time() < deadline:
        response = requests.get(status_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=status_timeout)
        response.raise_for_status()
        data = response.json()
        status = str(data.get("status", "UNKNOWN"))
        now = time.time()
        elapsed = timeout - max(deadline - now, 0)

        if status != last_status:
            log(f"    status={status} after {elapsed:.1f}s (job_id={job_id})")
            last_status = status
            last_status_log = now
        elif now - last_status_log >= status_log_interval:
            log(f"    waiting... status={status} after {elapsed:.1f}s (job_id={job_id})")
            last_status_log = now

        if status == "COMPLETED":
            output_url = data.get("output_url")
            if output_url:
                fetched = requests.get(str(output_url), timeout=output_timeout)
                fetched.raise_for_status()
                return fetched.json()
            output = data.get("output", {})
            return output if isinstance(output, dict) else {"output": output}

        if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            output = data.get("output", {}) or {}
            error = data.get("error") or output.get("error") or ""
            raise RuntimeError(f"Job {status}: {error}")

        time.sleep(poll_interval)

    raise TimeoutError(f"Job did not complete within {timeout}s")


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    if mask1 is None or mask2 is None:
        return 0.0

    m1 = (mask1 > 127).astype(np.uint8)
    m2 = (mask2 > 127).astype(np.uint8)

    if m1.shape != m2.shape:
        m2 = cv2.resize(m2, (m1.shape[1], m1.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 0.0
    return intersection / union


def build_summary(
    rows: list[dict[str, Any]],
    *,
    run_name: str,
    endpoint_id: str,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    success_rows = [row for row in rows if row["status"] == "success"]
    failed_rows = [row for row in rows if row["status"] != "success"]

    def _avg(key: str) -> float | None:
        values = [float(row[key]) for row in success_rows]
        if not values:
            return None
        return float(statistics.fmean(values))

    def _median(key: str) -> float | None:
        values = [float(row[key]) for row in success_rows]
        if not values:
            return None
        return float(statistics.median(values))

    summary: dict[str, Any] = {
        "run_name": run_name,
        "endpoint_id": endpoint_id,
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "success_rows": len(success_rows),
        "failed_rows": len(failed_rows),
        "success_rate": (len(success_rows) / len(rows)) if rows else 0.0,
        "clip_score_mean": _avg("clip_score"),
        "clip_score_median": _median("clip_score"),
        "mask_iou_mean": _avg("mask_iou"),
        "mask_iou_median": _median("mask_iou"),
        "elapsed_seconds_mean": _avg("elapsed_seconds"),
        "elapsed_seconds_median": _median("elapsed_seconds"),
        "failed_samples": [
            {"sample_id": row["sample_id"], "status": row["status"]}
            for row in failed_rows[:20]
        ],
    }
    return summary


def build_markdown_report(summary: dict[str, Any]) -> str:
    def fmt_ratio(value: Any) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.4f}"

    lines = [
        "# RunPod Benchmark Summary",
        "",
        f"- Run name: `{summary['run_name']}`",
        f"- Endpoint ID: `{summary['endpoint_id']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Success rows: `{summary['success_rows']}`",
        f"- Failed rows: `{summary['failed_rows']}`",
        f"- Success rate: `{fmt_ratio(summary['success_rate'])}`",
        f"- Mean CLIP score: `{fmt_ratio(summary['clip_score_mean'])}`",
        f"- Median CLIP score: `{fmt_ratio(summary['clip_score_median'])}`",
        f"- Mean mask IoU: `{fmt_ratio(summary['mask_iou_mean'])}`",
        f"- Median mask IoU: `{fmt_ratio(summary['mask_iou_median'])}`",
        f"- Mean elapsed seconds: `{fmt_ratio(summary['elapsed_seconds_mean'])}`",
        f"- Median elapsed seconds: `{fmt_ratio(summary['elapsed_seconds_median'])}`",
        f"- Manifest: `{summary['manifest_path']}`",
        f"- Output dir: `{summary['output_dir']}`",
    ]

    failed_samples = summary.get("failed_samples") or []
    if failed_samples:
        lines.extend(["", "## Failed Samples", ""])
        for item in failed_samples:
            lines.append(f"- `{item['sample_id']}`: `{item['status']}`")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if not args.api_key or not args.endpoint_id:
        log("Error: RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set.")
        return

    out_run_dir = args.output_dir / args.run_name
    out_run_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir = out_run_dir / "images"
    images_out_dir.mkdir(exist_ok=True)

    manifest_path = resolve_repo_local_path(str(args.manifest_path))
    log(f"Loading dataset from {manifest_path}...")
    if not manifest_path.exists():
        log(f"File not found: {manifest_path}")
        return

    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= args.limit:
                break
            samples.append(json.loads(line.strip()))

    log(
        "Run configuration: "
        f"endpoint={args.endpoint_id}, samples={len(samples)}, job_timeout={args.job_timeout}s, "
        f"poll_interval={args.poll_interval}s, status_log_interval={args.status_log_interval}s"
    )

    csv_path = out_run_dir / f"metrics_{args.run_name}.csv"
    results_summary: list[dict[str, Any]] = []
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sample_id", "clip_score", "mask_iou", "elapsed_seconds", "status"])
        csv_file.flush()

        for i, sample in enumerate(samples):
            sample_id = sample.get("sample_id", f"sample_{i}")
            log(f"[{i + 1}/{len(samples)}] Processing {sample_id}...")

            try:
                source_image_path = resolve_repo_local_path(sample.get("source_image_path", ""))
                mask_path = resolve_repo_local_path(sample.get("mask_path", ""))
                hairstyle_text = sample.get("hairstyle_text", "")
                color_text = sample.get("color_text", "")

                if not source_image_path.exists():
                    status = f"source_not_found:{source_image_path}"
                    log(f"  -> Skipping, source image not found: {source_image_path}")
                    writer.writerow([sample_id, 0.0, 0.0, 0, status])
                    results_summary.append(
                        {
                            "sample_id": sample_id,
                            "clip_score": 0.0,
                            "mask_iou": 0.0,
                            "elapsed_seconds": 0.0,
                            "status": status,
                        }
                    )
                    csv_file.flush()
                    continue

                with open(source_image_path, "rb") as bf:
                    image_b64 = base64.b64encode(bf.read()).decode()

                payload = {
                    "image": image_b64,
                    "hairstyle_text": hairstyle_text,
                    "color_text": color_text,
                    "top_k": 1,
                    "return_base64": True,
                    "mask_refine_mode": "sam2",
                }

                t0 = time.time()
                job_id = submit_job(args.endpoint_id, args.api_key, payload, args.submit_timeout)
                log(f"  -> Submitted RunPod job_id={job_id}")
                output = poll_job(
                    args.endpoint_id,
                    args.api_key,
                    job_id,
                    timeout=args.job_timeout,
                    status_timeout=args.status_timeout,
                    output_timeout=args.output_timeout,
                    poll_interval=args.poll_interval,
                    status_log_interval=args.status_log_interval,
                )
                elapsed = time.time() - t0

                if "error" in output:
                    log(f"  -> API Error: {output['error']}")
                    status = f"error: {output.get('error')}"
                    writer.writerow([sample_id, 0.0, 0.0, elapsed, status])
                    results_summary.append(
                        {
                            "sample_id": sample_id,
                            "clip_score": 0.0,
                            "mask_iou": 0.0,
                            "elapsed_seconds": elapsed,
                            "status": status,
                        }
                    )
                    csv_file.flush()
                    continue

                results = output.get("results", [])
                if not results:
                    log("  -> Execution failure (empty results)")
                    writer.writerow([sample_id, 0.0, 0.0, elapsed, "empty_results"])
                    results_summary.append(
                        {
                            "sample_id": sample_id,
                            "clip_score": 0.0,
                            "mask_iou": 0.0,
                            "elapsed_seconds": elapsed,
                            "status": "empty_results",
                        }
                    )
                    csv_file.flush()
                    continue

                res = results[0]

                gen_mask_b64 = res.get("mask_base64", "")
                mask_iou = 0.0
                if gen_mask_b64 and mask_path.exists():
                    gen_mask_bytes = base64.b64decode(gen_mask_b64)
                    gen_mask_arr = np.frombuffer(gen_mask_bytes, dtype=np.uint8)
                    gen_mask_img = cv2.imdecode(gen_mask_arr, cv2.IMREAD_GRAYSCALE)
                    gt_mask_img = imread_grayscale(mask_path)

                    if gen_mask_img is not None and gt_mask_img is not None:
                        mask_iou = calculate_iou(gen_mask_img, gt_mask_img)

                clip_score = res.get("clip_score", 0.0)

                gen_img_b64 = res.get("image_base64", "")
                if gen_img_b64:
                    out_img_path = images_out_dir / f"{sample_id}.jpg"
                    with open(out_img_path, "wb") as bf:
                        bf.write(base64.b64decode(gen_img_b64))

                log(f"  -> SUCCESS | IoU: {mask_iou:.4f} | CLIP: {clip_score:.4f} | Time: {elapsed:.1f}s")
                writer.writerow([sample_id, clip_score, mask_iou, elapsed, "success"])
                results_summary.append(
                    {
                        "sample_id": sample_id,
                        "clip_score": clip_score,
                        "mask_iou": mask_iou,
                        "elapsed_seconds": elapsed,
                        "status": "success",
                    }
                )
                csv_file.flush()

            except Exception as e:
                log(f"  -> Exception: {e}")
                status = f"exception: {str(e)}"
                writer.writerow([sample_id, 0.0, 0.0, 0, status])
                results_summary.append(
                    {
                        "sample_id": sample_id,
                        "clip_score": 0.0,
                        "mask_iou": 0.0,
                        "elapsed_seconds": 0.0,
                        "status": status,
                    }
                )
                csv_file.flush()

    summary = build_summary(
        results_summary,
        run_name=args.run_name,
        endpoint_id=args.endpoint_id,
        manifest_path=manifest_path,
        output_dir=out_run_dir,
    )
    summary_path = out_run_dir / "summary.json"
    readme_path = out_run_dir / "README.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    readme_path.write_text(build_markdown_report(summary), encoding="utf-8")

    log(f"\nAll tests finished. CSV saved to: {csv_path}")
    log(f"Summary JSON saved to: {summary_path}")
    log(f"Markdown report saved to: {readme_path}")


if __name__ == "__main__":
    main()
