from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_sd_inpainting import MirrAISDPipeline, SDInpaintConfig
from utils.manifest_paths import normalize_manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hairstyle benchmark inference with the local SD pipeline.")
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        required=True,
        help="JSONL benchmark manifest built by scripts/build_generation_benchmarks.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write predictions and metadata.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=7.2)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=0.24)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.32)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--lora-scale", type=float, default=0.9)
    parser.add_argument("--return-intermediates", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return normalize_manifest_rows(rows, manifest_path=path)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def compose_hairstyle_text(row: Dict) -> str:
    parts: List[str] = []
    hairstyle_text = str(row.get("hairstyle_text") or row.get("hairstyle_id") or "").strip()
    if hairstyle_text:
        parts.append(hairstyle_text)

    texture = str(row.get("hair_texture") or "").strip().lower()
    if texture and texture not in {"unknown", "mixed"}:
        parts.append(f"{texture} texture")

    bangs = str(row.get("bangs") or "").strip().lower()
    if bangs and bangs not in {"unknown", "none"}:
        parts.append(f"{bangs} bangs")

    return ", ".join(dict.fromkeys(parts))


def save_debug_images(debug_dir: Path, sample_id: str, result) -> List[str]:
    saved_paths: List[str] = []
    if not result.debug_images:
        return saved_paths
    sample_dir = debug_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    for name, image_bgr in result.debug_images.items():
        out_path = sample_dir / f"{name}.png"
        cv2.imwrite(str(out_path), image_bgr)
        saved_paths.append(str(out_path))
    return saved_paths


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.benchmark_manifest)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    output_dir = args.output_dir
    image_dir = output_dir / "images"
    debug_dir = output_dir / "debug"
    image_dir.mkdir(parents=True, exist_ok=True)

    config = SDInpaintConfig(
        device=args.device,
        dtype=args.dtype,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        enable_xformers=True,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
    )
    pipeline = MirrAISDPipeline(config)
    pipeline.load()

    predictions: List[Dict] = []
    for index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        image = cv2.imread(str(row["source_image_path"]))
        if image is None:
            raise FileNotFoundError(f"Could not read source image for {sample_id}: {row['source_image_path']}")

        seed = stable_seed(args.seed + index, sample_id)
        pipeline.config.seeds = [seed]
        result = pipeline.run(
            image=image,
            hairstyle_text=compose_hairstyle_text(row),
            color_text=str(row.get("color_text") or ""),
            top_k=1,
            return_intermediates=args.return_intermediates,
            lora_path=args.lora_path,
            lora_scale=args.lora_scale,
        )[0]

        prediction_path = image_dir / f"{sample_id}.png"
        cv2.imwrite(str(prediction_path), result.image)
        debug_paths = save_debug_images(debug_dir, sample_id, result) if args.return_intermediates else []

        predictions.append(
            {
                **row,
                "prediction_image_path": str(prediction_path),
                "inference_seed": int(result.seed),
                "clip_score": float(result.clip_score),
                "mask_used": result.mask_used,
                "debug_image_paths": debug_paths,
            }
        )

    write_jsonl(output_dir / "predictions.jsonl", predictions)
    summary = {
        "benchmark_manifest": str(args.benchmark_manifest),
        "output_dir": str(output_dir),
        "rows": len(predictions),
        "lora_path": args.lora_path,
        "lora_scale": args.lora_scale,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
        "ip_adapter_scale": args.ip_adapter_scale,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
