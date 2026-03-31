#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable


@dataclass(frozen=True)
class EnvRule:
    category: str
    note: str
    source: str


RULES: Dict[str, EnvRule] = {
    "HF_HOME": EnvRule("runtime_recommended", "HF cache root for persistent model caching.", "huggingface libraries"),
    "HF_TOKEN": EnvRule("runtime_recommended", "HF auth token for private or gated assets.", "runtime_download.py/loading.py/sam2_runtime.py"),
    "HUGGINGFACE_HUB_TOKEN": EnvRule("runtime_recommended", "Alternate HF auth token name.", "loading.py/sam2_runtime.py"),
    "SAM2_CHECKPOINT_PATH": EnvRule("runtime_recommended", "Persistent SAM2 checkpoint path.", "utils/sam2_runtime.py"),
    "SEGFACE_HF_REPO_ID": EnvRule("runtime_recommended", "Default custom SegFace HF repository.", "pipeline_sd_components/loading.py"),
    "SEGFACE_HF_FILENAME": EnvRule("runtime_recommended", "Default custom SegFace checkpoint filename.", "pipeline_sd_components/loading.py"),
    "MIRRAI_LORA_HF_REPO_ID": EnvRule("runtime_recommended", "Default runtime LoRA source repository.", "pipeline_sd_components/loading.py"),
    "MIRRAI_LORA_HF_FILENAME": EnvRule("runtime_recommended", "Default runtime LoRA weight filename.", "pipeline_sd_components/loading.py"),
    "ENABLE_SAM2": EnvRule("runtime_recommended", "Toggle SAM2 refinement.", "handler_sd.py"),
    "LORA_PATH": EnvRule("runtime_optional", "Override runtime LoRA location.", "handler_sd.py"),
    "LORA_SCALE": EnvRule("runtime_optional", "Override runtime LoRA scale.", "handler_sd.py"),
    "SEGFACE_HF_SUBFOLDER": EnvRule("runtime_optional", "Optional SegFace HF subfolder.", "pipeline_sd_components/loading.py"),
    "SEGFACE_HF_TOKEN": EnvRule("runtime_optional", "Optional dedicated SegFace HF token.", "pipeline_sd_components/loading.py"),
    "SEGFACE_CKPT_PATH": EnvRule("runtime_optional", "Use a local custom SegFace checkpoint path.", "pipeline_sd_components/loading.py"),
    "SEGFACE_MODEL_VARIANT": EnvRule("runtime_optional", "Override custom SegFace architecture.", "pipeline_sd_components/loading.py"),
    "SEGFACE_INPUT_RESOLUTION": EnvRule("runtime_optional", "Override custom SegFace input resolution.", "pipeline_sd_components/loading.py"),
    "SEGFACE_BASE_CKPT_PATH": EnvRule("runtime_optional", "Use a local base SegFace checkpoint path.", "pipeline_sd_components/loading.py"),
    "SEGFACE_BASE_HF_REPO_ID": EnvRule("runtime_optional", "Override base SegFace HF repository.", "pipeline_sd_components/loading.py"),
    "SEGFACE_BASE_HF_SUBFOLDER": EnvRule("runtime_optional", "Override base SegFace HF subfolder.", "pipeline_sd_components/loading.py"),
    "SEGFACE_BASE_HF_FILENAME": EnvRule("runtime_optional", "Override base SegFace filename.", "pipeline_sd_components/loading.py"),
    "SEGFACE_BASE_HF_TOKEN": EnvRule("runtime_optional", "Optional dedicated base SegFace HF token.", "pipeline_sd_components/loading.py"),
    "SEGFACE_BASE_MODEL_VARIANT": EnvRule("runtime_optional", "Override base SegFace architecture.", "pipeline_sd_components/loading.py"),
    "SEGFACE_BASE_INPUT_RESOLUTION": EnvRule("runtime_optional", "Override base SegFace input resolution.", "pipeline_sd_components/loading.py"),
    "SEGFACE_HAIR_THRESHOLD": EnvRule("runtime_optional", "Override SegFace hair threshold after pipeline load.", "handler_sd.py"),
    "SEGFACE_CUSTOM_HAIR_THRESHOLD": EnvRule("runtime_optional", "Override SegFace hair threshold during loader setup.", "pipeline_sd_components/loading.py"),
    "SEGFACE_CROP_SCALE": EnvRule("runtime_optional", "Adjust SegFace crop scale.", "pipeline_sd_inpainting.py"),
    "SEGFACE_CROP_CENTER_Y_OFFSET": EnvRule("runtime_optional", "Adjust SegFace crop center offset.", "pipeline_sd_inpainting.py"),
    "SEGFACE_CUSTOM_HAIR_RATIO_MIN": EnvRule("runtime_optional", "Adjust custom hair ratio fallback.", "pipeline_sd_inpainting.py"),
    "MASK_REFINE_MODE": EnvRule("runtime_optional", "Set default runtime mask refinement mode.", "pipeline_sd_inpainting.py"),
    "MIRRAI_PRELOAD_ON_STARTUP": EnvRule("runtime_optional", "Optional preload behavior toggle.", "handler_sd.py"),
    "MIRRAI_BUILD_TAG": EnvRule("runtime_optional", "Build metadata included in handler output.", "handler_sd.py"),
    "SAM2_HF_REPO_ID": EnvRule("runtime_optional", "Override SAM2 HF repository.", "utils/sam2_runtime.py"),
    "SAM2_HF_FILENAME": EnvRule("runtime_optional", "Override SAM2 HF filename.", "utils/sam2_runtime.py"),
    "SAM2_MODEL_CONFIG": EnvRule("runtime_optional", "Override SAM2 model config file.", "utils/sam2_runtime.py"),
    "LORA_HF_REPO_ID": EnvRule("runtime_optional", "Legacy-compatible LoRA HF repository alias.", "pipeline_sd_components/loading.py"),
    "LORA_HF_FILENAME": EnvRule("runtime_optional", "Legacy-compatible LoRA HF filename alias.", "pipeline_sd_components/loading.py"),
    "LLM_REFINED_TRENDS_PATH": EnvRule("runtime_optional", "Override trend prompt data path.", "utils/trend_prompt.py"),
    "MEDIAPIPE_FACE_LANDMARKER_MODEL": EnvRule("runtime_optional", "Override MediaPipe landmark model path.", "utils/runtime_compat.py"),
    "MEDIAPIPE_AUTO_DOWNLOAD_FACE_LANDMARKER": EnvRule("runtime_optional", "Enable MediaPipe landmark auto-download.", "utils/runtime_compat.py"),
    "MEDIAPIPE_MODEL_CACHE_DIR": EnvRule("runtime_optional", "MediaPipe model cache directory.", "utils/runtime_compat.py"),
    "TORCH_CUDA_ARCH_LIST": EnvRule("runtime_optional", "Optional Torch CUDA arch override.", "utils/runtime_compat.py"),
    "RUNPOD_ENDPOINT_ID": EnvRule("platform_runtime", "Usually injected by RunPod at runtime; local release tools also read it.", "handler_sd.py/scripts"),
    "RUNPOD_POD_ID": EnvRule("platform_runtime", "Usually injected by RunPod at runtime.", "handler_sd.py"),
    "RUNPOD_GPU_TYPE_ID": EnvRule("platform_runtime", "Usually injected by RunPod at runtime.", "handler_sd.py"),
    "RUNPOD_GPU_SIZE": EnvRule("platform_runtime", "Usually injected by RunPod at runtime.", "handler_sd.py"),
    "RUNPOD_HANDLER_FILE": EnvRule("platform_runtime", "Optional RunPod handler entry override.", "entrypoint_sd.sh"),
    "RUNPOD_WEBHOOK_GET_JOB": EnvRule("platform_runtime", "RunPod webhook metadata.", "entrypoint_sd.sh"),
    "RUNPOD_WEBHOOK_PING": EnvRule("platform_runtime", "RunPod webhook metadata.", "entrypoint_sd.sh"),
    "RUNPOD_WEBHOOK_POST_OUTPUT": EnvRule("platform_runtime", "RunPod webhook metadata.", "entrypoint_sd.sh"),
    "RUNPOD_WEBHOOK_POST_STREAM": EnvRule("platform_runtime", "RunPod webhook metadata.", "entrypoint_sd.sh"),
    "RUNPOD_API_KEY": EnvRule("local_tooling", "Used by local release and smoke-test tooling.", "scripts/runpod_release.py/tests/test_runpod.py"),
    "SEG_PTH_GITHUB_OWNER": EnvRule("legacy_unused", "Legacy seg.pth GitHub download owner.", "not used by current runtime"),
    "SEG_PTH_GITHUB_REPO": EnvRule("legacy_unused", "Legacy seg.pth GitHub download repo.", "not used by current runtime"),
    "SEG_PTH_GITHUB_PATH": EnvRule("legacy_unused", "Legacy seg.pth GitHub download path.", "not used by current runtime"),
    "SEG_PTH_GITHUB_REF": EnvRule("legacy_unused", "Legacy seg.pth GitHub download ref.", "not used by current runtime"),
    "SEG_PTH_PATH": EnvRule("legacy_unused", "Legacy seg.pth local path.", "not used by current runtime"),
    "GITHUB_TOKEN": EnvRule("legacy_unused", "Not used by the current SD serverless runtime.", "not used by current runtime"),
    "ENABLE_STARTUP_GIT_PULL": EnvRule("legacy_unused", "Not used by the current SD serverless runtime.", "not used by current runtime"),
    "RUNPOD_DEBUG_LEVEL": EnvRule("legacy_unused", "Not used by the current SD serverless runtime.", "not used by current runtime"),
    "MODEL_DOWNLOAD_TIMEOUT": EnvRule("legacy_unused", "Present in Docker image env, but not consumed by current Python runtime.", "Dockerfile.sd.app"),
    "TORCH_HOME": EnvRule("legacy_unused", "Not read by repo code in the current SD serverless path.", "no direct repository usage"),
}

DISPLAY_ORDER = [
    "runtime_recommended",
    "runtime_optional",
    "platform_runtime",
    "local_tooling",
    "legacy_unused",
    "unknown",
]


def parse_env_file(env_file: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def classify_keys(keys: Iterable[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    grouped: Dict[str, Dict[str, Dict[str, str]]] = {category: {} for category in DISPLAY_ORDER}
    for key in sorted(set(keys)):
        rule = RULES.get(key)
        if rule is None:
            grouped["unknown"][key] = {
                "note": "Not mapped in the current audit rules. Review manually.",
                "source": "unknown",
            }
            continue
        grouped[rule.category][key] = {
            "note": rule.note,
            "source": rule.source,
        }
    return grouped


def render_text(grouped: Dict[str, Dict[str, Dict[str, str]]]) -> str:
    labels = {
        "runtime_recommended": "Runtime: Recommended",
        "runtime_optional": "Runtime: Optional",
        "platform_runtime": "Runtime: Platform metadata",
        "local_tooling": "Local tooling only",
        "legacy_unused": "Legacy or unused by current runtime",
        "unknown": "Unknown",
    }
    lines: list[str] = []
    for category in DISPLAY_ORDER:
        items = grouped.get(category) or {}
        if not items:
            continue
        lines.append(f"[{labels[category]}]")
        for key, meta in items.items():
            lines.append(f"- {key}: {meta['note']} ({meta['source']})")
        lines.append("")
    return "\n".join(lines).rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit RunPod serverless env keys against the current repository runtime."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional dotenv-style file to audit. Defaults to current process env.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.env_file is not None:
        env_values = parse_env_file(args.env_file)
    else:
        env_values = dict(os.environ)

    grouped = classify_keys(env_values.keys())

    if args.format == "json":
        print(json.dumps(grouped, ensure_ascii=False, indent=2, sort_keys=False))
    else:
        print(render_text(grouped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
