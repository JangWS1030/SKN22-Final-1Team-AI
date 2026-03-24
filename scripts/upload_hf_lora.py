#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_DIRS = [
    PROJECT_ROOT / "output" / "serverless_fix16",
    PROJECT_ROOT / "output" / "full_pipeline_bob_mediapipe_v4",
    PROJECT_ROOT / "output" / "full_pipeline_bob_mediapipe_v5",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
COPYABLE_SUFFIXES = {
    *IMAGE_SUFFIXES,
    ".json", ".csv", ".md", ".txt",
}


def discover_default_bundle_dir() -> Path | None:
    candidates = [path for path in PROJECT_ROOT.glob("sd_lora_final_bundle_*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


DEFAULT_BUNDLE_DIR = discover_default_bundle_dir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the trained SD LoRA bundle to a Hugging Face model repo."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face repo id, for example: siik/hair-swap-sd-lora-stage2",
    )
    parser.add_argument(
        "--bundle-dir",
        default=str(DEFAULT_BUNDLE_DIR) if DEFAULT_BUNDLE_DIR else None,
        help=(
            "Path to the recovered LoRA bundle directory. "
            "If omitted, the newest local sd_lora_final_bundle_* directory is used when available."
        ),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="HF access token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--sample-dir",
        action="append",
        default=[],
        help="Extra directory to include under samples/. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sample-file",
        action="append",
        default=[],
        help="Extra file to include under samples/. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-default-samples",
        action="store_true",
        help="Disable the default sample directories bundled with the upload.",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--public",
        action="store_true",
        help="Create or update the repo as public.",
    )
    visibility.add_argument(
        "--private",
        action="store_true",
        help="Create or update the repo as private. Default behavior.",
    )
    parser.add_argument(
        "--include-reports",
        action="store_true",
        default=True,
        help="Include the recovered markdown reports. Default: on.",
    )
    parser.add_argument(
        "--photos-only",
        action="store_true",
        help="Upload only final LoRA-applied result photos from sample dirs/files. Skip weights, summary, and reports.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload SD LoRA bundle",
        help="Commit message for the upload.",
    )
    return parser.parse_args()


def load_project_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def resolve_token(explicit_token: str | None) -> str | None:
    load_project_dotenv()
    token = (
        explicit_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    token = str(token or "").strip().strip('"')
    return token or None


def load_hf_api(token: str):
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run `python -m pip install huggingface_hub` first."
        ) from exc
    return HfApi(token=token)


def resolve_bundle(bundle_dir: Path) -> tuple[Path, Path | None]:
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_dir}")

    if (bundle_dir / "pytorch_lora_weights.safetensors").is_file():
        model_dir = bundle_dir
        reports_dir = None
    else:
        model_dir = bundle_dir / "model"
        reports_dir = bundle_dir / "reports"

    if not (model_dir / "pytorch_lora_weights.safetensors").is_file():
        raise FileNotFoundError(
            f"LoRA weights not found under: {model_dir / 'pytorch_lora_weights.safetensors'}"
        )
    if not (model_dir / "training_summary.json").is_file():
        raise FileNotFoundError(
            f"training_summary.json not found under: {model_dir / 'training_summary.json'}"
        )

    if reports_dir is not None and not reports_dir.is_dir():
        reports_dir = None

    return model_dir, reports_dir


def load_summary(training_summary_path: Path) -> dict[str, Any]:
    with training_summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_readme(
    repo_id: str,
    summary: dict[str, Any],
    *,
    included_samples: list[str],
) -> str:
    cfg = summary.get("config") or {}
    base_model = cfg.get("pretrained_model_name_or_path", "runwayml/stable-diffusion-inpainting")
    controlnet_model = cfg.get("controlnet_model_name_or_path", "lllyasviel/control_v11p_sd15_canny")
    ip_adapter_weight = cfg.get("ip_adapter_weight_name", "ip-adapter-plus-face_sd15.bin")
    best_step = summary.get("best_step")
    best_val_loss = summary.get("best_val_loss")
    global_step = summary.get("global_step")
    rank = cfg.get("rank")
    alpha = cfg.get("lora_alpha")
    dropout = cfg.get("lora_dropout")
    resolution = cfg.get("resolution")
    lr = cfg.get("learning_rate")

    lines = [
        "---",
        "base_model: runwayml/stable-diffusion-inpainting",
        "library_name: diffusers",
        "tags:",
        "- lora",
        "- diffusers",
        "- stable-diffusion",
        "- stable-diffusion-inpainting",
        "- hair-style-transfer",
        "- image-to-image",
        "---",
        "",
        f"# {repo_id}",
        "",
        "Recovered SD Inpainting LoRA bundle for the hair swap generation pipeline.",
        "",
        "## What This Is",
        "",
        "- Trained target: Stable Diffusion Inpainting UNet LoRA",
        "- Frozen during training: ControlNet, IP-Adapter, VAE, text encoder",
        f"- Base model: `{base_model}`",
        f"- ControlNet during training: `{controlnet_model}`",
        f"- IP-Adapter weight during training: `{ip_adapter_weight}`",
        "",
        "## Training Summary",
        "",
        f"- Resolution: `{resolution}`",
        f"- Final global step: `{global_step}`",
        f"- Best step: `{best_step}`",
        f"- Best validation loss: `{best_val_loss}`",
        f"- LoRA rank: `{rank}`",
        f"- LoRA alpha: `{alpha}`",
        f"- LoRA dropout: `{dropout}`",
        f"- Learning rate: `{lr}`",
        "",
        "## Files",
        "",
        "- `pytorch_lora_weights.safetensors`: LoRA weights",
        "- `training_summary.json`: recovered training metadata",
        "- `reports/`: recovered markdown notes and test summaries",
        "- `samples/`: sample outputs, masks, and debug artifacts",
        "",
        "## Diffusers Usage",
        "",
        "```python",
        "from diffusers import StableDiffusionControlNetInpaintPipeline",
        "",
        f"repo_id = \"{repo_id}\"",
        "pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(",
        "    \"runwayml/stable-diffusion-inpainting\",",
        "    safety_checker=None,",
        ")",
        "pipe.load_lora_weights(repo_id, weight_name=\"pytorch_lora_weights.safetensors\", adapter_name=\"hairgen_runtime\")",
        "pipe.set_adapters(\"hairgen_runtime\", 1.0)",
        "```",
        "",
        "## Notes",
        "",
        "- This repo contains the LoRA only, not the full SD checkpoint.",
        "- Downstream use should follow the license and policy constraints of the base model.",
        "",
    ]
    if included_samples:
        sample_lines = [
            "## Included Samples",
            "",
        ]
        sample_lines.extend([f"- `{item}`" for item in included_samples])
        sample_lines.append("")
        lines.extend(sample_lines)
    return "\n".join(lines)


def _iter_copyable_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in COPYABLE_SUFFIXES else []
    if not path.is_dir():
        return []
    files: list[Path] = []
    for child in sorted(path.rglob("*")):
        if child.is_file() and child.suffix.lower() in COPYABLE_SUFFIXES:
            files.append(child)
    return files


def _is_result_photo(path: Path) -> bool:
    suffix = path.suffix.lower()
    name = path.name.lower()
    return suffix in IMAGE_SUFFIXES and "_result_" in name


def _iter_photo_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if _is_result_photo(path) else []
    if not path.is_dir():
        return []
    files: list[Path] = []
    for child in sorted(path.rglob("*")):
        if child.is_file() and _is_result_photo(child):
            files.append(child)
    return files


def _copy_sample_path(src: Path, samples_root: Path, *, photos_only: bool = False) -> list[str]:
    copied: list[str] = []
    if not src.exists():
        return copied

    if src.is_file():
        if photos_only:
            if not _is_result_photo(src):
                return copied
            dst_dir = samples_root / "extra_files"
        else:
            if src.suffix.lower() not in COPYABLE_SUFFIXES:
                return copied
            dst_dir = samples_root / "extra_files"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst.relative_to(samples_root.parent)).replace("\\", "/"))
        return copied

    base_dir = samples_root / src.name
    iter_files = _iter_photo_files(src) if photos_only else _iter_copyable_files(src)
    for file_path in iter_files:
        rel = file_path.relative_to(src)
        dst = base_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dst)
        copied.append(str(dst.relative_to(samples_root.parent)).replace("\\", "/"))
    return copied


def stage_bundle(
    bundle_dir: Path,
    repo_id: str,
    include_reports: bool,
    sample_dirs: list[Path],
    sample_files: list[Path],
    photos_only: bool = False,
) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="hf_lora_upload_"))

    if photos_only:
        samples_root = temp_dir / "photos"
    else:
        model_dir, reports_dir = resolve_bundle(bundle_dir)
        training_summary_path = model_dir / "training_summary.json"
        summary = load_summary(training_summary_path)

        shutil.copy2(model_dir / "pytorch_lora_weights.safetensors", temp_dir / "pytorch_lora_weights.safetensors")
        shutil.copy2(training_summary_path, temp_dir / "training_summary.json")

        if include_reports and reports_dir is not None:
            dst_reports = temp_dir / "reports"
            dst_reports.mkdir(parents=True, exist_ok=True)
            for src in sorted(reports_dir.glob("*.md")):
                shutil.copy2(src, dst_reports / src.name)

        samples_root = temp_dir / "samples"

    copied_samples: list[str] = []
    for sample_dir in sample_dirs:
        copied_samples.extend(_copy_sample_path(sample_dir, samples_root, photos_only=photos_only))
    for sample_file in sample_files:
        copied_samples.extend(_copy_sample_path(sample_file, samples_root, photos_only=photos_only))

    if photos_only:
        if not copied_samples:
            raise FileNotFoundError(
                "No final result photos were found in the selected sample dirs/files."
            )
        return temp_dir

    manifest = {
        "bundle_dir": str(bundle_dir),
        "sample_dirs": [str(path) for path in sample_dirs],
        "sample_files": [str(path) for path in sample_files],
        "copied_sample_paths": copied_samples,
    }
    reports_root = temp_dir / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "upload_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (temp_dir / "README.md").write_text(
        build_readme(repo_id, summary, included_samples=copied_samples),
        encoding="utf-8",
    )

    return temp_dir


def main() -> int:
    args = parse_args()
    token = resolve_token(args.token)
    if not token:
        raise SystemExit("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required.")

    if not args.bundle_dir:
        raise SystemExit("No local LoRA bundle was found. Pass --bundle-dir <path> explicitly.")

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    repo_private = not args.public
    sample_dirs = [] if args.no_default_samples else [path for path in DEFAULT_SAMPLE_DIRS if path.exists()]
    sample_dirs.extend(Path(path).expanduser().resolve() for path in args.sample_dir)
    sample_files = [Path(path).expanduser().resolve() for path in args.sample_file]

    api = load_hf_api(token)
    staged_dir = stage_bundle(
        bundle_dir,
        args.repo_id,
        include_reports=args.include_reports,
        sample_dirs=sample_dirs,
        sample_files=sample_files,
        photos_only=args.photos_only,
    )

    try:
        repo_url = api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=repo_private,
            exist_ok=True,
        )
        print(f"[repo] {repo_url}")
        commit_info = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(staged_dir),
            commit_message=args.commit_message,
        )
        print(f"[upload] {commit_info}")
        print(f"[done] https://huggingface.co/{args.repo_id}")
        return 0
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
