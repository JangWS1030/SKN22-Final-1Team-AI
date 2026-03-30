from __future__ import annotations

import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "runpod_bundle" / "hair_swap_longtail_training_bundle"
WORKSPACE_ROOT = OUTPUT_ROOT / "workspace" / "hair_swap_model"
VOLUME_ROOT = OUTPUT_ROOT / "workspace" / "datasets" / "longtail_training"

INCLUDE_FILES = [
    "pipeline_sd_inpainting.py",
    "handler_sd.py",
    "runtime_download.py",
    "requirements.txt",
    "requirements-train.txt",
    "README.md",
    "README_runpod_volume.md",
]

INCLUDE_DIRS = [
    "configs",
    "scripts",
    "utils",
    "models",
    "docs",
    "data",
]

INCLUDE_PATHS = [
    "pretrained_models/generation_lora_stage2_best",
    "dataset_build/configs/longtail_hair_sources.json",
    "dataset_build/processed/longtail_training/reports/longtail_training_summary.json",
    "dataset_build/processed/face_sketches_refined_generation/reports/summary.json",
    "dataset_build/processed/male_asian_hairstyles_generation/reports/summary.json",
]


def copy_file(relative_path: str, dst_root: Path) -> None:
    src = REPO_ROOT / relative_path
    dst = dst_root / relative_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir(relative_path: str, dst_root: Path) -> None:
    src = REPO_ROOT / relative_path
    dst = dst_root / relative_path
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def rewrite_manifest_path(raw_path: str, dataset_root: str) -> str:
    normalized = raw_path.replace("\\", "/")
    for dataset_name in (
        "celeba_dialog_hq_generation",
        "face_sketches_refined_generation",
        "male_asian_hairstyles_generation",
        "longtail_training",
    ):
        marker = f"/{dataset_name}/"
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            return f"{dataset_root.rstrip('/')}/{dataset_name}/{suffix}"
        marker = f"dataset_build/processed/{dataset_name}/"
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            return f"{dataset_root.rstrip('/')}/{dataset_name}/{suffix}"
    return normalized


def write_rewritten_jsonl_dir(src_dir: Path, dst_dir: Path, dataset_root: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    path_fields = (
        "source_image_path",
        "target_image_path",
        "mask_path",
        "face_protect_mask_path",
        "cloth_protect_mask_path",
        "control_image_path",
        "face_crop_path",
    )

    for jsonl_path in src_dir.glob("*.jsonl"):
        rows = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out = dict(row)
                for field in path_fields:
                    value = out.get(field)
                    if value:
                        out[field] = rewrite_manifest_path(str(value), dataset_root)
                rows.append(out)
        with (dst_dir / jsonl_path.name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_runpod_manifests(dataset_root: str = "/workspace/datasets") -> None:
    src_dir = REPO_ROOT / "dataset_build" / "processed" / "longtail_training" / "manifests"
    dst_dir = VOLUME_ROOT / "manifests"
    write_rewritten_jsonl_dir(src_dir, dst_dir, dataset_root)

    report_src = REPO_ROOT / "dataset_build" / "processed" / "longtail_training" / "reports"
    report_dst = VOLUME_ROOT / "reports"
    if report_dst.exists():
        shutil.rmtree(report_dst)
    shutil.copytree(report_src, report_dst)

    benchmark_src = REPO_ROOT / "dataset_build" / "processed" / "longtail_training" / "benchmarks"
    benchmark_dst = VOLUME_ROOT / "benchmarks"
    if benchmark_src.exists():
        if benchmark_dst.exists():
            shutil.rmtree(benchmark_dst)
        write_rewritten_jsonl_dir(benchmark_src, benchmark_dst, dataset_root)


def write_readme() -> None:
    readme = """# RunPod Long-tail Training Bundle

This bundle is prepared for a RunPod Pod with a Network Volume mounted at `/workspace`.

Copy order:

1. Copy `workspace/hair_swap_model` into `/workspace/hair_swap_model`
2. Copy `workspace/datasets/longtail_training` into `/workspace/datasets/longtail_training`
3. Copy the processed dataset folders below into `/workspace/datasets/`

Required dataset folders on the Pod:

- `/workspace/datasets/celeba_dialog_hq_generation`
- `/workspace/datasets/face_sketches_refined_generation`
- `/workspace/datasets/longtail_training`

Optional support dataset:

- `/workspace/datasets/male_asian_hairstyles_generation`

Recommended launch sequence:

```bash
apt-get update && apt-get install -y git tmux unzip
mkdir -p /workspace/hair_swap_model /workspace/datasets
```

Launch:

```bash
cd /workspace/hair_swap_model
WORK_ROOT=/workspace/hair_swap_generation \\
DATASET_ROOT=/workspace/datasets \\
AUTO_RESUME=1 \\
RUN_STAGE4=1 \\
bash scripts/start_generation_training_longtail_safe.sh
```

Attach:

```bash
tmux attach -t hairgen_longtail_train
```

Current training flow:

- Stage 3: rare long-tail continuation from the included stage2 LoRA
- Stage 4: garment reveal continuation from the stage3 best LoRA

Documentation-ready benchmark files:

- `/workspace/datasets/longtail_training/benchmarks/rare_style_showcase_eval.jsonl`
- `/workspace/datasets/longtail_training/benchmarks/garment_reveal_showcase_eval.jsonl`
- `/workspace/datasets/longtail_training/benchmarks/smoke_longtail_eval_quick.jsonl`
"""
    (OUTPUT_ROOT / "RUNPOD_COPY_README.md").write_text(readme, encoding="utf-8")


def write_summary() -> None:
    total_files = sum(1 for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    summary = {
        "bundle_root": str(OUTPUT_ROOT),
        "workspace_root": str(WORKSPACE_ROOT),
        "volume_root": str(VOLUME_ROOT),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "required_volume_datasets": [
            "/workspace/datasets/celeba_dialog_hq_generation",
            "/workspace/datasets/face_sketches_refined_generation",
            "/workspace/datasets/longtail_training",
        ],
        "optional_volume_datasets": [
            "/workspace/datasets/male_asian_hairstyles_generation",
        ],
    }
    (OUTPUT_ROOT / "bundle_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for relative_path in INCLUDE_FILES:
        copy_file(relative_path, WORKSPACE_ROOT)

    for relative_path in INCLUDE_DIRS:
        copy_dir(relative_path, WORKSPACE_ROOT)

    for relative_path in INCLUDE_PATHS:
        src = REPO_ROOT / relative_path
        dst = WORKSPACE_ROOT / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    write_runpod_manifests(dataset_root="/workspace/datasets")
    write_readme()
    write_summary()
    print(json.dumps({"bundle_root": str(OUTPUT_ROOT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
