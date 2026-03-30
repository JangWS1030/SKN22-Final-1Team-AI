from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "runpod_bundle" / "hair_swap_generation_bundle"

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
    "pipeline_sd_components",
    "models",
    "docs",
    "data",
]


def copy_file(relative_path: str) -> None:
    src = REPO_ROOT / relative_path
    dst = OUTPUT_ROOT / relative_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir(relative_path: str) -> None:
    src = REPO_ROOT / relative_path
    dst = OUTPUT_ROOT / relative_path
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_readme() -> None:
    readme = """# RunPod Training Bundle

This folder is the code-only bundle to copy into a RunPod training Pod.

Recommended layout on the Pod:

- code: `/workspace/hair_swap_model`
- dataset: `/runpod-volume/datasets/celeba_dialog_hq_generation`
- outputs: `/runpod-volume/hair_swap_generation/output`

Recommended launch sequence:

```bash
apt-get update && apt-get install -y git tmux unzip
mkdir -p /workspace/hair_swap_model
```

Copy this folder into:

```bash
/workspace/hair_swap_model
```

Then run:

```bash
cd /workspace/hair_swap_model

DATA_ROOT=/runpod-volume/datasets/celeba_dialog_hq_generation \\
WORK_ROOT=/runpod-volume/hair_swap_generation \\
AUTO_RESUME=1 \\
RUN_FULL_RECON_EVAL=0 \\
bash scripts/start_generation_training_safe.sh
```

Status:

```bash
WORK_ROOT=/runpod-volume/hair_swap_generation \\
bash scripts/check_generation_training_status.sh
```

Attach:

```bash
tmux attach -t hairgen_train
```
"""
    (OUTPUT_ROOT / "RUNPOD_COPY_README.md").write_text(readme, encoding="utf-8")


def write_summary() -> None:
    total_files = sum(1 for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    summary = {
        "bundle_root": str(OUTPUT_ROOT),
        "included_files": INCLUDE_FILES,
        "included_dirs": INCLUDE_DIRS,
        "total_files": total_files,
        "total_bytes": total_bytes,
    }
    (OUTPUT_ROOT / "bundle_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for relative_path in INCLUDE_FILES:
        copy_file(relative_path)

    for relative_path in INCLUDE_DIRS:
        copy_dir(relative_path)

    write_readme()
    write_summary()
    print(json.dumps({"bundle_root": str(OUTPUT_ROOT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
