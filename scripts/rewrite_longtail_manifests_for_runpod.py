from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


PATH_FIELDS = (
    "source_image_path",
    "target_image_path",
    "mask_path",
    "face_protect_mask_path",
    "cloth_protect_mask_path",
    "control_image_path",
    "face_crop_path",
)

DATASET_NAMES = (
    "celeba_dialog_hq_generation",
    "face_sketches_refined_generation",
    "male_asian_hairstyles_generation",
    "longtail_training",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite mixed-dataset longtail manifests to RunPod volume absolute paths."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset_build/processed/longtail_training/manifests"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/runpod_bundle/longtail_volume_stub/datasets/longtail_training/manifests"),
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/workspace/datasets",
        help="Root path on the RunPod Pod where datasets are copied.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rewrite_path(raw_path: str, dataset_root: str) -> str:
    normalized = raw_path.replace("\\", "/")
    for dataset_name in DATASET_NAMES:
        marker = f"/{dataset_name}/"
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            return f"{dataset_root.rstrip('/')}/{dataset_name}/{suffix}"
        marker = f"dataset_build/processed/{dataset_name}/"
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            return f"{dataset_root.rstrip('/')}/{dataset_name}/{suffix}"
        if normalized.startswith(f"{dataset_name}/"):
            suffix = normalized[len(dataset_name) + 1 :]
            return f"{dataset_root.rstrip('/')}/{dataset_name}/{suffix}"
    return normalized


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rewritten_files: List[str] = []
    for manifest_path in sorted(input_dir.glob("*.jsonl")):
        rows = load_jsonl(manifest_path)
        rewritten_rows: List[Dict] = []
        for row in rows:
            out = dict(row)
            for field in PATH_FIELDS:
                value = out.get(field)
                if value:
                    out[field] = rewrite_path(str(value), args.dataset_root)
            rewritten_rows.append(out)
        write_jsonl(output_dir / manifest_path.name, rewritten_rows)
        rewritten_files.append(manifest_path.name)

    report = {
        "input_dir": str(input_dir.as_posix()),
        "output_dir": str(output_dir.as_posix()),
        "dataset_root": args.dataset_root,
        "rewritten_files": rewritten_files,
    }
    (output_dir.parent / "rewrite_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
