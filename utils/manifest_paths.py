from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional


PATH_FIELDS = (
    "source_image_path",
    "target_image_path",
    "mask_path",
    "face_protect_mask_path",
    "cloth_protect_mask_path",
    "control_image_path",
    "face_crop_path",
    "prediction_image_path",
)


def infer_dataset_root(manifest_path: Path) -> Path:
    return manifest_path.resolve().parent.parent


def normalize_manifest_rows(
    rows: List[Dict],
    *,
    manifest_path: Path,
    path_fields: Optional[Iterable[str]] = None,
) -> List[Dict]:
    dataset_root = infer_dataset_root(manifest_path)
    normalized_rows: List[Dict] = []
    fields = tuple(path_fields or PATH_FIELDS)
    for row in rows:
        normalized = dict(row)
        for field in fields:
            value = normalized.get(field)
            if value:
                normalized[field] = normalize_dataset_path(str(value), dataset_root)
        normalized_rows.append(normalized)
    return normalized_rows


def normalize_dataset_path(raw_path: str, dataset_root: Path) -> str:
    candidate = Path(raw_path)
    if candidate.exists():
        return str(candidate)

    normalized = raw_path.replace("\\", "/")
    dataset_name = dataset_root.name
    marker = f"/{dataset_name}/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        return str((dataset_root / Path(suffix)).resolve())

    if normalized.startswith(f"{dataset_name}/"):
        suffix = normalized[len(dataset_name) + 1 :]
        return str((dataset_root / Path(suffix)).resolve())

    relative_candidate = dataset_root / Path(normalized)
    if relative_candidate.exists():
        return str(relative_candidate.resolve())

    return str(candidate)
