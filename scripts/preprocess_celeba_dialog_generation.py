from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


HAIR_CLASS = 13
HAT_CLASS = 14
CLOTH_CLASS = 18
FACE_CLASSES = {
    1,   # skin
    2,   # nose
    3,   # eye_g
    4,   # l_eye
    5,   # r_eye
    6,   # l_brow
    7,   # r_brow
    8,   # l_ear
    9,   # r_ear
    10,  # mouth
    11,  # u_lip
    12,  # l_lip
    15,  # ear_r
    16,  # neck_l
    17,  # neck
}

HAIR_KEYWORDS = (
    "bang", "fringe", "forehead", "eyebrow", "hair", "hairstyle", "curl", "wavy", "straight"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess CelebA-Dialog HQ into a generation-model-ready hair dataset."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("dataset_build/raw/celeba_dialog_hq"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("dataset_build/processed/celeba_dialog_hq_generation"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate processed files even when they already exist.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_identity_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            mapping[parts[0]] = parts[1]
    return mapping


def choose_split(identity_id: Optional[str], file_name: str) -> str:
    key = identity_id or file_name
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 1000
    if bucket < 960:
        return "train"
    if bucket < 980:
        return "val"
    return "test"


def compute_bbox(mask: np.ndarray, padding_ratio: float = 0.25) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    w = max(1, x2 - x1 + 1)
    h = max(1, y2 - y1 + 1)
    pad_x = int(round(w * padding_ratio))
    pad_y = int(round(h * padding_ratio))
    H, W = mask.shape
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(W, x2 + pad_x + 1),
        min(H, y2 + pad_y + 1),
    )


def save_mask(mask: np.ndarray, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_rgb(rgb: np.ndarray, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    Image.fromarray(rgb, mode="RGB").save(path)


def build_canny(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def find_single_dir(root: Path, suffix_name: str) -> Path:
    candidates = [p for p in root.rglob(suffix_name) if p.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one directory named '{suffix_name}' under {root}, found {len(candidates)}")
    return candidates[0]


def iter_image_files(image_root: Path) -> Iterable[Path]:
    files = sorted(image_root.glob("*.jpg"), key=lambda p: int(p.stem))
    return files


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()

    image_root = find_single_dir(raw_root / "images_hq", "image")
    mask_root = find_single_dir(raw_root / "mask_colorized_hq", "CelebAMask-HQ-mask-color-palette")

    captions = load_json(raw_root / "text_hq" / "captions_hq.json")
    requests = load_json(raw_root / "text_hq" / "request_hq.json")
    request_annotated = load_json(raw_root / "text_hq" / "request_annotated_hq.json")
    identity_map = load_identity_map(raw_root / "metadata" / "identity_hq.txt")

    image_out = out_root / "images_512"
    hair_mask_out = out_root / "masks" / "hair"
    face_mask_out = out_root / "masks" / "face_protect"
    cloth_mask_out = out_root / "masks" / "cloth_protect"
    control_out = out_root / "controls" / "canny"
    face_crop_out = out_root / "face_crops"
    manifest_dir = out_root / "manifests"
    report_dir = out_root / "reports"

    for path in (
        image_out,
        hair_mask_out,
        face_mask_out,
        cloth_mask_out,
        control_out,
        face_crop_out,
        manifest_dir,
        report_dir,
    ):
        ensure_dir(path)

    recon_records: List[Dict] = []
    hair_prompt_records: List[Dict] = []

    accepted_count = 0
    skipped_missing = 0
    skipped_quality = 0
    hair_request_count = 0

    image_files = list(iter_image_files(image_root))
    if args.limit is not None:
        image_files = image_files[: args.limit]

    for image_path in tqdm(image_files, desc="preprocess"):
        file_name = image_path.name
        stem = image_path.stem
        mask_path = mask_root / f"{stem}.png"
        if not mask_path.is_file():
            skipped_missing += 1
            continue

        image_rgb = np.array(Image.open(image_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS))
        parsing = np.array(Image.open(mask_path))
        if parsing.ndim != 2:
            parsing = np.array(Image.open(mask_path).convert("P"))

        hair_mask = (parsing == HAIR_CLASS).astype(np.uint8)
        face_mask = np.isin(parsing, list(FACE_CLASSES)).astype(np.uint8)
        cloth_mask = (parsing == CLOTH_CLASS).astype(np.uint8)
        hat_mask = (parsing == HAT_CLASS).astype(np.uint8)

        hair_ratio = float(hair_mask.mean())
        face_ratio = float(face_mask.mean())
        cloth_ratio = float(cloth_mask.mean())
        hat_ratio = float(hat_mask.mean())

        if hair_ratio < 0.015 or hair_ratio > 0.55 or face_ratio < 0.06 or face_ratio > 0.55 or hat_ratio > 0.02:
            skipped_quality += 1
            continue

        face_bbox = compute_bbox(face_mask, padding_ratio=0.25)
        if face_bbox is None:
            skipped_quality += 1
            continue

        x1, y1, x2, y2 = face_bbox
        face_crop = image_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            skipped_quality += 1
            continue

        canny_rgb = build_canny(image_rgb)

        processed_image_path = image_out / f"{stem}.png"
        processed_hair_mask_path = hair_mask_out / f"{stem}.png"
        processed_face_mask_path = face_mask_out / f"{stem}.png"
        processed_cloth_mask_path = cloth_mask_out / f"{stem}.png"
        processed_control_path = control_out / f"{stem}.png"
        processed_face_crop_path = face_crop_out / f"{stem}.png"

        save_rgb(image_rgb, processed_image_path, overwrite=args.overwrite)
        save_mask(hair_mask, processed_hair_mask_path, overwrite=args.overwrite)
        save_mask(face_mask, processed_face_mask_path, overwrite=args.overwrite)
        save_mask(cloth_mask, processed_cloth_mask_path, overwrite=args.overwrite)
        save_rgb(canny_rgb, processed_control_path, overwrite=args.overwrite)
        save_rgb(face_crop, processed_face_crop_path, overwrite=args.overwrite)

        caption_info = captions.get(file_name, {})
        overall_caption = str(caption_info.get("overall_caption") or "").strip()
        attribute_captions = caption_info.get("attribute_wise_captions") or {}
        request_text = str(requests.get(file_name) or "").strip()
        request_meta = request_annotated.get(file_name) or {}
        identity_id = identity_map.get(file_name)
        split = choose_split(identity_id, file_name)

        recon_record = {
            "sample_id": f"celeba_dialog_{stem}_recon_hair",
            "sample_type": "recon_hair",
            "split": split,
            "source_name": "celeba_dialog_hq",
            "identity_id": identity_id,
            "source_image_path": str(processed_image_path.as_posix()),
            "target_image_path": str(processed_image_path.as_posix()),
            "mask_path": str(processed_hair_mask_path.as_posix()),
            "face_protect_mask_path": str(processed_face_mask_path.as_posix()),
            "cloth_protect_mask_path": str(processed_cloth_mask_path.as_posix()),
            "control_image_path": str(processed_control_path.as_posix()),
            "face_crop_path": str(processed_face_crop_path.as_posix()),
            "caption_target": overall_caption or "portrait photo, natural hairstyle",
            "caption_source": overall_caption or None,
            "request_text": request_text or None,
            "request_attribute": request_meta.get("attribute"),
            "is_hair_related_request": bool(
                str(request_meta.get("attribute") or "").lower() == "bangs"
                or any(keyword in request_text.lower() for keyword in HAIR_KEYWORDS)
            ),
            "quality_score": round(1.0 - max(0.0, hat_ratio * 4.0), 4),
            "hair_ratio": round(hair_ratio, 6),
            "face_ratio": round(face_ratio, 6),
            "cloth_ratio": round(cloth_ratio, 6),
            "hat_ratio": round(hat_ratio, 6),
            "face_bbox_512": [int(x1), int(y1), int(x2), int(y2)],
            "raw_labels": {
                "attribute_wise_captions": attribute_captions,
                "request_annotated": request_meta,
            },
        }
        recon_records.append(recon_record)
        accepted_count += 1

        if recon_record["is_hair_related_request"]:
            hair_request_count += 1
            hair_prompt_records.append(
                {
                    "sample_id": f"celeba_dialog_{stem}_hair_prompt",
                    "split": split,
                    "source_name": "celeba_dialog_hq",
                    "image_path": str(processed_image_path.as_posix()),
                    "identity_id": identity_id,
                    "caption_source": overall_caption or None,
                    "request_text": request_text or None,
                    "request_attribute": request_meta.get("attribute"),
                    "request_mode": request_meta.get("request_mode"),
                    "target_score": request_meta.get("target_score"),
                    "hair_ratio": round(hair_ratio, 6),
                    "face_ratio": round(face_ratio, 6),
                }
            )

    recon_manifest = manifest_dir / "recon_hair.jsonl"
    prompt_manifest = manifest_dir / "hair_prompt_bank.jsonl"
    recon_manifest.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in recon_records) + ("\n" if recon_records else ""),
        encoding="utf-8",
    )
    prompt_manifest.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in hair_prompt_records) + ("\n" if hair_prompt_records else ""),
        encoding="utf-8",
    )

    summary = {
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "total_images_seen": len(image_files),
        "accepted_recon_samples": accepted_count,
        "skipped_missing_files": skipped_missing,
        "skipped_quality": skipped_quality,
        "hair_related_prompt_records": hair_request_count,
        "notes": [
            "recon_hair is directly usable for generation-model reconstruction training",
            "hair_prompt_bank is kept for later pseudo-labeling / evaluation / prompt mining",
            "CelebA-Dialog HQ official requests are mostly face attributes; only a subset is hair-related",
        ],
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
