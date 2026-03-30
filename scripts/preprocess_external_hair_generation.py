from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.face_parsing.model import BiSeNet


HAIR_CLASS = 10
HAT_CLASS = 11
CLOTH_CLASS = 15
FACE_CLASSES = {
    1,   # skin
    2,   # nose
    3,   # eye_g
    4,   # eyes
    5,   # brows
    6,   # ears
    7,   # mouth
    8,   # u_lip
    9,   # l_lip
    12,  # ear_r
    13,  # neck_l
    14,  # neck
}

STYLE_ALIASES = {
    "Afro": "afro",
    "Bald": "bald",
    "BobHair": "bob hair",
    "BowlCut": "bowl cut",
    "Bun": "bun",
    "CombOver": "comb over",
    "CornRows": "cornrows",
    "CrewCut": "crew cut",
    "Crop": "crop cut",
    "CurtainedHair": "curtain hairstyle",
    "DreadLocks": "dreadlocks",
    "EmoHair": "emo hair",
    "Fauxhawk": "fauxhawk",
    "Flattop": "flattop",
    "FrenchTwist": "french twist",
    "HimeCut": "hime cut",
    "HiTopFade": "hi-top fade",
    "HorseShoeFlattop": "horseshoe flattop",
    "LayeredHair": "layered hair",
    "LibertySpikes": "liberty spikes",
    "MedLenHair": "medium length hair",
    "Mohawk": "mohawk",
    "MopTop": "mop top",
    "Mullet": "mullet",
    "Odango": "odango buns",
    "Perm": "perm",
    "PixieCut": "pixie cut",
    "PonyTail": "ponytail",
    "RazorCut": "razor cut",
    "Ringlet": "ringlet curls",
    "shag": "shag haircut",
    "ShoulderLenHair": "shoulder length hair",
    "SpikyHair": "spiky hair",
    "UndercutCurly": "curly undercut",
    "UndercutLong": "long undercut",
    "UndercutPompadour": "undercut pompadour",
    "undercutSidepart": "undercut side part",
    "UndercutSlickback": "undercut slick back",
    "WaistLenHair": "waist length hair",
    "WaveHair": "wavy hair",
    "long_hairstyles": "long hairstyle",
    "medium_hairstyles": "medium hairstyle",
    "short_hairstyles": "short hairstyle",
}

STYLE_FAMILY_MAP = {
    "Afro": "coily_protective",
    "CornRows": "coily_protective",
    "DreadLocks": "coily_protective",
    "HiTopFade": "coily_protective",
    "WaveHair": "coily_protective",
    "Ringlet": "coily_protective",
    "Fauxhawk": "punk_editorial",
    "Mohawk": "punk_editorial",
    "LibertySpikes": "punk_editorial",
    "HorseShoeFlattop": "punk_editorial",
    "Flattop": "punk_editorial",
    "SpikyHair": "punk_editorial",
    "FrenchTwist": "retro_structured",
    "Odango": "retro_structured",
    "Bun": "retro_structured",
    "UndercutPompadour": "retro_structured",
    "HimeCut": "alt_asian_longtail",
    "EmoHair": "alt_asian_longtail",
    "CurtainedHair": "alt_asian_longtail",
    "Mullet": "alt_asian_longtail",
    "shag": "alt_asian_longtail",
    "Perm": "alt_asian_longtail",
    "ShoulderLenHair": "garment_reveal_support",
    "WaistLenHair": "garment_reveal_support",
    "MedLenHair": "garment_reveal_support",
    "LayeredHair": "garment_reveal_support",
    "PonyTail": "garment_reveal_support",
    "long_hairstyles": "garment_reveal_support",
    "medium_hairstyles": "garment_reveal_support",
}

LONG_HAIR_TOKENS = {
    "shoulder length hair",
    "waist length hair",
    "long hairstyle",
    "medium hairstyle",
    "medium length hair",
    "layered hair",
    "ponytail",
    "wavy hair",
    "ringlet curls",
    "hime cut",
}

RARITY_BOOSTS = {
    "coily_protective": 2.7,
    "punk_editorial": 2.4,
    "retro_structured": 2.0,
    "alt_asian_longtail": 1.9,
    "garment_reveal_support": 1.4,
    "default": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess external hairstyle image folders into generation-model manifests."
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Root directory of the raw source dataset.")
    parser.add_argument("--out-root", type=Path, required=True, help="Processed output root.")
    parser.add_argument("--source-name", type=str, required=True, help="Logical source identifier.")
    parser.add_argument(
        "--seg-checkpoint",
        type=Path,
        default=Path("pretrained_models/seg.pth"),
        help="BiSeNet face parsing checkpoint path.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--resolution", type=int, default=512, help="Output square resolution.")
    parser.add_argument("--min-face-ratio", type=float, default=0.04)
    parser.add_argument("--max-face-ratio", type=float, default=0.68)
    parser.add_argument("--min-hair-ratio", type=float, default=0.012)
    parser.add_argument("--max-hair-ratio", type=float, default=0.70)
    parser.add_argument("--max-hat-ratio", type=float, default=0.08)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def choose_split(source_name: str, relpath: str) -> str:
    relpath_lower = relpath.lower()
    if "/holdout/" in relpath_lower or relpath_lower.startswith("holdout/"):
        return "test"
    if "/reference_only/" in relpath_lower or relpath_lower.startswith("reference_only/"):
        return "val"
    if "/core/" in relpath_lower or relpath_lower.startswith("core/"):
        return "train"
    digest = hashlib.md5(f"{source_name}:{relpath}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 1000
    if bucket < 920:
        return "train"
    if bucket < 970:
        return "val"
    return "test"


def infer_label_from_path(source_root: Path, image_path: Path) -> Tuple[str, str]:
    rel_parts = image_path.relative_to(source_root).parts
    source_split = "unspecified"
    for part in rel_parts:
        lowered = part.lower()
        if lowered in {"core", "reference_only", "holdout", "train", "val", "validation", "test"}:
            source_split = lowered
            break

    parent_name = image_path.parent.name
    if parent_name.lower() == "image" and len(rel_parts) >= 2:
        parent_name = rel_parts[-2]
    return source_split, parent_name


def humanize_label(label: str) -> str:
    alias = STYLE_ALIASES.get(label)
    if alias:
        return alias
    text = label.replace("_", " ").replace("-", " ")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def infer_style_family(label: str, style_text: str) -> str:
    if label in STYLE_FAMILY_MAP:
        return STYLE_FAMILY_MAP[label]
    if style_text in LONG_HAIR_TOKENS:
        return "garment_reveal_support"
    if any(token in style_text for token in ("mohawk", "liberty spikes", "fauxhawk", "spiky")):
        return "punk_editorial"
    if any(token in style_text for token in ("afro", "dread", "cornrow", "ringlet", "coily")):
        return "coily_protective"
    return "default"


def build_prompt(style_text: str, family: str) -> str:
    parts = [
        "portrait photo",
        "realistic hairstyle",
        "preserved identity",
        style_text,
    ]
    if family == "coily_protective":
        parts.append("natural texture")
    elif family == "punk_editorial":
        parts.append("defined silhouette")
    elif family == "retro_structured":
        parts.append("structured styling")
    elif family == "garment_reveal_support":
        parts.append("visible neckline and clothing")
    return ", ".join(parts)


def iter_images(source_root: Path) -> Iterable[Path]:
    for path in sorted(source_root.rglob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and path.is_file():
            yield path


def load_parser(checkpoint_path: Path, device: str) -> BiSeNet:
    model = BiSeNet(16)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    return model


@torch.inference_mode()
def predict_parsing(model: BiSeNet, image_rgb: np.ndarray, device: str) -> np.ndarray:
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor * 2.0 - 1.0
    tensor = tensor.unsqueeze(0).to(device)
    logits, _ = model(tensor)
    parsing = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return parsing


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    out_root = args.out_root.resolve()

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

    parser = load_parser(args.seg_checkpoint.resolve(), args.device)
    records: List[Dict] = []
    accepted_count = 0
    skipped_quality = 0
    skipped_face = 0
    skipped_small = 0
    family_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()

    image_paths = list(iter_images(source_root))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    for image_path in tqdm(image_paths, desc=f"preprocess:{args.source_name}"):
        relpath = image_path.relative_to(source_root).as_posix()
        source_split, label = infer_label_from_path(source_root, image_path)

        img = Image.open(image_path).convert("RGB")
        if min(img.size) < 128:
            skipped_small += 1
            continue
        img_rgb = np.array(img.resize((args.resolution, args.resolution), Image.Resampling.LANCZOS))

        parsing = predict_parsing(parser, img_rgb, args.device)
        hair_mask = (parsing == HAIR_CLASS).astype(np.uint8)
        face_mask = np.isin(parsing, list(FACE_CLASSES)).astype(np.uint8)
        cloth_mask = (parsing == CLOTH_CLASS).astype(np.uint8)
        hat_mask = (parsing == HAT_CLASS).astype(np.uint8)

        hair_ratio = float(hair_mask.mean())
        face_ratio = float(face_mask.mean())
        cloth_ratio = float(cloth_mask.mean())
        hat_ratio = float(hat_mask.mean())

        if (
            hair_ratio < args.min_hair_ratio
            or hair_ratio > args.max_hair_ratio
            or face_ratio < args.min_face_ratio
            or face_ratio > args.max_face_ratio
            or hat_ratio > args.max_hat_ratio
        ):
            skipped_quality += 1
            continue

        face_bbox = compute_bbox(face_mask, padding_ratio=0.25)
        if face_bbox is None:
            skipped_face += 1
            continue
        x1, y1, x2, y2 = face_bbox
        face_crop = img_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            skipped_face += 1
            continue

        canny_rgb = build_canny(img_rgb)
        stem = hashlib.md5(f"{args.source_name}:{relpath}".encode("utf-8")).hexdigest()[:16]
        processed_image_path = image_out / f"{stem}.png"
        processed_hair_mask_path = hair_mask_out / f"{stem}.png"
        processed_face_mask_path = face_mask_out / f"{stem}.png"
        processed_cloth_mask_path = cloth_mask_out / f"{stem}.png"
        processed_control_path = control_out / f"{stem}.png"
        processed_face_crop_path = face_crop_out / f"{stem}.png"

        save_rgb(img_rgb, processed_image_path, overwrite=args.overwrite)
        save_mask(hair_mask, processed_hair_mask_path, overwrite=args.overwrite)
        save_mask(face_mask, processed_face_mask_path, overwrite=args.overwrite)
        save_mask(cloth_mask, processed_cloth_mask_path, overwrite=args.overwrite)
        save_rgb(canny_rgb, processed_control_path, overwrite=args.overwrite)
        save_rgb(face_crop, processed_face_crop_path, overwrite=args.overwrite)

        style_text = humanize_label(label)
        style_family = infer_style_family(label, style_text)
        split = choose_split(args.source_name, relpath if source_split == "unspecified" else f"{source_split}/{relpath}")
        garment_reveal_candidate = bool(style_text in LONG_HAIR_TOKENS or cloth_ratio >= 0.06 and style_family == "garment_reveal_support")

        record = {
            "sample_id": f"{args.source_name}_{stem}",
            "sample_type": "recon_hair_external",
            "split": split,
            "source_name": args.source_name,
            "identity_id": None,
            "source_image_path": str(processed_image_path.as_posix()),
            "target_image_path": str(processed_image_path.as_posix()),
            "mask_path": str(processed_hair_mask_path.as_posix()),
            "face_protect_mask_path": str(processed_face_mask_path.as_posix()),
            "cloth_protect_mask_path": str(processed_cloth_mask_path.as_posix()),
            "control_image_path": str(processed_control_path.as_posix()),
            "face_crop_path": str(processed_face_crop_path.as_posix()),
            "caption_target": build_prompt(style_text, style_family),
            "caption_source": build_prompt(style_text, style_family),
            "caption_target_hair_enriched": build_prompt(style_text, style_family),
            "train_prompt": build_prompt(style_text, style_family),
            "request_text": None,
            "request_attribute": None,
            "is_hair_related_request": True,
            "quality_score": round(1.0 - max(0.0, hat_ratio * 2.5), 4),
            "sample_weight": round(RARITY_BOOSTS.get(style_family, RARITY_BOOSTS["default"]), 4),
            "hair_ratio": round(hair_ratio, 6),
            "face_ratio": round(face_ratio, 6),
            "cloth_ratio": round(cloth_ratio, 6),
            "hat_ratio": round(hat_ratio, 6),
            "face_bbox_512": [int(x1), int(y1), int(x2), int(y2)],
            "hairstyle_id": re.sub(r"[^a-z0-9]+", "_", style_text).strip("_") or "unknown",
            "hairstyle_text": style_text,
            "style_domain": "longtail_external",
            "style_family": style_family,
            "style_confidence": 0.98 if style_family != "default" else 0.75,
            "style_condition_source": "external_folder_label",
            "source_relative_path": relpath,
            "source_split": source_split,
            "garment_reveal_candidate": garment_reveal_candidate,
            "raw_labels": {
                "folder_label": label,
                "source_split": source_split,
            },
        }
        records.append(record)
        accepted_count += 1
        family_counter[style_family] += 1
        split_counter[split] += 1
        class_counter[label] += 1

    manifest_path = manifest_dir / "recon_hair_external.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )

    summary = {
        "source_root": str(source_root.as_posix()),
        "out_root": str(out_root.as_posix()),
        "source_name": args.source_name,
        "total_images_seen": len(image_paths),
        "accepted_recon_samples": accepted_count,
        "skipped_quality": skipped_quality,
        "skipped_face": skipped_face,
        "skipped_small_images": skipped_small,
        "split_counts": dict(split_counter),
        "style_family_counts": dict(family_counter),
        "top_labels": class_counter.most_common(20),
        "notes": [
            "External rows are self-reconstruction samples with folder-label prompts.",
            "BiSeNet face parsing masks are used to produce hair/face/cloth masks.",
            "garment_reveal_candidate marks rows that are suitable for long-to-short clothing reveal fine-tuning.",
        ],
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
