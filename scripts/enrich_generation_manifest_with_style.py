from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


COLOR_TARGETS: List[Tuple[str, Tuple[int, int, int]]] = [
    ("ash beige", (173, 158, 136)),
    ("ash brown", (111, 92, 80)),
    ("ash blonde", (192, 176, 146)),
    ("ash black", (58, 58, 62)),
    ("ash gray", (124, 128, 134)),
    ("black", (44, 41, 39)),
    ("dark brown", (82, 62, 50)),
    ("brown", (98, 74, 58)),
    ("beige", (174, 153, 128)),
    ("blonde", (193, 166, 121)),
    ("silver", (170, 174, 182)),
    ("gray", (132, 132, 132)),
    ("red", (128, 56, 45)),
    ("auburn", (120, 63, 48)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich generation manifest with pseudo hairstyle labels and prompts."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_build/processed/celeba_dialog_hq_generation/manifests/recon_hair.jsonl"),
    )
    parser.add_argument(
        "--trend-data",
        type=Path,
        default=Path("data/llm_refined_trends.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_build/processed/celeba_dialog_hq_generation/manifests/recon_hair_style_enriched.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("dataset_build/processed/celeba_dialog_hq_generation/reports/style_enrichment_summary.json"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional record limit for smoke tests.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def infer_face_bbox(face_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(face_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    height, width = face_mask.shape
    row_counts = face_mask.sum(axis=1).astype(np.float32)
    active_rows = np.where(row_counts > max(4.0, float(row_counts.max()) * 0.08))[0]
    if len(active_rows) == 0:
        active_rows = np.where(row_counts > 0)[0]

    top = float(active_rows.min())
    face_width = max(40.0, float(np.percentile(row_counts[active_rows], 92)))
    face_height = face_width * 1.38

    y1 = max(0, int(round(top - face_height * 0.10)))
    y2 = min(height, int(round(y1 + face_height)))

    center_x = float(xs.mean())
    half_w = face_width * 0.60
    x1 = max(0, int(round(center_x - half_w)))
    x2 = min(width, int(round(center_x + half_w)))
    return x1, y1, x2, y2


def classify_bangs(text: str, hair_mask: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> str:
    value = (text or "").lower()
    if any(token in value for token in ["no fringe", "no bangs", "whole forehead is visible", "full forehead visible", "forehead is visible", "has no fringe"]):
        return "none"
    if any(token in value for token in ["very short bangs", "extremely short bangs", "baby bangs"]):
        return "baby"
    if any(token in value for token in ["covers the eyebrows", "covers the whole forehead", "covers the full forehead", "entire forehead visible"]):
        return "full"
    if any(token in value for token in ["half of the forehead visible", "partially", "part of the forehead", "curtain"]):
        return "curtain"
    if any(token in value for token in ["side bangs", "side-swept"]):
        return "side"

    x1, y1, x2, y2 = face_bbox
    face_h = max(1, y2 - y1)
    forehead_y2 = min(hair_mask.shape[0], int(y1 + face_h * 0.35))
    forehead_area = max(1, (x2 - x1) * max(1, forehead_y2 - y1))
    bangs_ratio = float(hair_mask[y1:forehead_y2, x1:x2].sum()) / float(forehead_area)
    if bangs_ratio < 0.02:
        return "none"
    if bangs_ratio < 0.08:
        return "baby"
    if bangs_ratio < 0.18:
        return "curtain"
    return "full"


def classify_length(hair_bbox: Tuple[int, int, int, int], face_bbox: Tuple[int, int, int, int], hair_ratio: float) -> Tuple[str, Dict[str, float]]:
    hx1, hy1, hx2, hy2 = hair_bbox
    fx1, fy1, fx2, fy2 = face_bbox
    face_h = max(1, fy2 - fy1)
    face_w = max(1, fx2 - fx1)

    down_extent = (hy2 - fy2) / float(face_h)
    up_extent = (fy1 - hy1) / float(face_h)
    side_extent = (max(0, fx1 - hx1) + max(0, hx2 - fx2)) / float(face_w)

    if hair_ratio < 0.10 and down_extent < 0.10 and up_extent > 0.10:
        length = "updo"
    elif down_extent < 0.12 and hair_ratio < 0.22:
        length = "short"
    elif down_extent < 0.42 or hair_ratio < 0.34:
        length = "medium"
    else:
        length = "long"

    features = {
        "down_extent": round(float(down_extent), 4),
        "up_extent": round(float(up_extent), 4),
        "side_extent": round(float(side_extent), 4),
    }
    return length, features


def classify_texture(image_rgb: np.ndarray, hair_mask: np.ndarray) -> Tuple[str, Dict[str, float]]:
    hair_area = float(hair_mask.sum())
    if hair_area <= 1:
        return "unknown", {"edge_density": 0.0, "shape_complexity": 0.0}

    mask_u8 = (hair_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(cnt, True) for cnt in contours))
    shape_complexity = (perimeter * perimeter) / max(hair_area, 1.0)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 160)
    edge_density = float(edges[hair_mask > 0].mean()) / 255.0

    if shape_complexity > 130.0 and edge_density > 0.20:
        texture = "curly"
    elif shape_complexity > 85.0 and edge_density > 0.13:
        texture = "wavy"
    else:
        texture = "straight"

    return texture, {
        "edge_density": round(edge_density, 4),
        "shape_complexity": round(shape_complexity, 2),
    }


def classify_color(image_rgb: np.ndarray, hair_mask: np.ndarray) -> str:
    pixels = image_rgb[hair_mask > 0]
    if len(pixels) == 0:
        return "dark brown"

    mean_rgb = np.mean(pixels.astype(np.float32), axis=0).reshape(1, 1, 3)
    mean_lab = cv2.cvtColor(mean_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]

    best_name = "dark brown"
    best_distance = float("inf")
    for name, target_rgb in COLOR_TARGETS:
        target = np.array(target_rgb, dtype=np.uint8).reshape(1, 1, 3)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
        distance = float(np.linalg.norm(mean_lab - target_lab))
        if distance < best_distance:
            best_distance = distance
            best_name = name
    return best_name


def choose_style(
    length: str,
    texture: str,
    bangs: str,
    asymmetry: float,
    side_extent: float,
    down_extent: float,
) -> Tuple[Optional[str], float]:
    if length == "updo":
        return None, 0.2

    if length == "short":
        if down_extent < -0.15:
            return "textured-crop", 0.76
        if bangs == "curtain":
            return "curtain-bob", 0.80
        if texture == "curly":
            return "bixie-cut", 0.62
        if side_extent > 0.28:
            return "italian-bob", 0.64
        return "volumized-pixie", 0.61

    if length == "medium":
        if texture == "curly":
            return "air-perm-layers", 0.77
        if texture == "wavy":
            if bangs == "curtain":
                return "soft-c-curled-lob", 0.74
            if asymmetry > 0.12:
                return "shaggy-midi", 0.66
            return "layered-midi-waves", 0.71
        if bangs == "curtain":
            return "curtain-bob", 0.68
        if side_extent > 0.42:
            return "soft-wolf-cut", 0.72
        return "sleek-lob", 0.66

    if length == "long":
        if texture == "curly":
            return "air-perm-layers", 0.70
        if texture == "wavy":
            if bangs == "curtain":
                return "hush-cut", 0.82
            if asymmetry > 0.16:
                return "side-swept-long-layers", 0.72
            return "butterfly-layers", 0.72
        if bangs == "curtain":
            return "hush-cut", 0.70
        if asymmetry > 0.16:
            return "side-swept-long-layers", 0.72
        return "butterfly-layers", 0.68

    return None, 0.2


def build_generic_hairstyle_text(length: str, texture: str, bangs: str) -> str:
    parts: List[str] = []
    if length != "updo":
        if length == "short":
            parts.append("short-length")
        elif length == "medium":
            parts.append("medium-length")
        elif length == "long":
            parts.append("long")
    else:
        parts.append("updo")

    if texture and texture != "unknown":
        parts.append(texture)

    parts.append("hair")

    if bangs == "curtain":
        parts.append("with curtain bangs")
    elif bangs == "full":
        parts.append("with full bangs")
    elif bangs == "side":
        parts.append("with side-swept bangs")
    elif bangs == "baby":
        parts.append("with baby bangs")
    elif bangs == "none":
        parts.append("with no bangs")

    return " ".join(parts)


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.manifest)
    if args.limit is not None:
        records = records[: args.limit]
    trends = json.loads(args.trend_data.read_text(encoding="utf-8"))
    trend_by_id = {
        str(row.get("id")).strip(): row
        for row in trends
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    }

    enriched: List[Dict] = []
    style_counts: Dict[str, int] = {}
    length_counts: Dict[str, int] = {}
    texture_counts: Dict[str, int] = {}
    bangs_counts: Dict[str, int] = {}
    matched_style_count = 0

    for record in tqdm(records, desc="style-enrich"):
        image_path = Path(record["source_image_path"])
        hair_mask_path = Path(record["mask_path"])
        face_mask_path = Path(record["face_protect_mask_path"])

        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        hair_mask = (np.array(Image.open(hair_mask_path).convert("L")) > 127).astype(np.uint8)
        face_mask = (np.array(Image.open(face_mask_path).convert("L")) > 127).astype(np.uint8)

        hair_bbox = compute_mask_bbox(hair_mask)
        face_bbox = infer_face_bbox(face_mask)
        if hair_bbox is None or face_bbox is None:
            enriched.append(record)
            continue

        fx1, fy1, fx2, fy2 = face_bbox
        left_extent = max(0.0, (fx1 - hair_bbox[0]) / max(1.0, float(fx2 - fx1)))
        right_extent = max(0.0, (hair_bbox[2] - fx2) / max(1.0, float(fx2 - fx1)))
        asymmetry = abs(left_extent - right_extent)

        length, length_features = classify_length(hair_bbox, face_bbox, float(record.get("hair_ratio", 0.0)))
        texture, texture_features = classify_texture(image_rgb, hair_mask)
        bangs_text = str(
            (record.get("raw_labels") or {})
            .get("attribute_wise_captions", {})
            .get("Bangs")
            or ""
        )
        bangs = classify_bangs(bangs_text, hair_mask, face_bbox)
        color = classify_color(image_rgb, hair_mask)

        style_id, style_confidence = choose_style(
            length=length,
            texture=texture,
            bangs=bangs,
            asymmetry=float(asymmetry),
            side_extent=float(length_features["side_extent"]),
            down_extent=float(length_features["down_extent"]),
        )

        if style_id and style_id in trend_by_id:
            trend = trend_by_id[style_id]
            hairstyle_text = trend["style_name"]
            style_description = trend["description"]
            style_domain = "kr" if "k-style" in " ".join(trend.get("keywords", [])).lower() or style_id in {
                "hush-cut", "air-perm-layers", "soft-c-curled-lob"
            } else "global"
            matched_style_count += 1
        else:
            trend = None
            style_id = None
            hairstyle_text = build_generic_hairstyle_text(length, texture, bangs)
            style_description = hairstyle_text
            style_domain = None

        prompt_parts = ["portrait photo"]
        prompt_parts.append(hairstyle_text.lower())
        prompt_parts.append(f"{color} hair")
        if texture != "unknown":
            prompt_parts.append(f"{texture} texture")
        if bangs == "none":
            prompt_parts.append("forehead visible")
        elif bangs == "curtain":
            prompt_parts.append("curtain bangs")
        elif bangs == "full":
            prompt_parts.append("full bangs")
        elif bangs == "side":
            prompt_parts.append("side-swept bangs")
        elif bangs == "baby":
            prompt_parts.append("baby bangs")

        enriched_record = dict(record)
        enriched_record.update(
            {
                "hairstyle_id": style_id,
                "hairstyle_text": hairstyle_text,
                "hairstyle_description": style_description,
                "color_text": color,
                "hair_length": length,
                "hair_texture": texture,
                "bangs": bangs,
                "style_domain": style_domain,
                "style_confidence": round(float(style_confidence), 4),
                "caption_target_hair_enriched": ", ".join(prompt_parts),
                "pseudo_style_features": {
                    **length_features,
                    **texture_features,
                    "left_extent": round(float(left_extent), 4),
                    "right_extent": round(float(right_extent), 4),
                    "asymmetry": round(float(asymmetry), 4),
                },
            }
        )
        enriched.append(enriched_record)

        style_counts[str(style_id or "__generic__")] = style_counts.get(str(style_id or "__generic__"), 0) + 1
        length_counts[length] = length_counts.get(length, 0) + 1
        texture_counts[texture] = texture_counts.get(texture, 0) + 1
        bangs_counts[bangs] = bangs_counts.get(bangs, 0) + 1

    write_jsonl(args.output, enriched)
    summary = {
        "input_manifest": str(args.manifest),
        "output_manifest": str(args.output),
        "total_records": len(enriched),
        "matched_trend_style_records": matched_style_count,
        "generic_style_records": len(enriched) - matched_style_count,
        "length_counts": dict(sorted(length_counts.items())),
        "texture_counts": dict(sorted(texture_counts.items())),
        "bangs_counts": dict(sorted(bangs_counts.items())),
        "top_style_counts": dict(sorted(style_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]),
        "notes": [
            "No local hairstyle classifier checkpoint was discovered in the repository.",
            "Style labels were pseudo-labeled from hair-mask geometry, bangs captions, and hair color/texture heuristics.",
            "Use hairstyle_id/style_confidence for curriculum weighting rather than treating all style labels as equally reliable.",
        ],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
