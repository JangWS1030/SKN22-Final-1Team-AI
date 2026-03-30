from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


LONGTAIL_PRIORITY_FAMILIES = {
    "coily_protective",
    "punk_editorial",
    "retro_structured",
    "alt_asian_longtail",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build stage3/stage4 manifests for long-tail hairstyle and garment reveal training."
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("dataset_build/processed/celeba_dialog_hq_generation/manifests/style_train_highconf.jsonl"),
    )
    parser.add_argument(
        "--base-enriched-manifest",
        type=Path,
        default=Path("dataset_build/processed/celeba_dialog_hq_generation/manifests/recon_hair_style_enriched.jsonl"),
    )
    parser.add_argument(
        "--external-manifests",
        nargs="+",
        type=Path,
        default=[
            Path("dataset_build/processed/face_sketches_refined_generation/manifests/recon_hair_external.jsonl"),
        ],
    )
    parser.add_argument(
        "--support-manifests",
        nargs="+",
        type=Path,
        default=[
            Path("dataset_build/processed/male_asian_hairstyles_generation/manifests/recon_hair_external.jsonl"),
        ],
        help="Optional support manifests kept out of the default train mix.",
    )
    parser.add_argument(
        "--include-support-manifests",
        action="store_true",
        help="Include optional support manifests such as male-only structure sets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset_build/processed/longtail_training/manifests"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("dataset_build/processed/longtail_training/reports/longtail_training_summary.json"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_stage3_mix(base_rows: List[Dict], external_rows: List[Dict]) -> List[Dict]:
    stage3: List[Dict] = []

    for row in base_rows:
        out = dict(row)
        out["curriculum_bucket"] = "base_stability"
        out["sample_weight"] = round(float(out.get("sample_weight") or 1.0) * 0.55, 4)
        out["stage3_role"] = "stability_mix"
        stage3.append(out)

    for row in external_rows:
        if str(row.get("split") or "train") != "train":
            continue
        family = str(row.get("style_family") or "default")
        out = dict(row)
        out["curriculum_bucket"] = "longtail_priority" if family in LONGTAIL_PRIORITY_FAMILIES else "longtail_support"
        out["sample_weight"] = round(float(out.get("sample_weight") or 1.0) * (1.8 if family in LONGTAIL_PRIORITY_FAMILIES else 1.25), 4)
        out["stage3_role"] = "rare_style" if family in LONGTAIL_PRIORITY_FAMILIES else "support_style"
        stage3.append(out)

    return stage3


def build_longtail_eval(external_rows: List[Dict]) -> List[Dict]:
    eval_rows: List[Dict] = []
    for row in external_rows:
        if str(row.get("split") or "train") == "train":
            continue
        out = dict(row)
        out["eval_bucket"] = "ood_longtail"
        eval_rows.append(out)
    return eval_rows


def looks_like_long_hair(row: Dict) -> bool:
    style_text = str(row.get("hairstyle_text") or row.get("caption_target_hair_enriched") or row.get("caption_target") or "").lower()
    if any(token in style_text for token in ("long", "shoulder", "waist", "lob", "layered", "wavy", "ringlet", "ponytail", "hime")):
        return True
    return bool(float(row.get("hair_ratio") or 0.0) >= 0.10)


def build_garment_reveal(base_enriched_rows: List[Dict], external_rows: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []

    for row in base_enriched_rows:
        if str(row.get("split") or "train") != "train":
            continue
        cloth_ratio = float(row.get("cloth_ratio") or 0.0)
        if cloth_ratio < 0.06 or not looks_like_long_hair(row):
            continue
        out = dict(row)
        out["curriculum_bucket"] = "garment_reveal_base"
        out["sample_weight"] = round(max(1.1, float(out.get("sample_weight") or 1.0) * 1.25), 4)
        out["stage4_role"] = "cloth_reconstruction_candidate"
        rows.append(out)

    for row in external_rows:
        if str(row.get("split") or "train") != "train":
            continue
        if not bool(row.get("garment_reveal_candidate")):
            continue
        out = dict(row)
        out["curriculum_bucket"] = "garment_reveal_support"
        out["sample_weight"] = round(float(out.get("sample_weight") or 1.0) * 1.5, 4)
        out["stage4_role"] = "external_long_hair_support"
        rows.append(out)

    return rows


def build_garment_reveal_eval(base_enriched_rows: List[Dict], external_rows: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []

    for row in base_enriched_rows:
        if str(row.get("split") or "train") == "train":
            continue
        cloth_ratio = float(row.get("cloth_ratio") or 0.0)
        if cloth_ratio < 0.06 or not looks_like_long_hair(row):
            continue
        out = dict(row)
        out["eval_bucket"] = "garment_reveal_eval"
        rows.append(out)

    for row in external_rows:
        if str(row.get("split") or "train") == "train":
            continue
        if not bool(row.get("garment_reveal_candidate")):
            continue
        out = dict(row)
        out["eval_bucket"] = "garment_reveal_eval"
        rows.append(out)

    return rows


def count_by(rows: List[Dict], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    base_rows = load_jsonl(args.base_manifest)
    base_enriched_rows = load_jsonl(args.base_enriched_manifest)
    external_rows: List[Dict] = []
    for manifest in args.external_manifests:
        external_rows.extend(load_jsonl(manifest))
    support_rows: List[Dict] = []
    for manifest in args.support_manifests:
        support_rows.extend(load_jsonl(manifest))
    if args.include_support_manifests:
        external_rows.extend(support_rows)

    stage3_rows = build_stage3_mix(base_rows, external_rows)
    eval_rows = build_longtail_eval(external_rows)
    garment_rows = build_garment_reveal(base_enriched_rows, external_rows)
    garment_eval_rows = build_garment_reveal_eval(base_enriched_rows, external_rows)

    out_dir = args.out_dir.resolve()
    write_jsonl(out_dir / "style_train_stage3_longtail_mix.jsonl", stage3_rows)
    write_jsonl(out_dir / "style_eval_stage3_longtail_ood.jsonl", eval_rows)
    write_jsonl(out_dir / "style_train_stage4_garment_reveal.jsonl", garment_rows)
    write_jsonl(out_dir / "style_eval_stage4_garment_reveal.jsonl", garment_eval_rows)

    summary = {
        "base_manifest": str(args.base_manifest.as_posix()),
        "base_enriched_manifest": str(args.base_enriched_manifest.as_posix()),
        "external_manifests": [str(path.as_posix()) for path in args.external_manifests if path.exists()],
        "support_manifests": [str(path.as_posix()) for path in args.support_manifests if path.exists()],
        "support_manifests_included": args.include_support_manifests,
        "stage3_longtail_mix_rows": len(stage3_rows),
        "stage3_longtail_family_counts": count_by(stage3_rows, "style_family"),
        "stage3_longtail_source_counts": count_by(stage3_rows, "source_name"),
        "stage3_eval_rows": len(eval_rows),
        "stage4_garment_reveal_rows": len(garment_rows),
        "stage4_garment_eval_rows": len(garment_eval_rows),
        "stage4_garment_source_counts": count_by(garment_rows, "source_name"),
        "notes": [
            "stage3_longtail_mix keeps a reduced CelebA stability mix and upweights external rare-style rows.",
            "male_asian_hairstyles is excluded from the default train mix because it is male-only and low-volume.",
            "stage4_garment_reveal focuses on long-hair rows with visible clothing/neckline support.",
            "Use style_eval_stage3_longtail_ood for qualitative validation of rare-style retention.",
        ],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
