from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


NEGATIVE_PROMPT = "ugly, deformed, blurry, low quality, bad anatomy, distorted face, distorted hair, artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build benchmark manifests for long-tail hairstyle and garment reveal evaluation."
    )
    parser.add_argument(
        "--rare-manifest",
        type=Path,
        default=Path("dataset_build/processed/face_sketches_refined_generation/manifests/recon_hair_external.jsonl"),
    )
    parser.add_argument(
        "--garment-manifest",
        type=Path,
        default=Path("dataset_build/processed/longtail_training/manifests/style_train_stage4_garment_reveal.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset_build/processed/longtail_training/benchmarks"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("dataset_build/processed/longtail_training/reports/benchmark_summary.json"),
    )
    parser.add_argument("--rare-count", type=int, default=24)
    parser.add_argument("--garment-count", type=int, default=18)
    parser.add_argument("--smoke-count", type=int, default=8)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_benchmark_fields(row: Dict, benchmark_type: str, tier: str) -> Dict:
    out = dict(row)
    out["benchmark_type"] = benchmark_type
    out["benchmark_tier"] = tier
    out["benchmark_prompt"] = str(
        row.get("train_prompt")
        or row.get("caption_target_hair_enriched")
        or row.get("caption_target")
        or "portrait photo, realistic hairstyle, preserved identity"
    )
    out["benchmark_negative_prompt"] = NEGATIVE_PROMPT
    return out


def stratified_pick(rows: List[Dict], key: str, count: int) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    for group_rows in grouped.values():
        group_rows.sort(
            key=lambda row: (
                -float(row.get("style_confidence") or 0.0),
                -float(row.get("quality_score") or 0.0),
                str(row.get("sample_id") or ""),
            )
        )

    ordered_keys = sorted(grouped.keys(), key=lambda item: (-len(grouped[item]), item))
    picked: List[Dict] = []
    idx_map = {key: 0 for key in ordered_keys}
    while len(picked) < count:
        advanced = False
        for group_key in ordered_keys:
            idx = idx_map[group_key]
            if idx >= len(grouped[group_key]):
                continue
            picked.append(grouped[group_key][idx])
            idx_map[group_key] += 1
            advanced = True
            if len(picked) >= count:
                break
        if not advanced:
            break
    return picked


def main() -> None:
    args = parse_args()
    rare_rows = load_jsonl(args.rare_manifest)
    garment_rows = load_jsonl(args.garment_manifest)

    rare_priority = [
        row
        for row in rare_rows
        if str(row.get("style_family") or "") in {"coily_protective", "punk_editorial", "retro_structured", "alt_asian_longtail"}
    ]
    rare_showcase = [
        add_benchmark_fields(row, "rare_style_showcase", "showcase")
        for row in stratified_pick(rare_priority, "hairstyle_text", args.rare_count)
    ]

    garment_priority = sorted(
        garment_rows,
        key=lambda row: (
            -float(row.get("cloth_ratio") or 0.0),
            -float(row.get("quality_score") or 0.0),
            str(row.get("sample_id") or ""),
        ),
    )
    garment_showcase = [
        add_benchmark_fields(row, "garment_reveal_showcase", "showcase")
        for row in garment_priority[: args.garment_count]
    ]

    smoke_take_rare = max(1, int(round(args.smoke_count * 0.5)))
    smoke_take_garment = max(1, args.smoke_count - smoke_take_rare)
    smoke_rows = [
        *[add_benchmark_fields(row, "rare_style_smoke", "smoke") for row in rare_showcase[:smoke_take_rare]],
        *[add_benchmark_fields(row, "garment_reveal_smoke", "smoke") for row in garment_showcase[:smoke_take_garment]],
    ][: args.smoke_count]

    out_dir = args.out_dir.resolve()
    write_jsonl(out_dir / "rare_style_showcase_eval.jsonl", rare_showcase)
    write_jsonl(out_dir / "garment_reveal_showcase_eval.jsonl", garment_showcase)
    write_jsonl(out_dir / "smoke_longtail_eval_quick.jsonl", smoke_rows)

    summary = {
        "rare_manifest": str(args.rare_manifest.as_posix()),
        "garment_manifest": str(args.garment_manifest.as_posix()),
        "rare_showcase_rows": len(rare_showcase),
        "garment_showcase_rows": len(garment_showcase),
        "smoke_rows": len(smoke_rows),
        "rare_style_counts": {
            key: sum(1 for row in rare_showcase if str(row.get("style_family") or "") == key)
            for key in sorted({str(row.get("style_family") or "") for row in rare_showcase})
        },
        "notes": [
            "rare_style_showcase_eval is for documentation-ready visual checks across niche style families.",
            "garment_reveal_showcase_eval is for long-to-short clothing and neckline reveal review.",
            "smoke_longtail_eval_quick is a quick sanity suite for Pod startup verification.",
        ],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
