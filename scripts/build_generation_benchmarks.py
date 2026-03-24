from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed benchmark manifests for hairstyle generation.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_build/processed/celeba_dialog_hq_generation/manifests/recon_hair_style_enriched.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset_build/processed/celeba_dialog_hq_generation/benchmarks"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("dataset_build/processed/celeba_dialog_hq_generation/reports/benchmark_summary.json"),
    )
    parser.add_argument("--recon-count", type=int, default=96)
    parser.add_argument("--edit-count", type=int, default=48)
    parser.add_argument("--smoke-count", type=int, default=12)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def round_robin_pick(grouped: Dict[str, List[Dict]], count: int) -> List[Dict]:
    ordered_keys = sorted(grouped, key=lambda key: (-len(grouped[key]), key))
    indices = {key: 0 for key in ordered_keys}
    picked: List[Dict] = []
    while len(picked) < count:
        advanced = False
        for key in ordered_keys:
            idx = indices[key]
            if idx >= len(grouped[key]):
                continue
            picked.append(grouped[key][idx])
            indices[key] += 1
            advanced = True
            if len(picked) >= count:
                break
        if not advanced:
            break
    return picked


def build_recon_benchmark(rows: List[Dict], count: int) -> List[Dict]:
    eval_rows = [row for row in rows if str(row.get("split")) != "train"]
    eval_rows.sort(
        key=lambda row: (
            -float(row.get("style_confidence") or 0.0),
            -float(row.get("quality_score") or 0.0),
            str(row.get("sample_id") or ""),
        )
    )
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in eval_rows:
        style_key = str(row.get("hairstyle_id") or "__generic__")
        grouped[style_key].append(row)

    selected = round_robin_pick(grouped, count)
    output: List[Dict] = []
    for row in selected:
        payload = dict(row)
        payload.update(
            {
                "benchmark_type": "recon",
                "benchmark_prompt": str(
                    row.get("train_prompt")
                    or row.get("caption_target_hair_enriched")
                    or row.get("caption_target")
                    or "portrait photo, realistic hairstyle"
                ),
                "benchmark_negative_prompt": (
                    "ugly, deformed, blurry, low quality, bad anatomy, distorted face, distorted hair, artifacts"
                ),
            }
        )
        output.append(payload)
    return output


def build_edit_benchmark(rows: List[Dict], count: int) -> List[Dict]:
    candidates = [
        row
        for row in rows
        if str(row.get("split")) != "train" and bool(row.get("is_hair_related_request"))
    ]
    candidates.sort(
        key=lambda row: (
            -float(row.get("quality_score") or 0.0),
            -float(row.get("style_confidence") or 0.0),
            str(row.get("sample_id") or ""),
        )
    )
    selected = candidates[:count]
    output: List[Dict] = []
    for row in selected:
        payload = dict(row)
        payload.update(
            {
                "benchmark_type": "edit_request",
                "benchmark_prompt": str(row.get("request_text") or row.get("train_prompt") or "add realistic hairstyle detail"),
                "benchmark_negative_prompt": (
                    "ugly, deformed, blurry, low quality, bad anatomy, distorted face, distorted hair, artifacts"
                ),
            }
        )
        output.append(payload)
    return output


def build_smoke_benchmark(
    recon_rows: List[Dict],
    edit_rows: List[Dict],
    count: int,
) -> List[Dict]:
    recon_take = max(1, int(round(count * 0.67)))
    edit_take = max(1, count - recon_take)
    smoke_rows = recon_rows[:recon_take] + edit_rows[:edit_take]
    smoke_rows = smoke_rows[:count]
    for row in smoke_rows:
        row["benchmark_tier"] = "smoke"
    return smoke_rows


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.manifest)
    recon_rows = build_recon_benchmark(rows, args.recon_count)
    edit_rows = build_edit_benchmark(rows, args.edit_count)
    smoke_rows = build_smoke_benchmark(recon_rows, edit_rows, args.smoke_count)

    write_jsonl(args.out_dir / "recon_eval_stratified.jsonl", recon_rows)
    write_jsonl(args.out_dir / "edit_eval_requests.jsonl", edit_rows)
    write_jsonl(args.out_dir / "smoke_eval_quick.jsonl", smoke_rows)

    summary = {
        "input_manifest": str(args.manifest),
        "recon_rows": len(recon_rows),
        "edit_rows": len(edit_rows),
        "smoke_rows": len(smoke_rows),
        "recon_style_counts": dict(Counter(str(row.get("hairstyle_id") or "__generic__") for row in recon_rows)),
        "edit_request_attribute_counts": dict(Counter(str(row.get("request_attribute") or "unknown") for row in edit_rows)),
        "notes": [
            "recon_eval_stratified is for objective masked reconstruction evaluation.",
            "edit_eval_requests is for prompt-following and qualitative hairstyle edit review.",
            "smoke_eval_quick is for <=10 minute serverless or cold-start sanity checks.",
        ],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
