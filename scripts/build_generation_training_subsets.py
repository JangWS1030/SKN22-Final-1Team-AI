from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training-ready subset manifests from the style-enriched generation manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_build/processed/celeba_dialog_hq_generation/manifests/recon_hair_style_enriched.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset_build/processed/celeba_dialog_hq_generation/manifests"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("dataset_build/processed/celeba_dialog_hq_generation/reports/style_training_subsets_summary.json"),
    )
    parser.add_argument(
        "--style-confidence-threshold",
        type=float,
        default=0.68,
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


def build_train_prompt(row: Dict) -> str:
    prompt = str(row.get("caption_target_hair_enriched") or "").strip()
    if prompt:
        return prompt

    parts: List[str] = ["portrait photo"]
    hairstyle_text = str(row.get("hairstyle_text") or "").strip()
    color_text = str(row.get("color_text") or "").strip()
    texture = str(row.get("hair_texture") or "").strip()
    bangs = str(row.get("bangs") or "").strip()

    if hairstyle_text:
        parts.append(hairstyle_text.lower())
    if color_text:
        parts.append(f"{color_text} hair")
    if texture and texture != "unknown":
        parts.append(f"{texture} texture")
    if bangs == "none":
        parts.append("forehead visible")
    elif bangs == "curtain":
        parts.append("curtain bangs")
    elif bangs == "full":
        parts.append("full bangs")
    elif bangs == "side":
        parts.append("side-swept bangs")
    elif bangs == "baby":
        parts.append("baby bangs")
    return ", ".join(parts)


def compute_sample_weight(row: Dict) -> float:
    quality_score = float(row.get("quality_score") or 1.0)
    style_confidence = float(row.get("style_confidence") or 0.25)

    weight = 0.85 + 0.25 * style_confidence
    if row.get("hairstyle_id"):
        weight += 0.10
    if row.get("style_domain") == "kr":
        weight += 0.10
    if bool(row.get("is_hair_related_request")):
        weight += 0.05
    if str(row.get("hair_texture") or "") in {"wavy", "curly"}:
        weight += 0.05
    if str(row.get("bangs") or "") != "none":
        weight += 0.03

    return round(quality_score * weight, 4)


def attach_training_fields(row: Dict, threshold: float) -> Dict:
    out = dict(row)
    style_confidence = float(row.get("style_confidence") or 0.0)
    high_conf = bool(row.get("hairstyle_id")) and style_confidence >= threshold
    kr_focus = high_conf and row.get("style_domain") == "kr"

    out.update(
        {
            "train_prompt": build_train_prompt(row),
            "sample_weight": compute_sample_weight(row),
            "style_condition_source": "trend_pseudo" if row.get("hairstyle_id") else "generic_pseudo",
            "curriculum_bucket": (
                "kr_focus"
                if kr_focus
                else "high_conf_style"
                if high_conf
                else "generic_style"
            ),
            "style_high_confidence": high_conf,
        }
    )
    return out


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.manifest)

    train_all: List[Dict] = []
    train_highconf: List[Dict] = []
    train_kr_focus: List[Dict] = []
    train_bangs_request: List[Dict] = []
    eval_rows: List[Dict] = []

    for row in rows:
        enriched = attach_training_fields(row, args.style_confidence_threshold)
        split = str(enriched.get("split") or "train")
        if split == "train":
            train_all.append(enriched)
            if enriched["style_high_confidence"]:
                train_highconf.append(enriched)
            if enriched["curriculum_bucket"] == "kr_focus":
                train_kr_focus.append(enriched)
            if bool(enriched.get("is_hair_related_request")):
                train_bangs_request.append(enriched)
        else:
            eval_rows.append(enriched)

    out_dir = args.out_dir
    write_jsonl(out_dir / "style_train_all.jsonl", train_all)
    write_jsonl(out_dir / "style_train_highconf.jsonl", train_highconf)
    write_jsonl(out_dir / "style_train_kr_focus.jsonl", train_kr_focus)
    write_jsonl(out_dir / "style_train_bangs_request.jsonl", train_bangs_request)
    write_jsonl(out_dir / "style_eval.jsonl", eval_rows)

    summary = {
        "input_manifest": str(args.manifest),
        "style_confidence_threshold": args.style_confidence_threshold,
        "train_all": len(train_all),
        "train_highconf": len(train_highconf),
        "train_kr_focus": len(train_kr_focus),
        "train_bangs_request": len(train_bangs_request),
        "eval_rows": len(eval_rows),
        "notes": [
            "style_train_all is the default training manifest for generation fine-tuning.",
            "style_train_highconf is suitable for later curriculum stages or oversampling.",
            "style_train_kr_focus emphasizes Korean-style pseudo labels only.",
            "style_train_bangs_request isolates the official CelebA-Dialog hair-edit supervision subset.",
        ],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
