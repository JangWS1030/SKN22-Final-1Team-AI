from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public long-tail hairstyle augmentation sources."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("dataset_build/configs/longtail_hair_sources.json"),
        help="JSON catalog describing downloadable hair datasets.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset_build/raw/public_hair_sources"),
        help="Output root for downloaded assets.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Subset of source_id values to download. Defaults to all snapshot-capable entries.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include metadata-only sources when no explicit --sources are provided.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download by bypassing the local cache snapshot target.",
    )
    return parser.parse_args()


def load_catalog(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_selected_sources(
    catalog: Dict[str, Any],
    explicit_sources: List[str] | None,
    include_optional: bool,
) -> Iterable[Dict[str, Any]]:
    all_sources = catalog.get("sources") or []
    if explicit_sources:
        selected = set(explicit_sources)
        missing = selected - {row["source_id"] for row in all_sources}
        if missing:
            raise SystemExit(f"Unknown source ids: {sorted(missing)}")
        for row in all_sources:
            if row["source_id"] in selected:
                yield row
        return

    for row in all_sources:
        mode = str(row.get("download_mode") or "").strip().lower()
        if mode == "snapshot":
            yield row
        elif include_optional and mode == "metadata_only":
            yield row


def main() -> None:
    args = parse_args()
    catalog = load_catalog(args.catalog)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    report_rows: List[Dict[str, Any]] = []

    for source in iter_selected_sources(catalog, args.sources, args.include_optional):
        source_id = source["source_id"]
        repo_id = source.get("repo_id")
        repo_type = source.get("repo_type", "dataset")
        patterns = source.get("default_patterns") or None
        target_dir = root / str(source.get("local_subdir") or source_id)
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        print(f"[download] {source_id} -> {target_dir}")
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=patterns,
            local_dir=str(target_dir),
            force_download=args.force,
        )

        report_rows.append(
            {
                "source_id": source_id,
                "display_name": source.get("display_name"),
                "repo_id": repo_id,
                "download_mode": source.get("download_mode"),
                "local_dir": str(target_dir.as_posix()),
                "snapshot_path": str(Path(snapshot_path).as_posix()),
                "patterns": patterns,
                "license": source.get("license"),
                "role": source.get("role"),
            }
        )

    report = {
        "catalog": str(args.catalog.as_posix()),
        "root": str(root.as_posix()),
        "downloaded_sources": report_rows,
        "notes": [
            "Manual/manual_query sources are intentionally excluded from auto-download.",
            "Use the catalog file as the single source of truth for licensing and intended role.",
        ],
    }
    report_path = root / "download_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
