from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import gdown


FILES = {
    "images_hq": {
        "url": "https://drive.google.com/uc?id=1A2dNWabg6_um-V3lhw1tyead5hCpjaW8",
        "output": "archives/images_hq.zip",
        "extract_to": "images_hq",
        "kind": "file",
    },
    "mask_colorized_hq": {
        "url": "https://drive.google.com/uc?id=1q2DWtGA1h4NcS1Az4OX-5sbLXsGWJZWq",
        "output": "archives/mask_colorized_hq.zip",
        "extract_to": "mask_colorized_hq",
        "kind": "file",
    },
    "identity_hq": {
        "url": "https://drive.google.com/uc?id=1yd0bfYkcQ9_BqIxhjyjZ6UHgNVg7n63V",
        "output": "metadata/identity_hq.txt",
        "kind": "file",
    },
    "binary_label_hq": {
        "url": "https://drive.google.com/uc?id=1QvfDVRW7W3-MOCro1EPdnhuXnQ1STOGG",
        "output": "metadata/binary_label_hq.txt",
        "kind": "file",
    },
    "fine_grained_label_hq": {
        "url": "https://drive.google.com/uc?id=1oscEGdTfvBqohlagtp9dfgduGOfgXZxx",
        "output": "metadata/fine_grained_label_hq.txt",
        "kind": "file",
    },
    "text_hq": {
        "url": "https://drive.google.com/drive/folders/1CzTZm8suzDWdoN6DQmv11tsZotYo1Yfu?usp=sharing",
        "output": "text_hq",
        "kind": "folder",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official CelebA-Dialog HQ assets for generation-model training."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset_build/raw/celeba_dialog_hq"),
        help="Target root directory.",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["images_hq", "mask_colorized_hq", "identity_hq", "text_hq"],
        choices=sorted(FILES.keys()),
        help="Assets to download.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded zip archives after extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download existing files and re-extract archives.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_zip(archive_path: Path, target_dir: Path, *, force: bool) -> None:
    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        print(f"[skip] extracted exists: {target_dir}")
        return
    if target_dir.exists() and force:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive_path} -> {target_dir}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_dir)


def download_file(url: str, output_path: Path, *, force: bool) -> Path:
    ensure_parent(output_path)
    if output_path.exists() and not force:
        print(f"[skip] file exists: {output_path}")
        return output_path
    print(f"[download:file] {url} -> {output_path}")
    downloaded = gdown.download(url=url, output=str(output_path), quiet=False, fuzzy=True)
    if downloaded is None:
        raise RuntimeError(f"Failed to download file from {url}")
    return Path(downloaded)


def download_folder(url: str, output_dir: Path, *, force: bool) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        print(f"[skip] folder exists: {output_dir}")
        return output_dir
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download:folder] {url} -> {output_dir}")
    downloaded = gdown.download_folder(
        url=url,
        output=str(output_dir),
        quiet=False,
        use_cookies=False,
    )
    if not downloaded:
        raise RuntimeError(f"Failed to download folder from {url}")
    return output_dir


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    for asset_name in args.assets:
        spec = FILES[asset_name]
        kind = spec["kind"]
        if kind == "file":
            output_path = root / spec["output"]
            downloaded_path = download_file(spec["url"], output_path, force=args.force)
            extract_to = spec.get("extract_to")
            if extract_to:
                extract_zip(downloaded_path, root / extract_to, force=args.force)
                if not args.keep_archives:
                    print(f"[cleanup] removing archive {downloaded_path}")
                    downloaded_path.unlink(missing_ok=True)
        elif kind == "folder":
            download_folder(spec["url"], root / spec["output"], force=args.force)
        else:
            raise ValueError(f"Unknown asset kind: {kind}")

    print(f"[done] raw dataset root: {root}")


if __name__ == "__main__":
    main()
