#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def default_image_repo() -> str:
    env_repo = (os.environ.get("DOCKER_IMAGE_REPO") or "").strip()
    if env_repo:
        return env_repo
    username = (os.environ.get("DOCKER_USERNAME") or "").strip()
    if username:
        return f"{username}/sd"
    return "sikersiker/sd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Docker build helper for MirrAI SD images."
    )
    parser.add_argument(
        "--image-repo",
        default=default_image_repo(),
        help="Docker image repository, e.g. sikersiker/sd",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Application image tag to build, e.g. male-medium-rerank-20260326-170000",
    )
    parser.add_argument(
        "--mode",
        choices=("app", "full"),
        default="app",
        help="Use fast app-only build by default. 'full' uses Dockerfile.sd.",
    )
    parser.add_argument(
        "--base-image",
        default="",
        help="Override base image for app mode. Default: <image-repo>:base-latest",
    )
    parser.add_argument(
        "--base-tag",
        default="",
        help="Optional version tag for base image when building it, e.g. v42",
    )
    parser.add_argument(
        "--ensure-base",
        action="store_true",
        help="Build/push the base image first when it is missing.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push built image tags to Docker Hub.",
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Do not add/push the :latest tag.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned docker commands without executing them.",
    )
    return parser.parse_args()


def run_command(command: list[str], *, dry_run: bool) -> None:
    pretty = " ".join(shlex.quote(part) for part in command)
    print(f"[build_sd_image] {pretty}")
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def manifest_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "manifest", "inspect", image],
        check=False,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def build_base_if_needed(
    *,
    image_repo: str,
    base_image: str,
    base_tag: str,
    ensure_base: bool,
    push: bool,
    dry_run: bool,
) -> None:
    if manifest_exists(base_image):
        print(f"[build_sd_image] base image found: {base_image}")
        return

    if not ensure_base:
        raise SystemExit(
            f"Base image is missing: {base_image}\n"
            "Run again with --ensure-base or build/push the base image first."
        )

    resolved_base_tag = base_tag.strip() or "manual"
    base_tags = [f"{image_repo}:base-{resolved_base_tag}", f"{image_repo}:base-latest"]

    build_cmd = ["docker", "build", "-f", "Dockerfile.sd.base"]
    for tag in base_tags:
        build_cmd += ["-t", tag]
    build_cmd.append(".")
    run_command(build_cmd, dry_run=dry_run)

    if push:
        for tag in base_tags:
            run_command(["docker", "push", tag], dry_run=dry_run)
    else:
        print("[build_sd_image] base image built locally only. Use --push to publish it.")


def build_app_image(
    *,
    image_repo: str,
    tag: str,
    base_image: str,
    push: bool,
    include_latest: bool,
    dry_run: bool,
) -> None:
    tags = [f"{image_repo}:{tag}"]
    if include_latest:
        tags.append(f"{image_repo}:latest")

    build_cmd = [
        "docker",
        "build",
        "-f",
        "Dockerfile.sd.app",
        "--build-arg",
        f"BASE_IMAGE={base_image}",
        "--build-arg",
        f"BUILD_TAG={tag}",
    ]
    for image_tag in tags:
        build_cmd += ["-t", image_tag]
    build_cmd.append(".")
    run_command(build_cmd, dry_run=dry_run)

    if push:
        for image_tag in tags:
            run_command(["docker", "push", image_tag], dry_run=dry_run)


def build_full_image(
    *,
    image_repo: str,
    tag: str,
    push: bool,
    include_latest: bool,
    dry_run: bool,
) -> None:
    tags = [f"{image_repo}:{tag}"]
    if include_latest:
        tags.append(f"{image_repo}:latest")

    build_cmd = ["docker", "build", "-f", "Dockerfile.sd"]
    for image_tag in tags:
        build_cmd += ["-t", image_tag]
    build_cmd.append(".")
    run_command(build_cmd, dry_run=dry_run)

    if push:
        for image_tag in tags:
            run_command(["docker", "push", image_tag], dry_run=dry_run)


def main() -> int:
    args = parse_args()

    image_repo = args.image_repo.strip()
    if not image_repo:
        raise SystemExit("--image-repo is required.")

    include_latest = not args.no_latest
    print(f"[build_sd_image] mode={args.mode} image_repo={image_repo} tag={args.tag}")

    if args.mode == "full":
        build_full_image(
            image_repo=image_repo,
            tag=args.tag,
            push=args.push,
            include_latest=include_latest,
            dry_run=args.dry_run,
        )
        return 0

    base_image = args.base_image.strip() or f"{image_repo}:base-latest"
    build_base_if_needed(
        image_repo=image_repo,
        base_image=base_image,
        base_tag=args.base_tag,
        ensure_base=args.ensure_base,
        push=args.push,
        dry_run=args.dry_run,
    )
    build_app_image(
        image_repo=image_repo,
        tag=args.tag,
        base_image=base_image,
        push=args.push,
        include_latest=include_latest,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
