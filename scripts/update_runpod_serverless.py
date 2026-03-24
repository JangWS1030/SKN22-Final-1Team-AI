#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import runpod_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy compatibility wrapper for updating a RunPod serverless endpoint image."
    )
    parser.add_argument("--api-key", default=None, help="RunPod API key.")
    parser.add_argument("--endpoint-id", default=None, help="RunPod endpoint ID.")
    parser.add_argument("--image", required=True, help="New Docker image.")
    parser.add_argument("--wait", action="store_true", help="Wait for rollout to start.")
    parser.add_argument("--timeout", type=int, default=900, help="Rollout timeout in seconds.")
    parser.add_argument("--poll-interval", type=int, default=10, help="Polling interval in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without changing RunPod.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    forwarded = ["runpod_release.py", "--image", args.image, "--timeout", str(args.timeout), "--poll-interval", str(args.poll_interval)]
    if args.api_key:
        forwarded += ["--api-key", args.api_key]
    if args.endpoint_id:
        forwarded += ["--endpoint-id", args.endpoint_id]
    if args.dry_run:
        forwarded.append("--dry-run")
    if not args.wait:
        forwarded.append("--skip-version-wait")

    # deploy-sd.yml runs dedicated smoke tests later, so keep update step focused on rollout.
    forwarded.append("--skip-health-check")

    original_argv = sys.argv[:]
    try:
        sys.argv = forwarded
        return int(runpod_release.main())
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
