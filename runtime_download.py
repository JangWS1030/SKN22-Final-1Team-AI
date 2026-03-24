"""
Cold-start cache warmup for the SD serverless runtime.

The SD path uses SegFace + SAM2, so legacy BiSeNet assets such as
`pretrained_models/seg.pth` are intentionally not downloaded here.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def ensure_models_cached() -> None:
    """
    Warm the Hugging Face cache for models required by `handler_sd.py`.

    If a model is already cached locally or on a mounted RunPod volume, this
    returns quickly without downloading it again.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN") or None

    models = [
        (
            "SD Inpainting",
            "runwayml/stable-diffusion-inpainting",
            ["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        ),
        (
            "ControlNet Canny",
            "lllyasviel/control_v11p_sd15_canny",
            ["*.msgpack", "*.h5"],
        ),
    ]

    for name, repo_id, ignore_patterns in models:
        logger.info("[models] %s cache check...", name)
        try:
            snapshot_download(
                repo_id,
                token=token,
                ignore_patterns=ignore_patterns,
                local_files_only=True,
            )
            logger.info("[models] %s cache hit", name)
        except Exception:
            logger.info("[models] %s downloading...", name)
            snapshot_download(
                repo_id,
                token=token,
                ignore_patterns=ignore_patterns,
            )
            logger.info("[models] %s ready", name)

    _ensure_file(
        "IP-Adapter weight",
        "h94/IP-Adapter",
        "ip-adapter-plus-face_sd15.bin",
        "models",
        token,
    )


def _ensure_file(
    name: str,
    repo_id: str,
    filename: str,
    subfolder: str | None,
    token: str | None,
) -> None:
    from huggingface_hub import hf_hub_download

    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "token": token,
        "local_files_only": True,
    }
    if subfolder:
        kwargs["subfolder"] = subfolder

    try:
        hf_hub_download(**kwargs)
        logger.info("[models] %s cache hit", name)
        return
    except Exception:
        logger.info("[models] %s downloading...", name)

    kwargs.pop("local_files_only", None)
    hf_hub_download(**kwargs)
    logger.info("[models] %s ready", name)
