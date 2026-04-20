"""
Cold-start cache warmup for the SD serverless runtime.

The SD path uses SegFace + SAM2, so legacy BiSeNet assets such as
`pretrained_models/seg.pth` are intentionally not downloaded here.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def ensure_models_cached(generation_backends: Optional[Iterable[str]] = None) -> None:
    """
    Warm the Hugging Face cache for models required by `handler_sd.py`.

    If a model is already cached locally or on a mounted RunPod volume, this
    returns quickly without downloading it again.
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    from pipeline_sd_components.config import (
        CONTROLNET_MODEL_ID,
        IP_ADAPTER_REPO_ID,
        IP_ADAPTER_WEIGHT,
        SDXL_INPAINT_MODEL_ID,
        SD_INPAINT_MODEL_ID,
    )
    from pipeline_sd_components.generation_backends import normalize_generation_backend

    token = os.environ.get("HF_TOKEN") or None
    requested_backends = list(generation_backends or [])
    preload_env = os.environ.get("MIRRAI_PRELOAD_GENERATION_BACKENDS", "")
    if preload_env.strip():
        requested_backends.extend(part.strip() for part in preload_env.split(","))
    if not requested_backends:
        requested_backends = [os.environ.get("MIRRAI_GENERATION_BACKEND", "sdxl_inpaint")]
    backend_keys = sorted({normalize_generation_backend(key) for key in requested_backends})

    models = []
    for backend_key in backend_keys:
        if backend_key == "sd15_controlnet":
            models.extend(
                [
                    (
                        "SD 1.5 Inpainting",
                        os.environ.get("MIRRAI_SD15_INPAINT_MODEL_ID") or SD_INPAINT_MODEL_ID,
                        ["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*", "*.onnx", "*.pb"],
                    ),
                    (
                        "ControlNet Canny",
                        os.environ.get("MIRRAI_CONTROLNET_MODEL_ID") or CONTROLNET_MODEL_ID,
                        ["*.msgpack", "*.h5", "*.onnx"],
                    ),
                ]
            )
        elif backend_key == "sdxl_inpaint":
            models.append(
                (
                    "SDXL Inpainting",
                    os.environ.get("MIRRAI_SDXL_INPAINT_MODEL_ID") or SDXL_INPAINT_MODEL_ID,
                    ["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*", "*.onnx", "*.pb"],
                )
            )
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

    if "sd15_controlnet" in backend_keys:
        _ensure_file(
            "IP-Adapter weight",
            IP_ADAPTER_REPO_ID,
            IP_ADAPTER_WEIGHT,
            "models",
            token,
        )
    _ensure_file(
        "SAM2 checkpoint",
        "facebook/sam2-hiera-large",
        "sam2_hiera_large.pt",
        None,
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
