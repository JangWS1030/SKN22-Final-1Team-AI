"""
Generation backend registry for the hair inpainting pipeline.

This module intentionally owns only the model loading and invocation metadata.
Mask construction, face protection, prompt building, and compositing stay in
their existing domains.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Dict, Iterable, Optional

import torch

from .config import (
    CONTROLNET_MODEL_ID,
    IP_ADAPTER_REPO_ID,
    IP_ADAPTER_WEIGHT,
    SDXL_INPAINT_MODEL_ID,
    SD_INPAINT_MODEL_ID,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GenerationBackendSpec:
    key: str
    label: str
    model_id: str
    default_size: int
    default_steps: int
    pipeline_kind: str
    supports_controlnet: bool = False
    supports_ip_adapter: bool = False
    supports_negative_prompt: bool = True
    supports_strength: bool = True
    supports_lora: bool = False
    batchable: bool = True
    generator_device: str = "pipeline"
    default_guidance_scale: Optional[float] = None
    default_strength: float = 1.0
    variant: Optional[str] = None


_BACKEND_ALIASES = {
    "": "sd15_controlnet",
    "default": "sd15_controlnet",
    "legacy": "sd15_controlnet",
    "controlnet": "sd15_controlnet",
    "sd15": "sd15_controlnet",
    "sd": "sd15_controlnet",
    "sd_inpaint": "sd15_controlnet",
    "sd15_controlnet": "sd15_controlnet",
    "sdxl": "sdxl_inpaint",
    "sdxl_inpainting": "sdxl_inpaint",
    "sdxl_inpaint": "sdxl_inpaint",
}

_BASE_SPECS: Dict[str, GenerationBackendSpec] = {
    "sd15_controlnet": GenerationBackendSpec(
        key="sd15_controlnet",
        label="SD 1.5 Inpainting + ControlNet Canny + IP-Adapter",
        model_id=SD_INPAINT_MODEL_ID,
        default_size=512,
        default_steps=30,
        pipeline_kind="sd15_controlnet",
        supports_controlnet=True,
        supports_ip_adapter=True,
        supports_negative_prompt=True,
        supports_strength=True,
        supports_lora=True,
        batchable=True,
        generator_device="pipeline",
        default_strength=1.0,
    ),
    "sdxl_inpaint": GenerationBackendSpec(
        key="sdxl_inpaint",
        label="SDXL Inpainting",
        model_id=SDXL_INPAINT_MODEL_ID,
        default_size=1024,
        default_steps=24,
        pipeline_kind="sdxl_inpaint",
        supports_negative_prompt=True,
        supports_strength=True,
        batchable=True,
        generator_device="pipeline",
        default_strength=0.99,
        variant="fp16",
    ),
}


def normalize_generation_backend(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    key = _BACKEND_ALIASES.get(raw, raw)
    if key not in _BASE_SPECS:
        allowed = ", ".join(sorted(_BASE_SPECS))
        raise ValueError(f"Unsupported generation_backend={value!r}. Allowed: {allowed}")
    return key


def iter_generation_backend_keys() -> Iterable[str]:
    return tuple(_BASE_SPECS)


def get_generation_backend_spec(config: Any) -> GenerationBackendSpec:
    key = normalize_generation_backend(
        getattr(config, "generation_backend", None)
        or os.environ.get("MIRRAI_GENERATION_BACKEND")
    )
    spec = _BASE_SPECS[key]

    model_id = spec.model_id
    if key == "sd15_controlnet":
        model_id = os.environ.get("MIRRAI_SD15_INPAINT_MODEL_ID", model_id).strip() or model_id
    elif key == "sdxl_inpaint":
        model_id = (
            os.environ.get("MIRRAI_SDXL_INPAINT_MODEL_ID", "").strip()
            or str(getattr(config, "sdxl_inpaint_model_id", "") or "").strip()
            or model_id
        )

    return dataclasses.replace(spec, model_id=model_id)


def resolve_generation_canvas_size(config: Any) -> int:
    spec = get_generation_backend_spec(config)
    requested = getattr(config, "generation_size", None)
    if requested in (None, ""):
        requested = os.environ.get("MIRRAI_GENERATION_SIZE")
    if requested not in (None, ""):
        size = int(requested)
    else:
        size = int(spec.default_size)
    if size < 256:
        raise ValueError(f"generation_size must be >=256, got {size}")
    # Diffusion pipelines require dimensions divisible by 8.
    return max(256, int(round(size / 8)) * 8)


def resolve_generation_steps(config: Any) -> int:
    spec = get_generation_backend_spec(config)
    requested = getattr(config, "generation_backend_steps", None)
    if requested in (None, ""):
        requested = os.environ.get("MIRRAI_GENERATION_STEPS")
    if requested not in (None, ""):
        return max(1, int(requested))
    if spec.key == "sd15_controlnet":
        return max(1, int(getattr(config, "num_inference_steps", spec.default_steps)))
    return max(1, int(spec.default_steps))


def resolve_generation_guidance_scale(config: Any, prompt_guidance_scale: float) -> float:
    spec = get_generation_backend_spec(config)
    requested = getattr(config, "generation_backend_guidance_scale", None)
    if requested in (None, ""):
        requested = os.environ.get("MIRRAI_GENERATION_GUIDANCE_SCALE")
    if requested not in (None, ""):
        return float(requested)
    if spec.default_guidance_scale is not None:
        return float(spec.default_guidance_scale)
    return float(prompt_guidance_scale)


def _from_pretrained_with_optional_variant(factory: Any, model_id: str, kwargs: Dict[str, Any], variant: Optional[str]) -> Any:
    if not variant:
        return factory.from_pretrained(model_id, **kwargs)
    try:
        return factory.from_pretrained(model_id, variant=variant, **kwargs)
    except Exception as exc:
        logger.warning(
            "[GenerationBackend] retrying %s without variant=%s after load error: %s",
            model_id,
            variant,
            exc,
        )
        return factory.from_pretrained(model_id, **kwargs)


def load_generation_backend(config: Any, device: torch.device, dtype: torch.dtype) -> tuple[Any, GenerationBackendSpec]:
    spec = get_generation_backend_spec(config)
    logger.info("[GenerationBackend] loading backend=%s model=%s", spec.key, spec.model_id)

    if spec.pipeline_kind == "sd15_controlnet":
        from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
        from diffusers.schedulers import DPMSolverMultistepScheduler

        controlnet_model_id = (
            os.environ.get("MIRRAI_CONTROLNET_MODEL_ID", "").strip()
            or CONTROLNET_MODEL_ID
        )
        logger.info("[GenerationBackend] ControlNet load: %s", controlnet_model_id)
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=dtype,
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            spec.model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
        )
        logger.info("[GenerationBackend] IP-Adapter load: %s", IP_ADAPTER_WEIGHT)
        pipe.load_ip_adapter(
            IP_ADAPTER_REPO_ID,
            subfolder="models",
            weight_name=IP_ADAPTER_WEIGHT,
        )
        pipe.set_ip_adapter_scale(float(getattr(config, "ip_adapter_scale", 0.35)))
        pipe.to(device)
        return pipe, spec

    if spec.pipeline_kind == "sdxl_inpaint":
        from diffusers import StableDiffusionXLInpaintPipeline
        from diffusers.schedulers import DPMSolverMultistepScheduler

        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": True,
        }
        pipe = _from_pretrained_with_optional_variant(
            StableDiffusionXLInpaintPipeline,
            spec.model_id,
            load_kwargs,
            spec.variant,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
        )
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        pipe.to(device)
        return pipe, spec

    raise ValueError(f"Unsupported pipeline_kind={spec.pipeline_kind!r}")
