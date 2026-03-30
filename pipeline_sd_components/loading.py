from __future__ import annotations

import dataclasses
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from pipeline_sd_inpainting import (
    CLOTH_CLASS_IDX,
    CONTROLNET_MODEL_ID,
    DEFAULT_RUNTIME_LORA_HF_FILENAME,
    DEFAULT_RUNTIME_LORA_HF_REPO_ID,
    DEFAULT_SEGFACE_BASE_HF_FILENAME,
    DEFAULT_SEGFACE_BASE_HF_REPO_ID,
    DEFAULT_SEGFACE_BASE_HF_SUBFOLDER,
    DEFAULT_SEGFACE_HF_FILENAME,
    DEFAULT_SEGFACE_HF_REPO_ID,
    DEFAULT_SEGFACE_HF_SUBFOLDER,
    DEFAULT_SEGFACE_INPUT_RES,
    DEFAULT_SEGFACE_MODEL_VARIANT,
    EARRING_CLASS_IDX,
    FACE_CLASS_IDXS,
    GLASS_CLASS_IDX,
    HAIR_CLASS_IDX,
    IP_ADAPTER_REPO_ID,
    IP_ADAPTER_WEIGHT,
    NECKLACE_CLASS_IDX,
    PROJECT_ROOT,
    SD_INPAINT_MODEL_ID,
    SD_SIZE,
    _COMMON_STYLE_BLOCK_NEGATIVE,
    _FEMALE_STYLE_HINTS,
    _FEMALE_SUBJECT_HINTS,
    _HAIR_COLOR_TARGET_RGB,
    _MALE_STYLE_HINTS,
    _MALE_SUBJECT_HINTS,
    _MEDIUM_HAIR_KEYWORDS,
    _NEGATIVE_BASE,
    _NO_COLOR_HINTS,
    _SHORT_HAIR_KEYWORDS,
    logger,
)

# Extracted from pipeline_sd_inpainting.py to keep MirrAISDPipeline smaller.

def _load_sam2(self) -> None:
    if not self.config.use_sam2:
        logger.info("[SDPipeline] SAM2 비활성화 (config.use_sam2=False)")
        return

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from utils.sam2_runtime import create_sam2_predictor_factory

    factory = create_sam2_predictor_factory(
        device=str(self.device),
        auto_download=True,
    )
    if factory is None:
        logger.warning(
            "[SDPipeline] SAM2 factory 생성 실패 (checkpoint 없음 or sam2 미설치). "
            "SegFace-only로 진행."
        )
    else:
        self._sam2_factory = factory
        logger.info("[SDPipeline] SAM2 factory 등록 완료")

def _load_mediapipe(self) -> None:
    import mediapipe as mp
    self._mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )
    self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    logger.info("[SDPipeline] MediaPipe FaceDetection/FaceMesh 로드 완료")

def _load_sd_pipeline(self) -> None:
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetInpaintPipeline,
    )
    from diffusers.schedulers import DPMSolverMultistepScheduler

    logger.info(f"[SDPipeline] ControlNet 로드: {CONTROLNET_MODEL_ID}")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_ID, torch_dtype=self.dtype
    )

    logger.info(f"[SDPipeline] SD Inpainting 로드: {SD_INPAINT_MODEL_ID}")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        SD_INPAINT_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=self.dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # DPM-Solver++ 스케줄러 (20~30 steps로 고품질)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    # IP-Adapter face
    logger.info(f"[SDPipeline] IP-Adapter 로드: {IP_ADAPTER_WEIGHT}")
    pipe.load_ip_adapter(
        IP_ADAPTER_REPO_ID,
        subfolder="models",
        weight_name=IP_ADAPTER_WEIGHT,
    )
    pipe.set_ip_adapter_scale(self.config.ip_adapter_scale)

    # PyTorch 2.0+ 기본 SDPA 사용.
    # xformers 강제 활성화는 IP-Adapter attention processor와 충돌한 전력이 있어 비활성 상태로 둔다.

    pipe.to(self.device)
    self._sd_pipe = pipe
    self._apply_runtime_lora()
    logger.info("[SDPipeline] SD Pipeline 로드 완료")

def _discover_default_lora_path(self) -> Optional[str]:
    default_hf_repo_id = (
        _clean_optional_env_text(os.environ.get("MIRRAI_LORA_HF_REPO_ID"))
        or _clean_optional_env_text(os.environ.get("LORA_HF_REPO_ID"))
        or DEFAULT_RUNTIME_LORA_HF_REPO_ID
    )
    if default_hf_repo_id:
        logger.info(f"[SDPipeline] LoRA default source(HF): {default_hf_repo_id}")
        return default_hf_repo_id

    candidate_dirs: List[Path] = [
        PROJECT_ROOT / "pretrained_models" / "generation_lora_stage4_garment_reveal_best",
        PROJECT_ROOT / "pretrained_models" / "generation_lora_stage4_garment_reveal_final",
        PROJECT_ROOT / "pretrained_models" / "generation_lora_stage3_longtail_best",
        PROJECT_ROOT / "pretrained_models" / "generation_lora_stage3_longtail_final",
        PROJECT_ROOT / "pretrained_models" / "generation_lora_stage2_best",
        PROJECT_ROOT / "pretrained_models" / "generation_lora_stage2_final",
        PROJECT_ROOT / "pretrained_models" / "generation_lora_stage1_best",
    ]
    candidate_dirs.extend(
        sorted(
            PROJECT_ROOT.glob("sd_lora_final_bundle_*/model"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        )
    )

    for candidate in candidate_dirs:
        if not candidate.is_dir():
            continue
        if (candidate / "pytorch_lora_weights.safetensors").is_file():
            return str(candidate.resolve())
        if (candidate / "pytorch_lora_weights.bin").is_file():
            return str(candidate.resolve())
    return None

def _resolve_lora_source(
    self,
    lora_path: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    requested = str(lora_path).strip() if lora_path else ""
    if not requested:
        requested = self._discover_default_lora_path() or ""
        if requested:
            logger.info(f"[SDPipeline] LoRA auto-discovered: {requested}")
    if not requested:
        return None, None, None

    raw_path = Path(requested).expanduser()
    if raw_path.is_file():
        resolved_file = raw_path.resolve()
        return str(resolved_file.parent), resolved_file.name, str(resolved_file)

    if raw_path.is_dir():
        weight_dir = raw_path
        if not (weight_dir / "pytorch_lora_weights.safetensors").is_file():
            model_dir = weight_dir / "model"
            if (model_dir / "pytorch_lora_weights.safetensors").is_file():
                weight_dir = model_dir
            elif not (weight_dir / "pytorch_lora_weights.bin").is_file():
                raise FileNotFoundError(
                    f"LoRA directory does not contain pytorch_lora_weights: {raw_path}"
                )
        resolved_dir = weight_dir.resolve()
        return str(resolved_dir), None, str(resolved_dir)

    resolved = str(raw_path)
    if not raw_path.exists():
        default_weight_name = (
            _clean_optional_env_text(os.environ.get("MIRRAI_LORA_HF_FILENAME"))
            or _clean_optional_env_text(os.environ.get("LORA_HF_FILENAME"))
            or DEFAULT_RUNTIME_LORA_HF_FILENAME
        )
        return resolved, default_weight_name, resolved
    return resolved, None, resolved

def _apply_runtime_lora(
    self,
    lora_path: Optional[str] = None,
    lora_scale: Optional[float] = None,
) -> None:
    if self._sd_pipe is None:
        return

    requested_path, requested_weight_name, requested_cache_key = self._resolve_lora_source(
        lora_path or self.config.lora_path
    )
    requested_scale = float(self.config.lora_scale if lora_scale is None else lora_scale)
    if (
        requested_cache_key == self._active_lora_path
        and abs(requested_scale - self._active_lora_scale) < 1e-8
    ):
        return

    if self._active_lora_path is not None and hasattr(self._sd_pipe, "unload_lora_weights"):
        try:
            self._sd_pipe.unload_lora_weights()
        except Exception as exc:
            logger.warning(f"[SDPipeline] unload_lora_weights 실패 (계속 진행): {exc}")
        self._active_lora_path = None
        self._active_lora_adapter_name = None
        self._active_lora_scale = 1.0

    if not requested_path:
        return

    adapter_name = "hairgen_runtime"
    logger.info(
        f"[SDPipeline] LoRA 로드: path={requested_path}, scale={requested_scale:.3f}"
    )
    load_kwargs: Dict[str, Any] = {"adapter_name": adapter_name}
    if requested_weight_name:
        load_kwargs["weight_name"] = requested_weight_name
    self._sd_pipe.load_lora_weights(requested_path, **load_kwargs)
    if hasattr(self._sd_pipe, "set_adapters"):
        try:
            self._sd_pipe.set_adapters(adapter_name, requested_scale)
        except Exception as exc:
            logger.warning(f"[SDPipeline] set_adapters 실패 (기본 scale 사용): {exc}")

    self._active_lora_path = requested_cache_key
    self._active_lora_scale = requested_scale
    self._active_lora_adapter_name = adapter_name

def _load_lama(self) -> None:
    """LaMa (Large Mask Inpainting) 모델 로드"""
    if self._lama is not None:
        return
    logger.info("[SDPipeline] LaMa 모델 로드 중... preferred_device=%s", self.device)

    try:
        from simple_lama_inpainting import SimpleLama
    except Exception as exc:
        self._lama_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "[SDPipeline] simple_lama_inpainting import 실패. cv2 fallback 사용: %s",
            self._lama_error,
        )
        return

    def _try_load(device: torch.device) -> bool:
        try:
            self._lama = SimpleLama(device=device)
            self._lama_device = str(device)
            self._lama_error = None
            logger.info("[SDPipeline] LaMa 로드 완료 (device=%s)", device)
            return True
        except Exception as exc:
            self._lama = None
            self._lama_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "[SDPipeline] LaMa 로드 실패 (device=%s): %s",
                device,
                self._lama_error,
            )
            return False

    if _try_load(self.device):
        return

    if self.device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        logger.warning("[SDPipeline] LaMa CPU fallback 시도")
        if _try_load(torch.device("cpu")):
            return

    logger.warning("[SDPipeline] LaMa 비활성화. cv2 fallback 사용")

def _cv2_inpaint_rgb(img_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Fallback inpaint path when LaMa is unavailable or incompatible."""
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    if not np.any(mask_u8):
        return img_rgb.copy()
    telea = cv2.inpaint(img_rgb, mask_u8, 3, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(img_rgb, mask_u8, 4, cv2.INPAINT_NS)
    return cv2.addWeighted(telea, 0.72, ns, 0.28, 0.0)

def _lama_inpaint(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    LaMa로 대규모 영역 inpainting (progressive 방식).

    대규모 마스크(이미지의 20%+ 영역)인 경우, 바깥 테두리부터 안쪽으로
    단계적으로 인페인팅하여 품질을 높임.

    img_rgb: (H, W, 3) uint8 RGB
    mask: (H, W) float32 [0,1] 또는 uint8 [0,255]
    Returns: (H, W, 3) uint8 RGB
    """
    src_h, src_w = img_rgb.shape[:2]

    def _match_source_size(result_rgb: np.ndarray) -> np.ndarray:
        res_h, res_w = result_rgb.shape[:2]
        if (res_h, res_w) == (src_h, src_w):
            return result_rgb
        if res_h >= src_h and res_w >= src_w:
            cropped = result_rgb[:src_h, :src_w]
            if cropped.shape[:2] == (src_h, src_w):
                logger.info(
                    f"[SDPipeline] LaMa output size corrected by crop: "
                    f"{res_w}x{res_h} -> {src_w}x{src_h}"
                )
                return cropped
        logger.warning(
            f"[SDPipeline] LaMa output size mismatch, resizing: "
            f"{res_w}x{res_h} -> {src_w}x{src_h}"
        )
        return cv2.resize(result_rgb, (src_w, src_h), interpolation=cv2.INTER_LINEAR)

    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    else:
        mask_u8 = mask.copy()

    if not np.any(mask_u8):
        return img_rgb.copy()

    def _run_inpaint_step(source_rgb: np.ndarray, step_mask_u8: np.ndarray, label: str) -> np.ndarray:
        if self._lama is None:
            logger.info("[SDPipeline] LaMa unavailable (%s). cv2 fallback 사용", label)
            return self._cv2_inpaint_rgb(source_rgb, step_mask_u8)

        try:
            result_pil = self._lama(Image.fromarray(source_rgb), Image.fromarray(step_mask_u8))
            return _match_source_size(np.array(result_pil))
        except Exception as exc:
            self._lama_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "[SDPipeline] LaMa inpaint 실패 (%s, device=%s). cv2 fallback 사용: %s",
                label,
                self._lama_device or "unknown",
                self._lama_error,
            )
            self._lama = None
            self._lama_device = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return self._cv2_inpaint_rgb(source_rgb, step_mask_u8)

    total_pixels = int((mask_u8 > 0).sum())
    image_pixels = mask_u8.shape[0] * mask_u8.shape[1]

    # 작은 마스크(이미지의 15% 미만)는 단일 패스
    if total_pixels < image_pixels * 0.15:
        return _run_inpaint_step(img_rgb, mask_u8, "single-pass")

    # ── Progressive inpainting: 바깥→안쪽 단계적 처리 ──────────────
    # 큰 마스크를 3단계로 나눠서 테두리부터 인페인팅
    logger.info(
        f"[SDPipeline] LaMa progressive 모드: "
        f"total_pixels={total_pixels}, ratio={total_pixels/image_pixels:.1%}"
    )

    current_img = img_rgb.copy()
    remaining_mask = mask_u8.copy()
    n_stages = 3
    # 각 단계에서 사용할 erosion 커널 크기 (점점 안쪽으로)
    erode_sizes = [31, 21, 0]  # 마지막은 나머지 전부

    for stage_i, erode_k in enumerate(erode_sizes):
        if remaining_mask.sum() == 0:
            break

        if erode_k > 0:
            # remaining_mask에서 erosion → 안쪽 영역 제거 → 테두리만 남김
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
            inner = cv2.erode(remaining_mask, k, iterations=1)
            stage_mask = remaining_mask - inner  # 테두리 밴드
            stage_mask = np.clip(stage_mask, 0, 255).astype(np.uint8)
        else:
            # 마지막 단계: 남은 영역 전부
            stage_mask = remaining_mask.copy()

        stage_pixels = int((stage_mask > 0).sum())
        if stage_pixels < 100:
            continue

        logger.info(
            f"[SDPipeline] LaMa stage {stage_i+1}/{n_stages}: "
            f"pixels={stage_pixels}"
        )

        current_img = _run_inpaint_step(
            current_img,
            stage_mask,
            f"progressive-stage-{stage_i+1}",
        )

        # 처리 완료된 부분 제거
        remaining_mask = np.clip(
            remaining_mask.astype(np.int16) - stage_mask.astype(np.int16),
            0, 255
        ).astype(np.uint8)

    logger.info("[SDPipeline] LaMa progressive 완료")
    return current_img

def _as_state_dict(candidate: Any) -> Optional[Dict[str, torch.Tensor]]:
    if not isinstance(candidate, dict) or not candidate:
        return None
    state_dict = {
        str(k): v for k, v in candidate.items()
        if torch.is_tensor(v)
    }
    return state_dict or None

def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }

def _coerce_checkpoint_config(raw_config: Any) -> Dict[str, Any]:
    if isinstance(raw_config, dict):
        return dict(raw_config)
    return {}

def _prepare_custom_segface_checkpoint(
    self,
    checkpoint: Any,
    *,
    label: str,
) -> Tuple[Any, Dict[str, Any]]:
    if not isinstance(checkpoint, dict):
        return checkpoint, {}

    raw_state = checkpoint.get("model_state")
    if not isinstance(raw_state, dict):
        return checkpoint, {}

    state_dict = self._strip_prefix(raw_state, "segface.")
    config = self._coerce_checkpoint_config(checkpoint.get("config"))
    alpha_default = float(config.get("lora_alpha", 1.0) or 1.0)

    merged_state: Dict[str, torch.Tensor] = {}
    lora_parts: Dict[str, Dict[str, torch.Tensor]] = {}
    lora_modules_merged = 0
    lora_modules_skipped = 0

    for key, value in state_dict.items():
        if key.endswith(".lora_down.weight"):
            prefix = key[: -len(".lora_down.weight")]
            lora_parts.setdefault(prefix, {})["down"] = value
            continue
        if key.endswith(".lora_up.weight"):
            prefix = key[: -len(".lora_up.weight")]
            lora_parts.setdefault(prefix, {})["up"] = value
            continue
        if ".base." in key:
            merged_state[key.replace(".base.", ".", 1)] = value
            continue
        merged_state[key] = value

    for prefix, parts in lora_parts.items():
        base_weight_key = f"{prefix}.weight"
        base_weight = merged_state.get(base_weight_key)
        down = parts.get("down")
        up = parts.get("up")
        if base_weight is None or down is None or up is None:
            lora_modules_skipped += 1
            continue

        rank = int(down.shape[0]) if down.ndim == 2 else int(config.get("lora_rank", 0) or 0)
        if rank <= 0:
            lora_modules_skipped += 1
            continue

        delta = torch.matmul(
            up.detach().to(dtype=torch.float32),
            down.detach().to(dtype=torch.float32),
        )
        if tuple(delta.shape) != tuple(base_weight.shape):
            logger.warning(
                "[SDPipeline] %s LoRA merge skipped for %s: delta shape=%s base shape=%s",
                label,
                prefix,
                tuple(delta.shape),
                tuple(base_weight.shape),
            )
            lora_modules_skipped += 1
            continue

        scaling = alpha_default / float(rank)
        merged_state[base_weight_key] = base_weight + delta.to(base_weight.dtype) * scaling
        lora_modules_merged += 1

    info = {
        "checkpoint_type": "segface_hair_model",
        "lora_rank": int(config.get("lora_rank", 0) or 0),
        "lora_alpha": float(config.get("lora_alpha", 0.0) or 0.0),
        "lora_modules_seen": int(len(lora_parts)),
        "lora_modules_merged": int(lora_modules_merged),
        "lora_modules_skipped": int(lora_modules_skipped),
        "hair_threshold": float(config.get("threshold", 0.5) or 0.5),
        "image_size": int(config.get("image_size", DEFAULT_SEGFACE_INPUT_RES) or DEFAULT_SEGFACE_INPUT_RES),
        "model_name": str(config.get("model_name", DEFAULT_SEGFACE_MODEL_VARIANT) or DEFAULT_SEGFACE_MODEL_VARIANT),
        "binary_hair_head": True,
    }
    logger.info(
        "[SDPipeline] %s custom checkpoint prepared: lora_modules=%d merged=%d skipped=%d",
        label,
        len(lora_parts),
        lora_modules_merged,
        lora_modules_skipped,
    )
    return {"state_dict": merged_state}, info

def _iter_state_dict_candidates(
    self,
    checkpoint: Any,
) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    seen: set[Tuple[str, ...]] = set()

    def _add(label: str, raw_candidate: Any) -> None:
        state_dict = self._as_state_dict(raw_candidate)
        if state_dict is None:
            return
        fingerprint = tuple(sorted(state_dict.keys()))
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        candidates.append((label, state_dict))

    _add("raw", checkpoint)
    if isinstance(checkpoint, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model_state",
            "model",
            "module",
            "net",
            "ema_state_dict",
            "state_dict_backbone",
        ):
            if key in checkpoint:
                _add(key, checkpoint[key])

        for key, value in checkpoint.items():
            if key in {"state_dict", "model_state_dict", "model_state", "model", "module", "net", "ema_state_dict", "state_dict_backbone"}:
                continue
            if not isinstance(value, dict):
                continue
            key_lower = str(key).lower()
            if "state" in key_lower or "model" in key_lower:
                _add(str(key), value)

    return candidates

def _load_segface_checkpoint(
    self,
    model: torch.nn.Module,
    checkpoint: Any,
    *,
    label: str,
) -> Dict[str, Any]:
    model_keys = set(model.state_dict().keys())

    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_label = ""
    best_matched = -1
    best_non_backbone = -1

    for candidate_label, candidate_state in self._iter_state_dict_candidates(checkpoint):
        variants = [(candidate_label, candidate_state)]
        for prefix in ("module.", "model.", "_orig_mod.", "segface."):
            if any(key.startswith(prefix) for key in candidate_state.keys()):
                variants.append(
                    (f"{candidate_label}|strip:{prefix}", self._strip_prefix(candidate_state, prefix))
                )

        for variant_label, variant_state in variants:
            matched_keys = model_keys & set(variant_state.keys())
            if not matched_keys:
                continue
            non_backbone_matches = sum(
                1 for key in matched_keys if not key.startswith("backbone.")
            )
            if (
                non_backbone_matches > best_non_backbone
                or (
                    non_backbone_matches == best_non_backbone
                    and len(matched_keys) > best_matched
                )
            ):
                best_state_dict = variant_state
                best_label = variant_label
                best_matched = len(matched_keys)
                best_non_backbone = non_backbone_matches

    if best_state_dict is None:
        raise RuntimeError(f"No compatible SegFace weights found for {label}.")

    incompatible = model.load_state_dict(best_state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    info = {
        "candidate": best_label,
        "matched_keys": int(best_matched),
        "matched_non_backbone_keys": int(best_non_backbone),
        "total_model_keys": int(len(model_keys)),
        "missing_keys_count": int(len(missing_keys)),
        "unexpected_keys_count": int(len(unexpected_keys)),
        "missing_keys_preview": missing_keys[:20],
        "unexpected_keys_preview": unexpected_keys[:20],
        "full_decoder_loaded": bool(best_non_backbone > 0),
    }
    logger.info(
        "[SDPipeline] %s load candidate=%s matched=%d/%d non_backbone=%d missing=%d unexpected=%d",
        label,
        best_label,
        best_matched,
        len(model_keys),
        best_non_backbone,
        len(missing_keys),
        len(unexpected_keys),
    )
    if best_non_backbone <= 0:
        logger.warning(
            "[SDPipeline] %s loaded without decoder/head weights; hair parsing quality may be degraded.",
            label,
        )
    return info

def _load_segface(self) -> None:
    """SegFace (Swin-B) 모델 로드"""
    if self._segface is not None and self._segface_base is not None:
        return

    from models.segface.models.segface_celeb import SegFaceCeleb
    from huggingface_hub import hf_hub_download

    logger.info("[SDPipeline] SegFace checkpoint load start")
    model_variant_env = os.environ.get("SEGFACE_MODEL_VARIANT", "").strip()
    input_resolution_env = os.environ.get("SEGFACE_INPUT_RESOLUTION", "").strip()
    local_ckpt_path = os.environ.get("SEGFACE_CKPT_PATH", "").strip()

    if local_ckpt_path:
        ckpt_path = Path(local_ckpt_path).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"SegFace checkpoint not found: {ckpt_path}")
        logger.info("[SDPipeline] SegFace checkpoint source=local (%s)", ckpt_path)
    else:
        repo_id = os.environ.get(
            "SEGFACE_HF_REPO_ID",
            DEFAULT_SEGFACE_HF_REPO_ID,
        ).strip() or DEFAULT_SEGFACE_HF_REPO_ID
        subfolder_raw = os.environ.get("SEGFACE_HF_SUBFOLDER")
        filename_raw = os.environ.get("SEGFACE_HF_FILENAME")
        subfolder = (
            subfolder_raw.strip()
            if isinstance(subfolder_raw, str)
            else DEFAULT_SEGFACE_HF_SUBFOLDER
        )
        filename = (
            filename_raw.strip()
            if isinstance(filename_raw, str) and filename_raw.strip()
            else DEFAULT_SEGFACE_HF_FILENAME
        )
        token = (
            os.environ.get("SEGFACE_HF_TOKEN")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or None
        )
        candidates: List[Tuple[str, str]] = [(subfolder, filename)]
        if (subfolder, filename) != ("", "best.pt"):
            candidates.append(("", "best.pt"))

        last_error: Optional[Exception] = None
        ckpt_path = None
        for candidate_subfolder, candidate_filename in candidates:
            download_kwargs = {
                "repo_id": repo_id,
                "filename": candidate_filename,
                "token": token,
            }
            if candidate_subfolder:
                download_kwargs["subfolder"] = candidate_subfolder

            try:
                ckpt_path = Path(hf_hub_download(**download_kwargs))
                source = (
                    f"{repo_id}/{candidate_subfolder}/{candidate_filename}"
                    if candidate_subfolder
                    else f"{repo_id}/{candidate_filename}"
                )
                logger.info("[SDPipeline] SegFace checkpoint source=hf (%s)", source)
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    "[SDPipeline] SegFace checkpoint candidate failed: repo=%s, subfolder=%s, filename=%s, error=%s",
                    repo_id,
                    candidate_subfolder or "<root>",
                    candidate_filename,
                    e,
                )

        if ckpt_path is None:
            raise RuntimeError(
                f"Failed to download SegFace checkpoint from repo '{repo_id}'."
            ) from last_error

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    custom_checkpoint_config = self._coerce_checkpoint_config(
        ckpt.get("config") if isinstance(ckpt, dict) else None
    )
    model_variant = (
        model_variant_env
        or str(custom_checkpoint_config.get("model_name", "")).strip()
        or DEFAULT_SEGFACE_MODEL_VARIANT
    )
    input_resolution = int(
        input_resolution_env
        or str(custom_checkpoint_config.get("image_size", "")).strip()
        or str(DEFAULT_SEGFACE_INPUT_RES)
    )

    logger.info(
        "[SDPipeline] SegFace load start: variant=%s, input_resolution=%s",
        model_variant,
        input_resolution,
    )
    segface = SegFaceCeleb(
        input_resolution=input_resolution,
        model=model_variant,
    )
    prepared_ckpt, custom_ckpt_info = self._prepare_custom_segface_checkpoint(
        ckpt,
        label="SegFace custom",
    )
    self._segface_load_info = self._load_segface_checkpoint(
        segface,
        prepared_ckpt,
        label="SegFace custom",
    )
    self._segface_load_info["checkpoint_path"] = str(ckpt_path)
    self._segface_load_info.update(custom_ckpt_info)
    self._segface_custom_binary_hair = bool(custom_ckpt_info.get("binary_hair_head"))
    self._segface_custom_hair_threshold = float(
        os.environ.get(
            "SEGFACE_CUSTOM_HAIR_THRESHOLD",
            str(custom_ckpt_info.get("hair_threshold", 0.5)),
        )
    )
    # SegFace는 항상 float32로 실행 (내부에 dtype=torch.float32 하드코딩 있음)
    segface.float().to(self.device).eval()
    self._segface = segface

    if self._segface_base is None:
        base_model_variant = os.environ.get(
            "SEGFACE_BASE_MODEL_VARIANT",
            DEFAULT_SEGFACE_MODEL_VARIANT,
        ).strip() or DEFAULT_SEGFACE_MODEL_VARIANT
        base_input_resolution = int(
            os.environ.get(
                "SEGFACE_BASE_INPUT_RESOLUTION",
                str(DEFAULT_SEGFACE_INPUT_RES),
            )
        )
        base_local_ckpt_path = os.environ.get("SEGFACE_BASE_CKPT_PATH", "").strip()
        base_repo_id = os.environ.get(
            "SEGFACE_BASE_HF_REPO_ID",
            DEFAULT_SEGFACE_BASE_HF_REPO_ID,
        ).strip() or DEFAULT_SEGFACE_BASE_HF_REPO_ID
        base_subfolder_raw = os.environ.get("SEGFACE_BASE_HF_SUBFOLDER")
        base_filename_raw = os.environ.get("SEGFACE_BASE_HF_FILENAME")
        base_subfolder = (
            base_subfolder_raw.strip()
            if isinstance(base_subfolder_raw, str)
            else DEFAULT_SEGFACE_BASE_HF_SUBFOLDER
        )
        base_filename = (
            base_filename_raw.strip()
            if isinstance(base_filename_raw, str) and base_filename_raw.strip()
            else DEFAULT_SEGFACE_BASE_HF_FILENAME
        )
        base_token = (
            os.environ.get("SEGFACE_BASE_HF_TOKEN")
            or os.environ.get("SEGFACE_HF_TOKEN")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or None
        )

        logger.info(
            "[SDPipeline] SegFace base load start: variant=%s, input_resolution=%s",
            base_model_variant,
            base_input_resolution,
        )
        segface_base = SegFaceCeleb(
            input_resolution=base_input_resolution,
            model=base_model_variant,
        )

        if base_local_ckpt_path:
            base_ckpt_path = Path(base_local_ckpt_path).expanduser()
            if not base_ckpt_path.is_file():
                raise FileNotFoundError(
                    f"SegFace base checkpoint not found: {base_ckpt_path}"
                )
            logger.info(
                "[SDPipeline] SegFace base checkpoint source=local (%s)",
                base_ckpt_path,
            )
        else:
            download_kwargs = {
                "repo_id": base_repo_id,
                "filename": base_filename,
                "token": base_token,
            }
            if base_subfolder:
                download_kwargs["subfolder"] = base_subfolder

            base_ckpt_path = Path(hf_hub_download(**download_kwargs))
            base_source = (
                f"{base_repo_id}/{base_subfolder}/{base_filename}"
                if base_subfolder
                else f"{base_repo_id}/{base_filename}"
            )
            logger.info(
                "[SDPipeline] SegFace base checkpoint source=hf (%s)",
                base_source,
            )

        base_ckpt = torch.load(
            str(base_ckpt_path),
            map_location="cpu",
            weights_only=True,
        )
        self._segface_base_load_info = self._load_segface_checkpoint(
            segface_base,
            base_ckpt,
            label="SegFace base",
        )
        self._segface_base_load_info["checkpoint_path"] = str(base_ckpt_path)

        segface_base.float().to(self.device).eval()
        self._segface_base = segface_base

    logger.info("[SDPipeline] SegFace 로드 완료")

def bind_loading_methods_to_pipeline(cls) -> None:
    cls._load_sam2 = _load_sam2
    cls._load_mediapipe = _load_mediapipe
    cls._load_sd_pipeline = _load_sd_pipeline
    cls._discover_default_lora_path = _discover_default_lora_path
    cls._resolve_lora_source = _resolve_lora_source
    cls._apply_runtime_lora = _apply_runtime_lora
    cls._load_lama = _load_lama
    cls._cv2_inpaint_rgb = staticmethod(_cv2_inpaint_rgb)
    cls._lama_inpaint = _lama_inpaint
    cls._as_state_dict = staticmethod(_as_state_dict)
    cls._strip_prefix = staticmethod(_strip_prefix)
    cls._coerce_checkpoint_config = staticmethod(_coerce_checkpoint_config)
    cls._prepare_custom_segface_checkpoint = _prepare_custom_segface_checkpoint
    cls._iter_state_dict_candidates = _iter_state_dict_candidates
    cls._load_segface_checkpoint = _load_segface_checkpoint
    cls._load_segface = _load_segface
