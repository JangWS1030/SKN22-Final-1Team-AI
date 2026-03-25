"""
MirrAI SD Inpainting — RunPod Serverless Handler
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

입력 스키마 (JSON):
{
  "input": {
    "image":          "<base64 or URL>",   // 필수
    "hairstyle_text": "wolf cut, layered", // 헤어스타일 설명
    "color_text":     "auburn",            // 헤어 색상 (선택)
    "top_k":          3,                   // 결과 수 (1~5, 기본 3)
    "mask_refine_mode": "sam2",            // "sam2" | "segface_priority" | "segface_only"
    "return_base64":  true,                // true=base64, false=이미지 없이 메타만
    "return_intermediates": false          // true=중간 산출물(base64) 포함
  }
}

출력 스키마:
{
  "results": [
    {
      "rank":         0,          // CLIP score 기준 0=best
      "seed":         42,
      "clip_score":   0.312,
      "mask_used":    "sam2",     // "sam2" | "sam2_soft" | "segface"
      "mask_refine_mode": "sam2",
      "image_base64": "..."       // return_base64=true 일 때
    },
    ...
  ],
  "intermediates": { ... },   // return_intermediates=true 일 때
  "elapsed_seconds": 12.3
}
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import sys
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
RUNPOD_INPUT_DIR = PROJECT_ROOT / "output" / "runpod_inputs"
RUNPOD_INPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_INPUT_PIXELS   = 20_000_000    # 20MP
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
DOWNLOAD_TIMEOUT   = 30

# ── 파이프라인 싱글톤 ───────────────────────────────────────────────────────────
_PIPELINE = None

MASK_DEBUG_KEYWORDS = (
    "mask",
)

_IMPORT_ERROR: Optional[str] = None
try:
    import cv2
    import numpy as np
    from PIL import Image
    from pipeline_sd_inpainting import MirrAISDPipeline, SDInpaintConfig
except Exception as _e:
    _IMPORT_ERROR = f"{type(_e).__name__}: {_e}\n{traceback.format_exc()}"
    logger.error(f"[handler_sd] import 실패:\n{_IMPORT_ERROR}")


def _get_pipeline() -> "MirrAISDPipeline":
    global _PIPELINE
    if _PIPELINE is None:
        logger.info("[handler_sd] 모델 다운로드 확인 중 (cold start)...")
        try:
            from runtime_download import ensure_models_cached
            ensure_models_cached()
        except Exception as e:
            logger.warning(f"[handler_sd] runtime_download 실패 (계속 진행): {e}")

        logger.info("[handler_sd] 파이프라인 초기화...")

        # SDInpaintConfig에 실제 존재하는 필드만 전달
        # (구 버전 이미지와 실행시 호환성 보장)
        import dataclasses
        _cfg_fields = {f.name for f in dataclasses.fields(SDInpaintConfig)}
        _cfg_kwargs = {
            "use_sam2":            os.environ.get("ENABLE_SAM2", "1") in {"1", "true", "yes"},
            "use_clip_ranking":    True,
            "use_color_match":     True,
            "use_poisson_blend":   True,
            "lora_path":           os.environ.get("LORA_PATH") or None,
            "lora_scale":          float(os.environ.get("LORA_SCALE", "1.0")),
        }
        cfg = SDInpaintConfig(**{k: v for k, v in _cfg_kwargs.items() if k in _cfg_fields})

        _PIPELINE = MirrAISDPipeline(cfg)
        _PIPELINE.load()

        # segface hair threshold override (환경변수로 빌드 없이 조정 가능)
        _hair_thresh = os.environ.get("SEGFACE_HAIR_THRESHOLD")
        if _hair_thresh is not None:
            _PIPELINE._segface_custom_hair_threshold = float(_hair_thresh)
            logger.info(f"[handler_sd] segface_custom_hair_threshold overridden to {_hair_thresh}")

        logger.info("[handler_sd] 파이프라인 준비 완료")
    return _PIPELINE


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def _coerce_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _is_mask_debug_image(name: str) -> bool:
    key = str(name).strip().lower()
    return any(token in key for token in MASK_DEBUG_KEYWORDS)


def _mask_image_to_float(mask_image: Any) -> Optional["np.ndarray"]:
    if mask_image is None:
        return None
    if not isinstance(mask_image, np.ndarray):
        return None

    mask_arr = mask_image
    if mask_arr.ndim == 3:
        mask_arr = cv2.cvtColor(mask_arr, cv2.COLOR_BGR2GRAY)
    if mask_arr.ndim != 2:
        return None

    mask_f = mask_arr.astype(np.float32)
    if mask_f.max() > 1.0:
        mask_f /= 255.0
    return np.clip(mask_f, 0.0, 1.0)


def _resolve_display_mask(
    pipeline_mask: Optional["np.ndarray"],
    debug_images: Optional[Dict[str, "np.ndarray"]],
    mask_used: str,
) -> Tuple[Optional["np.ndarray"], str]:
    debug_images = debug_images or {}
    normalized_used = str(mask_used or "").strip().lower()

    candidates: list[Tuple[str, Any]] = []
    if "pipeline_short_removal_mask" in debug_images:
        candidates.append(("pipeline_short_removal_mask", debug_images["pipeline_short_removal_mask"]))
    if normalized_used.startswith("sam2") and "sam2_refined_hair_mask" in debug_images:
        candidates.append(("sam2_refined_hair_mask", debug_images["sam2_refined_hair_mask"]))
    if normalized_used == "segface" and "segface_hair_mask" in debug_images:
        candidates.append(("segface_hair_mask", debug_images["segface_hair_mask"]))
    if pipeline_mask is not None:
        candidates.append(("pipeline_sd_inpaint_mask", pipeline_mask))

    for name, source in candidates:
        mask_f = _mask_image_to_float(source)
        if mask_f is not None and float(mask_f.sum()) > 0.0:
            return mask_f, name
    return None, "none"


def _load_image_from_input(inp: Dict[str, Any]) -> "np.ndarray":
    """image 필드(base64 or URL or image_path)에서 BGR numpy array 반환"""

    # 1) 로컬 파일 경로
    image_path = inp.get("image_path")
    if image_path:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
        return _resize_if_needed(img)

    # 2) URL (image_url 키 → 반드시 HTTP 다운로드)
    image_url = inp.get("image_url")
    if image_url:
        if not isinstance(image_url, str) or not image_url.startswith(("http://", "https://")):
            raise ValueError(f"image_url이 유효한 URL이 아닙니다: {image_url!r}")
        img_bytes = _download_url(image_url)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("URL 이미지 디코딩 실패")
        return _resize_if_needed(img_bgr)

    # 3) base64 (image 또는 image_base64 키)
    raw = inp.get("image_base64") or inp.get("image")
    if raw:
        if isinstance(raw, str) and raw.startswith(("http://", "https://")):
            # image 키에 URL이 들어온 경우도 처리
            img_bytes = _download_url(raw)
        else:
            if isinstance(raw, str) and "," in raw:
                raw = raw.split(",", 1)[1]
            img_bytes = base64.b64decode(raw)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("이미지 디코딩 실패 (지원 형식: JPEG, PNG, WEBP)")
        return _resize_if_needed(img_bgr)

    raise ValueError("'image_url', 'image', 'image_base64', 'image_path' 중 하나 필요")


def _resize_if_needed(img_bgr: "np.ndarray") -> "np.ndarray":
    h, w = img_bgr.shape[:2]
    if h * w > MAX_INPUT_PIXELS:
        scale = (MAX_INPUT_PIXELS / (h * w)) ** 0.5
        img_bgr = cv2.resize(
            img_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return img_bgr


def _download_url(url: str) -> bytes:
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".jpg"
    cached = RUNPOD_INPUT_DIR / f"{digest}{suffix}"
    if cached.exists():
        return cached.read_bytes()

    req = urllib.request.Request(
        url, headers={"User-Agent": "MirrAI-SD/1.0"}
    )
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
        length = resp.headers.get("Content-Length")
        if length and int(length) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"이미지 크기 초과: {length} bytes")
        data = resp.read(MAX_DOWNLOAD_BYTES)
    cached.write_bytes(data)
    return data


def _image_to_base64(img_bgr: "np.ndarray", quality: int = 92) -> str:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── RunPod Handler ──────────────────────────────────────────────────────────────

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    # import 에러 체크
    if _IMPORT_ERROR:
        return {"error": f"파이프라인 import 실패:\n{_IMPORT_ERROR}"}

    inp = (job or {}).get("input") or {}

    # 헬스체크
    if _coerce_bool(inp.get("health_check")):
        import torch
        return {
            "status": "ok",
            "build_tag": os.environ.get("MIRRAI_BUILD_TAG", "unknown"),
            "runpod": {
                "endpoint_id": os.environ.get("RUNPOD_ENDPOINT_ID"),
                "pod_id": os.environ.get("RUNPOD_POD_ID"),
                "gpu_type_id": os.environ.get("RUNPOD_GPU_TYPE_ID"),
            },
            "cuda": {
                "available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }

    runtime_meta = {
        "build_tag": os.environ.get("MIRRAI_BUILD_TAG", "unknown"),
        "runpod": {
            "endpoint_id": os.environ.get("RUNPOD_ENDPOINT_ID"),
            "pod_id": os.environ.get("RUNPOD_POD_ID"),
            "gpu_type_id": os.environ.get("RUNPOD_GPU_TYPE_ID"),
        },
    }

    try:
        # ── 입력 파싱 ────────────────────────────────────────────────────────
        hairstyle_text = str(inp.get("hairstyle_text", "")).strip()
        color_text     = str(inp.get("color_text", "")).strip()
        top_k          = max(1, min(5, int(inp.get("top_k", 3))))
        return_base64  = _coerce_bool(inp.get("return_base64"), default=True)
        return_intermediates = _coerce_bool(inp.get("return_intermediates"), default=False)
        mask_debug_only = _coerce_bool(inp.get("mask_debug_only"), default=False)
        bg_fill_mode   = str(inp.get("bg_fill_mode", "cv2")).strip()  # "cv2" | "sd"
        mask_refine_mode = str(inp.get("mask_refine_mode", "")).strip().lower() or None
        lora_path = str(inp.get("lora_path", "")).strip() or None
        lora_scale = float(inp.get("lora_scale", 1.0))

        if not hairstyle_text and not color_text:
            return {"error": "hairstyle_text 또는 color_text 중 하나 이상 필요합니다."}

        # ── 이미지 로드 ──────────────────────────────────────────────────────
        img_bgr = _load_image_from_input(inp)
        h, w = img_bgr.shape[:2]
        logger.info(
            f"[handler_sd] 입력: {w}×{h}, "
            f"hairstyle='{hairstyle_text}', color='{color_text}', top_k={top_k}, "
            f"mask_refine_mode={mask_refine_mode or 'default'}, "
            "landmarks=mediapipe"
        )

        # ── 파이프라인 실행 ───────────────────────────────────────────────────
        pipeline = _get_pipeline()
        # bg_fill_mode를 런타임에 동적으로 설정 (빌드 없이 테스트 가능)
        pipeline.config.bg_fill_mode = bg_fill_mode
        logger.info(f"[handler_sd] bg_fill_mode={bg_fill_mode}")
        logger.info(f"[handler_sd] mask_debug_only={mask_debug_only}")
        results = pipeline.run(
            image=img_bgr,
            hairstyle_text=hairstyle_text,
            color_text=color_text,
            top_k=top_k,
            return_intermediates=return_intermediates,
            mask_refine_mode=mask_refine_mode,
            lora_path=lora_path,
            lora_scale=lora_scale,
        )

        # ── 결과 직렬화 ───────────────────────────────────────────────────────
        output_results = []
        for r in results:
            item: Dict[str, Any] = {
                "rank":       r.rank,
                "seed":       r.seed,
                "clip_score": round(float(r.clip_score), 4),
                "mask_used":  r.mask_used,
                "mask_refine_mode": r.mask_refine_mode,
            }
            if return_base64:
                item["image_base64"] = _image_to_base64(r.image)
                debug_images_for_overlay = r.debug_images or {}
                display_mask, display_mask_name = _resolve_display_mask(
                    pipeline_mask=r.mask,
                    debug_images=debug_images_for_overlay,
                    mask_used=r.mask_used,
                )
                item["mask_display_name"] = display_mask_name

                if r.mask is not None:
                    pipeline_mask_uint8 = (np.clip(r.mask, 0.0, 1.0) * 255).astype(np.uint8)
                    pipeline_mask_rgb = cv2.cvtColor(pipeline_mask_uint8, cv2.COLOR_GRAY2BGR)
                    item["pipeline_mask_name"] = "pipeline_sd_inpaint_mask"
                    item["pipeline_mask_base64"] = _image_to_base64(pipeline_mask_rgb)

                if display_mask is not None:
                    mask_uint8 = (np.clip(display_mask, 0.0, 1.0) * 255).astype(np.uint8)
                    mask_rgb = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
                    item["mask_base64"] = _image_to_base64(mask_rgb)

                    overlay_base_bgr = img_bgr
                    standardized_bgr = debug_images_for_overlay.get("pipeline_standardized_input_image")
                    if (
                        isinstance(standardized_bgr, np.ndarray)
                        and standardized_bgr.shape[:2] == display_mask.shape[:2]
                    ):
                        overlay_base_bgr = standardized_bgr
                    elif overlay_base_bgr.shape[:2] != display_mask.shape[:2]:
                        overlay_base_bgr = cv2.resize(
                            overlay_base_bgr,
                            (display_mask.shape[1], display_mask.shape[0]),
                            interpolation=cv2.INTER_AREA,
                        )
                    orig_rgb = cv2.cvtColor(overlay_base_bgr, cv2.COLOR_BGR2RGB)
                    overlay = orig_rgb.copy()
                    red_layer = np.zeros_like(overlay)
                    red_layer[:, :, 0] = 255
                    alpha = display_mask[..., np.newaxis]
                    overlay = (overlay * (1 - 0.5 * alpha) + red_layer * 0.5 * alpha).astype(np.uint8)
                    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    item["mask_overlay_base64"] = _image_to_base64(overlay_bgr)

                if r.face_bbox is not None:
                    x1, y1, x2, y2 = r.face_bbox
                    item["face_bbox"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

            output_results.append(item)

        intermediates: Dict[str, str] = {}
        intermediate_data: Dict[str, Any] = {}
        if return_intermediates and results:
            debug_images = results[0].debug_images or {}
            for name, dbg_bgr in debug_images.items():
                if mask_debug_only and not _is_mask_debug_image(name):
                    continue
                try:
                    intermediates[name] = _image_to_base64(dbg_bgr, quality=90)
                except Exception as e:
                    logger.warning(f"[handler_sd] intermediate 직렬화 실패({name}): {e}")
            debug_data = results[0].debug_data or {}
            if isinstance(debug_data, dict) and debug_data:
                intermediate_data = debug_data

        elapsed = time.time() - t0
        logger.info(f"[handler_sd] 완료: {elapsed:.1f}s, {len(results)}개 결과")

        response = {
            "results":         output_results,
            "elapsed_seconds": round(elapsed, 2),
            "build_tag":       runtime_meta["build_tag"],
            "runpod":          runtime_meta["runpod"],
        }
        if intermediates:
            response["intermediates"] = intermediates
        if intermediate_data:
            response["intermediate_data"] = intermediate_data
        return response

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[handler_sd] 오류: {e}\n{tb}")
        return {"error": f"{type(e).__name__}: {e}", "traceback": tb}


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def _normalize_runpod_env() -> None:
    """
    Normalize RunPod webhook placeholders for older runpod runtimes.

    Recent RunPod worker envs can expose webhook URLs with placeholders like
    `$RUNPOD_POD_ID` and `$RUNPOD_GPU_TYPE_ID` while omitting the matching env vars.
    `runpod==1.8.1` does not fully resolve those values on its own, so fresh workers
    can fail to start pinging and get reaped before they ever accept a job.
    """
    if not os.environ.get("RUNPOD_ENDPOINT_ID"):
        return

    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        pod_id = os.environ.get("HOSTNAME") or f"local-{uuid.uuid4().hex}"
        os.environ["RUNPOD_POD_ID"] = pod_id
        logger.info("[handler_sd] synthesized RUNPOD_POD_ID for worker startup compatibility")

    gpu_type_id = os.environ.get("RUNPOD_GPU_TYPE_ID")
    if not gpu_type_id:
        gpu_size = str(os.environ.get("RUNPOD_GPU_SIZE", "")).strip()
        if gpu_size:
            gpu_type_id = gpu_size.split(",", 1)[0].strip()
            os.environ["RUNPOD_GPU_TYPE_ID"] = gpu_type_id

    # Leave `$ID` intact so the RunPod SDK can substitute the actual job id
    # when it posts results back to the serverless API.
    replacements = {
        "$RUNPOD_POD_ID": pod_id,
    }
    if gpu_type_id:
        replacements["$RUNPOD_GPU_TYPE_ID"] = gpu_type_id

    for env_key in (
        "RUNPOD_WEBHOOK_GET_JOB",
        "RUNPOD_WEBHOOK_PING",
        "RUNPOD_WEBHOOK_POST_OUTPUT",
        "RUNPOD_WEBHOOK_POST_STREAM",
    ):
        raw = os.environ.get(env_key)
        if not raw:
            continue
        normalized = raw
        for needle, replacement in replacements.items():
            normalized = normalized.replace(needle, replacement)
        if normalized != raw:
            os.environ[env_key] = normalized
            logger.info("[handler_sd] normalized %s", env_key)

if __name__ == "__main__":
    _normalize_runpod_env()
    import runpod
    runpod.serverless.start({"handler": handler})
