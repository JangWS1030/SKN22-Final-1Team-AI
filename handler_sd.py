"""
MirrAI SD Inpainting — RunPod Serverless Handler
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

== 모드 1: 직접 지정 (기존) ==
{
    "input": {
    "image":          "<base64 or URL>",   // 필수
    "hairstyle_text": "wolf cut, layered", // 헤어스타일 설명
    "color_text":     "auburn",            // 헤어 색상 (선택)
    "white_tshirt_experiment": false,      // 선택: 흰색 티셔츠 고정 의상 프롬프트
    "sd_prompt_data": {                    // 선택: 백엔드에서 전달하는 SD 프롬프트
      "sd_positive": "short bob cut, compact side silhouette",
      "sd_negative": "long curtain hair, chest-length front hair",
      "sd_guidance": 8.5
    },
    "top_k":          3,                   // 결과 수 (1~5, 기본 3)
    "mask_refine_mode": "sam2",            // "sam2" | "segface_priority" | "segface_only"
    "return_base64":  true,
    "return_intermediates": false
  }
}

출력 스키마:
{
  "results": [ ... ],
  "elapsed_seconds": 12.3
}
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
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
_ANALYZER_PIPELINE = None

MASK_DEBUG_KEYWORDS = (
    "mask",
)

_CANONICAL_TARGET_LENGTHS = frozenset({"short", "medium", "long", "bob"})
_CANONICAL_TARGET_VIBES = frozenset({"natural", "chic", "cute", "elegant"})
_CANONICAL_SCALP_TYPES = frozenset({"straight", "waved", "curly", "damaged"})
_CANONICAL_HAIR_COLOURS = frozenset({"black", "brown", "ash", "bleach"})
_CANONICAL_BUDGET_RANGES = frozenset({"low", "mid", "high"})
_SURVEY_HAIR_COLOUR_TEXT = {
    "black": "black",
    "brown": "brown",
    "ash": "ash",
    "bleach": "bleach blonde",
}

_IMPORT_ERROR: Optional[str] = None
MirrAISDPipeline = None
SDInpaintConfig = None
try:
    import cv2
    import numpy as np
    from PIL import Image
    from utils.face_metrics import classify_face_shape, golden_ratio_score
except Exception as _e:
    _IMPORT_ERROR = f"{type(_e).__name__}: {_e}\n{traceback.format_exc()}"
    logger.error(f"[handler_sd] import 실패:\n{_IMPORT_ERROR}")


def _ensure_pipeline_module_imported() -> None:
    global _IMPORT_ERROR, MirrAISDPipeline, SDInpaintConfig

    if MirrAISDPipeline is not None and SDInpaintConfig is not None:
        return
    if _IMPORT_ERROR:
        raise RuntimeError(_IMPORT_ERROR)

    started = time.time()
    try:
        from pipeline_sd_inpainting import (
            MirrAISDPipeline as _MirrAISDPipeline,
            SDInpaintConfig as _SDInpaintConfig,
        )
    except Exception as _e:
        _IMPORT_ERROR = f"{type(_e).__name__}: {_e}\n{traceback.format_exc()}"
        logger.error(f"[handler_sd] pipeline import 실패:\n{_IMPORT_ERROR}")
        raise RuntimeError(_IMPORT_ERROR) from _e

    MirrAISDPipeline = _MirrAISDPipeline
    SDInpaintConfig = _SDInpaintConfig
    logger.info(
        "[handler_sd] pipeline module import 완료 (%.2fs)",
        time.time() - started,
    )


def _get_pipeline() -> "MirrAISDPipeline":
    global _PIPELINE
    if _PIPELINE is None:
        _ensure_pipeline_module_imported()
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


def _extract_sd_prompt_data(inp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = inp.get("sd_prompt_data")
    if not isinstance(raw, dict):
        return None

    positive = str(raw.get("sd_positive", "")).strip()
    if not positive:
        return None

    data: Dict[str, Any] = {"sd_positive": positive}

    negative = str(raw.get("sd_negative", "")).strip()
    if negative:
        data["sd_negative"] = negative

    guidance = raw.get("sd_guidance")
    if guidance not in (None, ""):
        try:
            data["sd_guidance"] = float(guidance)
        except Exception:
            logger.warning("[handler_sd] invalid sd_guidance ignored: %r", guidance)

    return data


def _normalize_choice(value: Any, allowed: frozenset[str]) -> str:
    lowered = str(value or "").strip().lower()
    return lowered if lowered in allowed else ""


def _coerce_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_gender_branch(value: Any) -> str:
    lowered = _clean_text(value).lower()
    if lowered in {"m", "male", "man", "men", "boy", "masculine", "남자", "남성"}:
        return "male"
    if lowered in {"f", "female", "woman", "women", "girl", "feminine", "여자", "여성"}:
        return "female"
    return ""


def _merge_legacy_style_text(*values: Any) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(text)
    return ", ".join(parts)


def _extract_generation_request_context(inp: Dict[str, Any]) -> Dict[str, Any]:
    survey_data = _coerce_dict(inp.get("survey_data"))
    survey_profile = _coerce_dict(survey_data.get("survey_profile"))

    canonical_preferences = {
        "target_length": _normalize_choice(survey_data.get("target_length"), _CANONICAL_TARGET_LENGTHS),
        "target_vibe": _normalize_choice(survey_data.get("target_vibe"), _CANONICAL_TARGET_VIBES),
        "scalp_type": _normalize_choice(survey_data.get("scalp_type"), _CANONICAL_SCALP_TYPES),
        "hair_colour": _normalize_choice(survey_data.get("hair_colour"), _CANONICAL_HAIR_COLOURS),
        "budget_range": _normalize_choice(survey_data.get("budget_range"), _CANONICAL_BUDGET_RANGES),
    }
    gender_branch = _normalize_choice(
        survey_profile.get("gender_branch"),
        frozenset({"male", "female"}),
    )

    legacy_hairstyle_text = _clean_text(inp.get("hairstyle_text"))
    legacy_preference_text = _clean_text(inp.get("preference_text"))
    legacy_preference = _clean_text(inp.get("preference"))
    legacy_color_text = _clean_text(inp.get("color_text"))
    legacy_style_text = _merge_legacy_style_text(
        legacy_hairstyle_text,
        legacy_preference_text,
        legacy_preference,
    )

    resolved_color_text = legacy_color_text
    if canonical_preferences["hair_colour"]:
        resolved_color_text = _SURVEY_HAIR_COLOUR_TEXT.get(
            canonical_preferences["hair_colour"],
            canonical_preferences["hair_colour"],
        )

    legacy_subject_gender = _clean_text(inp.get("subject_gender", inp.get("gender", "")))
    normalized_legacy_gender = _normalize_gender_branch(legacy_subject_gender)
    resolved_subject_gender = gender_branch or normalized_legacy_gender or legacy_subject_gender or None
    fallback_mode = bool(survey_data) and not bool(survey_profile)
    structured_payload_used = bool(
        survey_data
        or gender_branch
        or any(canonical_preferences.values())
        or survey_profile.get("style_axes")
        or survey_profile.get("derived_preferences")
        or survey_data.get("question_answers")
    )
    prompt_context = {
        "structured_payload_present": structured_payload_used,
        "fallback_mode": fallback_mode,
        "gender_branch": gender_branch,
        "canonical_preferences": canonical_preferences,
        "style_axes": survey_profile.get("style_axes") if isinstance(survey_profile.get("style_axes"), dict) else {},
        "derived_preferences": survey_profile.get("derived_preferences"),
        "question_answers": survey_data.get("question_answers"),
        "legacy_fields": {
            "hairstyle_text": legacy_hairstyle_text,
            "preference_text": legacy_preference_text,
            "preference": legacy_preference,
            "color_text": legacy_color_text,
        },
    }
    return {
        "hairstyle_text": legacy_style_text,
        "color_text": resolved_color_text,
        "subject_gender": resolved_subject_gender,
        "prompt_context": prompt_context,
        "resolved_canonical_preferences": canonical_preferences,
        "resolved_gender_branch": gender_branch or normalized_legacy_gender,
        "structured_payload_used": structured_payload_used,
        "fallback_mode": fallback_mode,
    }


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


def _crop_bgr_with_output_box(
    img_bgr: "np.ndarray",
    crop_box: Tuple[int, int, int, int],
) -> "np.ndarray":
    from pipeline_sd_components.output import crop_with_padding

    return crop_with_padding(img_bgr, crop_box)


class FaceAnalysisError(RuntimeError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


def _round_nullable(value: Any, digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _serialize_face_bbox(face_bbox: Tuple[int, int, int, int]) -> Dict[str, int]:
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _create_analysis_pipeline() -> "MirrAISDPipeline":
    _ensure_pipeline_module_imported()
    import dataclasses

    cfg_fields = {f.name for f in dataclasses.fields(SDInpaintConfig)}
    cfg_kwargs: Dict[str, Any] = {
        "use_sam2": False,
        "use_clip_ranking": False,
    }
    cfg = SDInpaintConfig(**{k: v for k, v in cfg_kwargs.items() if k in cfg_fields})
    pipeline = MirrAISDPipeline(cfg)
    pipeline._load_mediapipe()
    return pipeline


def _get_analysis_pipeline() -> "MirrAISDPipeline":
    global _ANALYZER_PIPELINE
    if _ANALYZER_PIPELINE is None:
        _ANALYZER_PIPELINE = _create_analysis_pipeline()
    return _ANALYZER_PIPELINE


def analyze_face_input(
    inp: Dict[str, Any],
    *,
    include_visualization: bool = False,
) -> Dict[str, Any]:
    img_bgr = _load_image_from_input(inp)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pipeline = _get_analysis_pipeline()

    face_bbox = pipeline._detect_face(img_rgb)
    if face_bbox is None:
        raise FaceAnalysisError(
            error_code="FACE_NOT_DETECTED",
            message="No face was detected in the provided image.",
        )

    landmark_obs = pipeline._detect_landmark_data(img_rgb, face_bbox)
    if not bool(landmark_obs.get("detected")):
        raise FaceAnalysisError(
            error_code="LANDMARKS_NOT_DETECTED",
            message="Face mesh landmarks could not be detected from the provided image.",
        )

    debug_data = landmark_obs.get("debug_data") or {}
    face_ratios = dict(debug_data.get("ratios") or {})
    face_shape, face_shape_scores = classify_face_shape(face_ratios)
    golden_score = golden_ratio_score(face_ratios)

    payload: Dict[str, Any] = {
        "face_shape": face_shape,
        "face_shape_scores": {
            key: round(float(value), 4) for key, value in face_shape_scores.items()
        },
        "golden_ratio_score": round(float(golden_score), 4),
        "face_ratios": {key: _round_nullable(value, 6) for key, value in face_ratios.items()},
        "face_bbox": _serialize_face_bbox(face_bbox),
    }

    if include_visualization:
        debug_images = landmark_obs.get("debug_images") or {}
        contour_bgr = debug_images.get("mediapipe_face_mesh_contours")
        if contour_bgr is not None:
            payload["visualization_image_bgr"] = contour_bgr

    return payload


# ── RunPod Handler ──────────────────────────────────────────────────────────────

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    # import 에러 체크
    if _IMPORT_ERROR:
        return {"error": f"파이프라인 import 실패:\n{_IMPORT_ERROR}"}

    inp = (job or {}).get("input") or {}
    runtime_meta = {
        "build_tag": os.environ.get("MIRRAI_BUILD_TAG", "unknown"),
        "runpod": {
            "endpoint_id": os.environ.get("RUNPOD_ENDPOINT_ID"),
            "pod_id": os.environ.get("RUNPOD_POD_ID"),
            "gpu_type_id": os.environ.get("RUNPOD_GPU_TYPE_ID"),
        },
    }

    # ── action 라우팅 ─────────────────────────────────────────────────────
    action = str(inp.get("action", "")).strip().lower().replace("-", "_")

    # 헬스체크
    if action == "health_check" or _coerce_bool(inp.get("health_check")):
        import torch
        return {
            "status": "ok",
            "build_tag": runtime_meta["build_tag"],
            "runpod": runtime_meta["runpod"],
            "cuda": {
                "available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }
    if action == "analyze_face" or _coerce_bool(inp.get("analyze_face")):
        include_visualization = _coerce_bool(inp.get("include_visualization"), default=False)
        try:
            analysis = analyze_face_input(inp, include_visualization=include_visualization)
            visualization_image = analysis.pop("visualization_image_bgr", None)
            response: Dict[str, Any] = {
                "status": "ok",
                "build_tag": runtime_meta["build_tag"],
                "runpod": runtime_meta["runpod"],
                **analysis,
            }
            if visualization_image is not None:
                response["visualization_base64"] = _image_to_base64(visualization_image)
            return response
        except FaceAnalysisError as e:
            return {
                "status": "error",
                "error_code": e.error_code,
                "error": e.message,
                "build_tag": runtime_meta["build_tag"],
                "runpod": runtime_meta["runpod"],
            }

    try:
        # ── 입력 파싱 ────────────────────────────────────────────────────────
        request_context = _extract_generation_request_context(inp)
        hairstyle_text = request_context["hairstyle_text"]
        color_text     = request_context["color_text"]
        top_k          = max(1, min(5, int(inp.get("top_k", 3))))
        return_base64  = _coerce_bool(inp.get("return_base64"), default=True)
        return_intermediates = _coerce_bool(inp.get("return_intermediates"), default=False)
        mask_debug_only = _coerce_bool(inp.get("mask_debug_only"), default=False)
        bg_fill_mode   = str(inp.get("bg_fill_mode", "cv2")).strip()  # "cv2" | "sd"
        mask_refine_mode = str(inp.get("mask_refine_mode", "")).strip().lower() or None
        subject_gender = request_context["subject_gender"]
        sd_prompt_data = _extract_sd_prompt_data(inp)
        prompt_context = request_context["prompt_context"]
        white_tshirt_experiment = _coerce_bool(inp.get("white_tshirt_experiment"), default=False)
        lora_path = str(inp.get("lora_path", "")).strip() or None
        lora_scale = float(inp.get("lora_scale", 1.0))

        deprecated_recommend_keys = ("face_ratios", "age", "weights")
        deprecated_inputs = [key for key in deprecated_recommend_keys if inp.get(key) not in (None, "", {}, [])]
        if deprecated_inputs:
            return {
                "error": "추천 입력은 더 이상 지원하지 않습니다. survey_data, hairstyle_text, color_text, sd_prompt_data를 사용하세요.",
                "unsupported_inputs": deprecated_inputs,
            }

        if not hairstyle_text and not color_text and not request_context["structured_payload_used"]:
            return {"error": "hairstyle_text 또는 color_text 중 하나 이상 필요합니다."}

        # ── 이미지 로드 ──────────────────────────────────────────────────────
        img_bgr = _load_image_from_input(inp)
        h, w = img_bgr.shape[:2]
        logger.info(
            f"[handler_sd] 입력: {w}×{h}, "
            f"hairstyle='{hairstyle_text}', color='{color_text}', top_k={top_k}, "
            f"mask_refine_mode={mask_refine_mode or 'default'}, "
            f"subject_gender={subject_gender or 'auto'}, "
            f"sd_prompt_data={'yes' if sd_prompt_data else 'no'}, "
            f"white_tshirt_experiment={white_tshirt_experiment}"
        )
        logger.info(
            "[handler_sd] request_resolution: gender_branch=%s canonical=%s structured=%s fallback=%s",
            request_context["resolved_gender_branch"] or "legacy",
            request_context["resolved_canonical_preferences"],
            request_context["structured_payload_used"],
            request_context["fallback_mode"],
        )

        # ── 파이프라인 실행 ───────────────────────────────────────────────────
        pipeline = _get_pipeline()
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
            subject_gender=subject_gender,
            lora_path=lora_path,
            lora_scale=lora_scale,
            sd_prompt_data=sd_prompt_data,
            prompt_context=prompt_context,
            white_tshirt_experiment=white_tshirt_experiment,
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
                    crop_box = getattr(r, "output_crop_box", None)
                    if crop_box is not None:
                        cropped_overlay_base = _crop_bgr_with_output_box(
                            overlay_base_bgr,
                            tuple(int(v) for v in crop_box),
                        )
                        if cropped_overlay_base.size > 0:
                            overlay_base_bgr = cropped_overlay_base
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
                if getattr(r, "output_crop_box", None) is not None:
                    crop_x1, crop_y1, crop_x2, crop_y2 = [int(v) for v in r.output_crop_box]
                    item["output_crop_box"] = {
                        "x1": crop_x1,
                        "y1": crop_y1,
                        "x2": crop_x2,
                        "y2": crop_y2,
                    }

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

        request_resolution = None
        if results:
            request_resolution = results[0].style_meta or None
        if request_resolution is None:
            request_resolution = {
                "resolved_gender_branch": request_context["resolved_gender_branch"] or "legacy",
                "resolved_canonical_preferences": request_context["resolved_canonical_preferences"],
                "blocked_vocabulary": [],
                "fallback_mode": request_context["fallback_mode"],
                "structured_payload_used": request_context["structured_payload_used"],
            }

        elapsed = time.time() - t0
        logger.info(f"[handler_sd] 완료: {elapsed:.1f}s, {len(results)}개 결과")

        response: Dict[str, Any] = {
            "results":         output_results,
            "elapsed_seconds": round(elapsed, 2),
            "build_tag":       runtime_meta["build_tag"],
            "runpod":          runtime_meta["runpod"],
            "request_resolution": request_resolution,
        }
        if intermediates:
            response["intermediates"] = intermediates
        if intermediate_data:
            response["intermediate_data"] = intermediate_data
        
        try:
            resp_json_len = len(json.dumps(response))
            logger.info(f"[handler_sd] 응답 생성 완료: json_len={resp_json_len}")
        except:
            pass
            
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

    preload_flag = str(os.environ.get("MIRRAI_PRELOAD_ON_STARTUP", "")).strip().lower()
    should_preload = preload_flag in {"1", "true", "yes", "on"}
    is_runpod_serverless = bool(os.environ.get("RUNPOD_ENDPOINT_ID"))
    force_serverless_preload = str(os.environ.get("MIRRAI_FORCE_SERVERLESS_PRELOAD", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if should_preload and is_runpod_serverless:
        if force_serverless_preload:
            logger.info("[handler_sd] serverless startup preload forced on")
        else:
            logger.info("[handler_sd] serverless startup preload enabled")
    if not should_preload:
        logger.info("[handler_sd] startup preload skipped; pipeline will load on first request")
        import runpod
        runpod.serverless.start({"handler": handler})
        raise SystemExit(0)

    # ── Cold Start 해소: 요청 받기 전에 모델 미리 로드 ─────────────────────
    logger.info("[handler_sd] 서버 시작 전 모델 프리로드 시작...")
    try:
        _get_pipeline()
        logger.info("[handler_sd] 모델 프리로드 완료 — ready to serve")
    except Exception as e:
        logger.error(f"[handler_sd] 모델 프리로드 실패: {e}\n{traceback.format_exc()}")

    import runpod
    runpod.serverless.start({"handler": handler})
