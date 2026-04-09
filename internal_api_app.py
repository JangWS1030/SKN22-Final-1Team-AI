from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from handler_sd import (
    PROJECT_ROOT,
    _ensure_pipeline_module_imported,
    _generate_per_recommendation,
    _get_pipeline,
    _load_image_from_input,
    _run_recommendation,
)
from style_recommender import _load_hairstyles, classify_face_shape, golden_ratio_score


logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2026-04-03"
API_VERSION = "2026-04-03"
SERVICE_ROLE = "model-ai-analysis-service"
SERVICE_ENV = os.environ.get("MIRRAI_SERVICE_ENV", "local").strip() or "local"
DEV_SERVICE_BASE_URL = "http://localhost:8000"
PROD_SERVICE_BASE_URL = "https://mirrai.shop"
ASSET_TTL_SECONDS = max(60, int(os.environ.get("MIRRAI_ASSET_TTL_SECONDS", "3600")))
ASSET_DIR = PROJECT_ROOT / "output" / "internal_api_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)
SERVICE_STARTED_AT = time.time()

_ANALYZER_PIPELINE = None


class ApiError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        error_code: str,
        message: str,
        detail: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.detail = detail or {}
        self.retryable = retryable


class AnalyzeFaceRequest(BaseModel):
    request_id: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    include_visualization: bool = False


class GenerateSimulationsRequest(BaseModel):
    request_id: Optional[str] = None
    client_id: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    analysis_data: Optional[Dict[str, Any]] = None
    survey_data: Optional[Dict[str, Any]] = None
    scoring_weights: Optional[Dict[str, float]] = None
    color_text: str = ""
    subject_gender: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=5)


class ExplainStyleRequest(BaseModel):
    request_id: Optional[str] = None
    style_id: Optional[str] = None
    style_name: Optional[str] = None
    analysis_data: Optional[Dict[str, Any]] = None
    survey_data: Optional[Dict[str, Any]] = None
    simulation_image_url: Optional[str] = None
    reasoning_snapshot: Optional[Dict[str, Any]] = None


app = FastAPI(
    title="MirrAI Internal AI Service",
    version=API_VERSION,
    description=(
        "Internal AI service contract for MirrAI backend integration. "
        f"Development base URL: {DEV_SERVICE_BASE_URL}. "
        f"Production base URL: {PROD_SERVICE_BASE_URL}. "
        "All endpoints are mounted under /internal/ without path versioning."
    ),
    docs_url="/internal/docs",
    openapi_url="/internal/openapi.json",
    servers=[
        {"url": DEV_SERVICE_BASE_URL, "description": "Development"},
        {"url": PROD_SERVICE_BASE_URL, "description": "Production"},
    ],
)


def _get_asset_secret() -> str:
    configured = os.environ.get("MIRRAI_ASSET_SIGNING_SECRET")
    if configured:
        return configured
    basis = f"{PROJECT_ROOT}:{os.environ.get('MIRRAI_BUILD_TAG', 'dev')}:{SERVICE_ENV}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _error_payload(
    *,
    request_id: str,
    error_code: str,
    message: str,
    detail: Optional[Dict[str, Any]] = None,
    retryable: bool = False,
    processing_time_ms: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "error",
        "schema_version": SCHEMA_VERSION,
        "response_version": API_VERSION,
        "request_id": request_id,
        "error": {
            "error_code": error_code,
            "message": message,
            "detail": detail or {},
            "retryable": retryable,
        },
    }
    if processing_time_ms is not None:
        payload["processing_time_ms"] = processing_time_ms
    return payload


def _success_payload(
    *,
    request_id: str,
    data: Dict[str, Any],
    processing_time_ms: int,
    status: str = "ok",
) -> Dict[str, Any]:
    return {
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "response_version": API_VERSION,
        "request_id": request_id,
        "processing_time_ms": processing_time_ms,
        "data": data,
    }


def _get_request_id(request: Request, body_request_id: Optional[str] = None) -> str:
    return (
        _normalize_optional_text(body_request_id)
        or _normalize_optional_text(request.headers.get("X-Request-ID"))
        or getattr(request.state, "request_id", None)
        or uuid.uuid4().hex
    )


def _elapsed_ms(request: Request) -> int:
    started = getattr(request.state, "started_at", None)
    if started is None:
        return 0
    return int((time.perf_counter() - float(started)) * 1000)


def _verify_internal_auth(
    authorization: Optional[str] = Header(default=None),
    x_internal_api_key: Optional[str] = Header(default=None, alias="X-Internal-API-Key"),
) -> None:
    expected = _normalize_optional_text(
        os.environ.get("MIRRAI_INTERNAL_API_TOKEN") or os.environ.get("MIRRAI_INTERNAL_API_KEY")
    )
    if not expected:
        return

    bearer = None
    if authorization:
        parts = authorization.strip().split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            bearer = parts[1].strip()
    token = bearer or _normalize_optional_text(x_internal_api_key)
    if token != expected:
        raise ApiError(
            status_code=401,
            error_code="AUTH_REQUIRED",
            message="A valid bearer token is required.",
            retryable=False,
        )


def _verify_api_version(
    x_mirrai_api_version: Optional[str] = Header(default=None, alias="X-MirrAI-API-Version"),
) -> None:
    supplied = _normalize_optional_text(x_mirrai_api_version)
    if supplied and supplied != API_VERSION:
        raise ApiError(
            status_code=409,
            error_code="VERSION_MISMATCH",
            message="Requested API version does not match the deployed internal API version.",
            detail={"requested_version": supplied, "supported_version": API_VERSION},
            retryable=False,
        )


def _build_input_payload(image_url: Optional[str], image_base64: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if _normalize_optional_text(image_url):
        payload["image_url"] = str(image_url).strip()
    if _normalize_optional_text(image_base64):
        payload["image_base64"] = str(image_base64).strip()
    if not payload:
        raise ApiError(
            status_code=422,
            error_code="IMAGE_REQUIRED",
            message="Either image_url or image_base64 is required.",
            retryable=False,
        )
    return payload


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


def _create_analysis_pipeline():
    _ensure_pipeline_module_imported()
    from handler_sd import MirrAISDPipeline, SDInpaintConfig

    cfg_fields = {f.name for f in dataclasses.fields(SDInpaintConfig)}
    cfg_kwargs: Dict[str, Any] = {
        "use_sam2": False,
        "use_clip_ranking": False,
    }
    cfg = SDInpaintConfig(**{k: v for k, v in cfg_kwargs.items() if k in cfg_fields})
    pipeline = MirrAISDPipeline(cfg)
    pipeline._load_mediapipe()
    return pipeline


def _get_analysis_pipeline():
    global _ANALYZER_PIPELINE
    if _ANALYZER_PIPELINE is None:
        _ANALYZER_PIPELINE = _create_analysis_pipeline()
    return _ANALYZER_PIPELINE


def _analyze_face_core(
    *,
    image_url: Optional[str],
    image_base64: Optional[str],
    include_visualization: bool,
    request: Request,
) -> Dict[str, Any]:
    img_bgr = _load_image_from_input(_build_input_payload(image_url, image_base64))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pipeline = _get_analysis_pipeline()

    face_bbox = pipeline._detect_face(img_rgb)
    if face_bbox is None:
        raise ApiError(
            status_code=422,
            error_code="FACE_NOT_DETECTED",
            message="No face was detected in the provided image.",
            retryable=False,
        )

    landmark_obs = pipeline._detect_landmark_data(img_rgb, face_bbox)
    if not bool(landmark_obs.get("detected")):
        raise ApiError(
            status_code=422,
            error_code="LANDMARKS_NOT_DETECTED",
            message="Face mesh landmarks could not be detected from the provided image.",
            retryable=False,
        )

    debug_data = landmark_obs.get("debug_data") or {}
    face_ratios = dict(debug_data.get("ratios") or {})
    face_shape, face_shape_scores = classify_face_shape(face_ratios)
    golden_score = golden_ratio_score(face_ratios)

    visualization_url = None
    visualization_expires_at = None
    if include_visualization:
        debug_images = landmark_obs.get("debug_images") or {}
        contour_bgr = debug_images.get("mediapipe_face_mesh_contours")
        if contour_bgr is not None:
            visualization_url, visualization_expires_at = _persist_image_asset(
                image_bgr=contour_bgr,
                request=request,
                prefix="analyze-face",
            )

    return {
        "face_shape": face_shape,
        "face_shape_scores": {
            key: round(float(value), 4) for key, value in face_shape_scores.items()
        },
        "golden_ratio_score": round(float(golden_score), 4),
        "face_ratios": {key: _round_nullable(value, 6) for key, value in face_ratios.items()},
        "face_bbox": _serialize_face_bbox(face_bbox),
        "image_url": visualization_url,
        "image_url_expires_at": visualization_expires_at,
        "schema_version": SCHEMA_VERSION,
    }


def _resolve_analysis_ratios(
    *,
    request: Request,
    image_url: Optional[str],
    image_base64: Optional[str],
    analysis_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if isinstance(analysis_data, dict):
        face_ratios = analysis_data.get("face_ratios") or analysis_data.get("ratios")
        if isinstance(face_ratios, dict):
            return {
                "face_shape": analysis_data.get("face_shape"),
                "golden_ratio_score": analysis_data.get("golden_ratio_score"),
                "face_ratios": face_ratios,
            }
    return _analyze_face_core(
        image_url=image_url,
        image_base64=image_base64,
        include_visualization=False,
        request=request,
    )


def _persist_image_asset(
    *,
    image_bgr,
    request: Request,
    prefix: str,
    ttl_seconds: Optional[int] = None,
) -> Tuple[str, str]:
    ttl = max(60, int(ttl_seconds or ASSET_TTL_SECONDS))
    asset_id = f"{prefix}-{uuid.uuid4().hex}"
    expires_at = int(time.time()) + ttl
    image_path = ASSET_DIR / f"{asset_id}.jpg"
    meta_path = ASSET_DIR / f"{asset_id}.json"
    cv2.imwrite(str(image_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    meta_path.write_text(
        json.dumps(
            {
                "asset_id": asset_id,
                "content_type": "image/jpeg",
                "file_path": str(image_path),
                "expires_at": expires_at,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    token = hmac.new(
        _get_asset_secret().encode("utf-8"),
        f"{asset_id}:{expires_at}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    base_url = str(request.url_for("get_internal_asset", asset_id=asset_id))
    return f"{base_url}?expires={expires_at}&token={token}", time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires_at)
    )


def _load_asset_metadata(asset_id: str) -> Dict[str, Any]:
    meta_path = ASSET_DIR / f"{asset_id}.json"
    if not meta_path.exists():
        raise ApiError(
            status_code=404,
            error_code="ASSET_NOT_FOUND",
            message="Requested asset does not exist.",
            retryable=False,
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _verify_asset_signature(asset_id: str, expires: str, token: str) -> None:
    try:
        expires_at = int(expires)
    except Exception as exc:
        raise ApiError(
            status_code=422,
            error_code="INVALID_ASSET_SIGNATURE",
            message="Asset expiry is invalid.",
            retryable=False,
        ) from exc
    if expires_at < int(time.time()):
        raise ApiError(
            status_code=403,
            error_code="ASSET_URL_EXPIRED",
            message="Asset URL has expired.",
            retryable=False,
        )
    expected = hmac.new(
        _get_asset_secret().encode("utf-8"),
        f"{asset_id}:{expires_at}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, token):
        raise ApiError(
            status_code=403,
            error_code="INVALID_ASSET_SIGNATURE",
            message="Asset token is invalid.",
            retryable=False,
        )


def _find_style_record(style_id: Optional[str], style_name: Optional[str]) -> Dict[str, Any]:
    normalized_style_id = _normalize_optional_text(style_id)
    normalized_style_name = _normalize_optional_text(style_name)
    for style in _load_hairstyles():
        if normalized_style_id and style.get("id") == normalized_style_id:
            return style
        if normalized_style_name and str(style.get("style_name", "")).strip() == normalized_style_name:
            return style
    raise ApiError(
        status_code=404,
        error_code="STYLE_NOT_FOUND",
        message="Requested style could not be found in the recommendation catalog.",
        detail={"style_id": normalized_style_id, "style_name": normalized_style_name},
        retryable=False,
    )


def _build_style_explanation(
    *,
    style: Dict[str, Any],
    analysis_data: Optional[Dict[str, Any]],
    survey_data: Optional[Dict[str, Any]],
    reasoning_snapshot: Optional[Dict[str, Any]],
    simulation_image_url: Optional[str],
) -> Dict[str, Any]:
    detected_face_shape = _normalize_optional_text(
        (analysis_data or {}).get("face_shape")
    ) or _normalize_optional_text((reasoning_snapshot or {}).get("face_shape_detected"))
    supported_shapes = list(style.get("face_shapes") or [])
    trend_name = str(style.get("trend_name", "")).strip()
    description = str(style.get("description", "")).strip()
    length = str(style.get("length", "medium")).strip()
    maintenance = str(style.get("maintenance", "medium")).strip()
    moods = list(style.get("mood") or [])

    why_parts: List[str] = []
    if detected_face_shape and detected_face_shape in supported_shapes:
        why_parts.append(f"{detected_face_shape} face shape compatibility is explicitly supported by this style.")
    elif detected_face_shape:
        why_parts.append(f"This style is being compared against a detected {detected_face_shape} face shape.")
    if moods:
        why_parts.append(f"It aligns with the target mood tags: {', '.join(moods)}.")
    if description:
        why_parts.append(description)

    styling_points = [
        f"Target length: {length}.",
        f"Expected maintenance level: {maintenance}.",
    ]
    if trend_name:
        styling_points.append(f"Trend anchor: {trend_name}.")

    cautions = [
        "Simulation output should be treated as reference imagery, not a guaranteed salon result.",
        "If the simulation image URL expires, the backend should refetch the explanation or store the image immediately.",
    ]
    if isinstance(survey_data, dict) and survey_data:
        cautions.append("Survey inputs were applied when generating the recommendation score.")

    llm_explanation = " ".join(why_parts or [f"{style['style_name']} is recommended based on the current style metadata."]).strip()

    card = {
        "style_id": style["id"],
        "style_name": style["style_name"],
        "summary": description or style["style_name"],
        "why_it_matches": why_parts,
        "styling_points": styling_points,
        "cautions": cautions,
        "simulation_image_url": simulation_image_url,
    }
    return {
        "style_id": style["id"],
        "style_name": style["style_name"],
        "llm_explanation": llm_explanation,
        "simulation_image_url": simulation_image_url,
        "card": card,
    }


@app.middleware("http")
async def attach_request_metadata(request: Request, call_next):
    request.state.started_at = time.perf_counter()
    request.state.request_id = _normalize_optional_text(request.headers.get("X-Request-ID")) or uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-MirrAI-API-Version"] = API_VERSION
    response.headers["X-MirrAI-Schema-Version"] = SCHEMA_VERSION
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(ApiError)
async def handle_api_error(request: Request, exc: ApiError):
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(
            request_id=getattr(request.state, "request_id", uuid.uuid4().hex),
            error_code=exc.error_code,
            message=exc.message,
            detail=exc.detail,
            retryable=exc.retryable,
            processing_time_ms=_elapsed_ms(request),
        ),
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=_error_payload(
            request_id=getattr(request.state, "request_id", uuid.uuid4().hex),
            error_code="VALIDATION_ERROR",
            message="Request validation failed.",
            detail={"errors": exc.errors()},
            retryable=False,
            processing_time_ms=_elapsed_ms(request),
        ),
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception("[internal_api] unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content=_error_payload(
            request_id=getattr(request.state, "request_id", uuid.uuid4().hex),
            error_code="INTERNAL_ERROR",
            message="Unexpected internal service error.",
            detail={"exception_type": type(exc).__name__},
            retryable=False,
            processing_time_ms=_elapsed_ms(request),
        ),
    )


@app.get("/internal/health")
def get_health(
    request: Request,
    _: None = Depends(_verify_api_version),
    __: None = Depends(_verify_internal_auth),
):
    uptime_seconds = int(time.time() - SERVICE_STARTED_AT)
    data = {
        "status": "online",
        "role": SERVICE_ROLE,
        "environment": SERVICE_ENV,
        "build_version": os.environ.get("MIRRAI_BUILD_TAG", "unknown"),
        "model_version": os.environ.get("MIRRAI_BUILD_TAG", "unknown"),
        "uptime_seconds": uptime_seconds,
        "schema_version": SCHEMA_VERSION,
    }
    return _success_payload(
        request_id=_get_request_id(request),
        data=data,
        processing_time_ms=_elapsed_ms(request),
    )


@app.post("/internal/analyze-face")
def analyze_face(
    body: AnalyzeFaceRequest,
    request: Request,
    _: None = Depends(_verify_api_version),
    __: None = Depends(_verify_internal_auth),
):
    data = _analyze_face_core(
        image_url=body.image_url,
        image_base64=body.image_base64,
        include_visualization=bool(body.include_visualization),
        request=request,
    )
    return _success_payload(
        request_id=_get_request_id(request, body.request_id),
        data=data,
        processing_time_ms=_elapsed_ms(request),
    )


@app.post("/internal/generate-simulations")
def generate_simulations(
    body: GenerateSimulationsRequest,
    request: Request,
    _: None = Depends(_verify_api_version),
    __: None = Depends(_verify_internal_auth),
):
    input_payload = _build_input_payload(body.image_url, body.image_base64)
    img_bgr = _load_image_from_input(input_payload)
    analysis_result = _resolve_analysis_ratios(
        request=request,
        image_url=body.image_url,
        image_base64=body.image_base64,
        analysis_data=body.analysis_data,
    )
    face_ratios = analysis_result.get("face_ratios")
    if not isinstance(face_ratios, dict):
        raise ApiError(
            status_code=422,
            error_code="FACE_ANALYSIS_REQUIRED",
            message="face_ratios are required to generate recommendation-based simulations.",
            retryable=False,
        )

    recommendations_data, _, resolved_color = _run_recommendation(
        face_ratios=face_ratios,
        preference=body.survey_data,
        preference_text=None,
        age=None,
        color_text=body.color_text,
        top_k=body.top_k,
        weights=body.scoring_weights,
    )

    pipeline = _get_pipeline()
    generated_results = _generate_per_recommendation(
        pipeline=pipeline,
        img_bgr=img_bgr,
        recommendations=recommendations_data,
        color_text=resolved_color,
        return_intermediates=False,
        mask_refine_mode=None,
        subject_gender=body.subject_gender,
        lora_path=None,
        lora_scale=1.0,
    )
    result_by_rank = {int(r.rank): r for r in generated_results}

    items: List[Dict[str, Any]] = []
    partial_failures: List[Dict[str, Any]] = []
    for rec in recommendations_data:
        rank = int(rec.get("rank", len(items)))
        generated = result_by_rank.get(rank)
        generation_error = _normalize_optional_text(rec.get("generation_error"))
        image_url = None
        expires_at = None
        if generated is not None:
            image_url, expires_at = _persist_image_asset(
                image_bgr=generated.image,
                request=request,
                prefix=f"simulation-{rec.get('style_id') or rank}",
            )
        elif generation_error:
            partial_failures.append(
                {
                    "style_id": rec.get("style_id"),
                    "style_name": rec.get("style_name"),
                    "error_code": "SIMULATION_GENERATION_FAILED",
                    "message": generation_error,
                }
            )

        items.append(
            {
                "style_id": rec.get("style_id"),
                "style_name": rec.get("style_name"),
                "rank": rank,
                "score": _round_nullable(rec.get("score"), 4),
                "simulation_image_url": image_url,
                "simulation_image_url_expires_at": expires_at,
                "reasoning_snapshot": {
                    "face_shape_detected": analysis_result.get("face_shape"),
                    "golden_ratio_score": analysis_result.get("golden_ratio_score"),
                    "matched_face_shapes": rec.get("face_shapes") or [],
                    "recommendation_score": _round_nullable(rec.get("score"), 4),
                    "trend_name": rec.get("trend_name"),
                    "description": rec.get("description"),
                },
            }
        )

    status = "partial_success" if partial_failures else "ok"
    data = {
        "client_id": body.client_id,
        "items": items,
        "processing_mode": "sync",
        "schema_version": SCHEMA_VERSION,
        "partial_failures": partial_failures,
    }
    return _success_payload(
        request_id=_get_request_id(request, body.request_id),
        data=data,
        processing_time_ms=_elapsed_ms(request),
        status=status,
    )


@app.post("/internal/explain-style")
def explain_style(
    body: ExplainStyleRequest,
    request: Request,
    _: None = Depends(_verify_api_version),
    __: None = Depends(_verify_internal_auth),
):
    style = _find_style_record(body.style_id, body.style_name)
    data = _build_style_explanation(
        style=style,
        analysis_data=body.analysis_data,
        survey_data=body.survey_data,
        reasoning_snapshot=body.reasoning_snapshot,
        simulation_image_url=body.simulation_image_url,
    )
    return _success_payload(
        request_id=_get_request_id(request, body.request_id),
        data=data,
        processing_time_ms=_elapsed_ms(request),
    )


@app.get("/internal/assets/{asset_id}", name="get_internal_asset")
def get_internal_asset(asset_id: str, expires: str, token: str):
    _verify_asset_signature(asset_id, expires, token)
    metadata = _load_asset_metadata(asset_id)
    file_path = Path(metadata["file_path"])
    if not file_path.exists():
        raise ApiError(
            status_code=404,
            error_code="ASSET_NOT_FOUND",
            message="Requested asset file no longer exists.",
            retryable=False,
        )
    return FileResponse(
        path=file_path,
        media_type=metadata.get("content_type", "image/jpeg"),
        filename=file_path.name,
    )
