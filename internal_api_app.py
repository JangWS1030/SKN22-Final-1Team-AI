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
from typing import Any, Dict, Optional, Tuple

import cv2
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from handler_sd import PROJECT_ROOT, _ensure_pipeline_module_imported, _load_image_from_input
from utils.face_metrics import classify_face_shape, golden_ratio_score


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
