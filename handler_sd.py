"""
MirrAI SD Inpainting — RunPod Serverless Handler
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

== 모드 1: 직접 지정 (기존) ==
{
  "input": {
    "image":          "<base64 or URL>",   // 필수
    "hairstyle_text": "wolf cut, layered", // 헤어스타일 설명
    "color_text":     "auburn",            // 헤어 색상 (선택)
    "top_k":          3,                   // 결과 수 (1~5, 기본 3)
    "return_base64":  true,
    "return_intermediates": false
  }
}

== 모드 2: 추천 기반 생성 (취향벡터 + RAG) ==
{
  "input": {
    "image":          "<base64 or URL>",   // 필수
    "face_ratios": {                       // EP1 얼굴 분석 결과
      "cheekbone_to_height": 0.72,
      "jaw_to_height": 0.60,
      "temple_to_height": 0.70,
      "jaw_to_cheekbone": 0.83
    },
    "preference": {                        // 구조화된 취향 (옵션 A)
      "length": "medium",
      "mood": ["trendy", "natural"],
      "hair_type": "wavy",
      "color_temp": "warm",
      "budget": "medium"
    },
    "preference_text": "자연스러운 웨이브", // 자연어 취향 (옵션 B)
    "age": 28,                             // 나이 (분위기 추론용)
    "color_text": "ash brown",
    "top_k": 5,
    "return_base64": true
  }
}

== 모드 3: 트렌드 데이터 최신화 ==
// 3-A: 파이프라인 실행 (RunPod 내부)
{
  "input": {
    "action": "refresh_trends",
    "steps": ["crawl", "refine", "llm_refine", "vectorize", "rebuild_styles"]
  }
}
// 3-B: Django에서 빌드한 ChromaDB 수신 (권장, ~5MB)
{
  "input": {
    "action": "refresh_trends",
    "chromadb_tar_base64": "<base64 tar.gz>"
  }
}

출력 스키마:
{
  "results": [ ... ],
  "recommendations": [ ... ],    // 추천 모드일 때 Top-K 추천 정보
  "rag_context": "...",          // 추천 모드일 때 RAG 트렌드 컨텍스트
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
from typing import Any, Dict, Optional

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


# ── 트렌드 최신화 ──────────────────────────────────────────────────────────────

def _handle_refresh_trends(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    트렌드 데이터 최신화 엔드포인트.

    == 모드 A: 파이프라인 실행 (서버 내부에서 크롤링~벡터화) ==
    {
      "action": "refresh_trends",
      "steps": ["crawl", "refine", "llm_refine", "vectorize", "rebuild_styles"]
    }

    == 모드 B: Django에서 빌드한 ChromaDB 파일 수신 (권장) ==
    {
      "action": "refresh_trends",
      "chromadb_tar_base64": "<base64 encoded tar.gz>"
    }

    Django(CPU)에서 크롤링+정제+벡터화 → tar.gz → base64로 전송하면
    RunPod에서는 압축 해제 + 컬렉션 핫 리로드만 수행.
    """
    t0 = time.time()
    try:
        chromadb_payload = inp.get("chromadb_tar_base64")

        if chromadb_payload:
            # 모드 B: Django에서 빌드한 ChromaDB 파일 수신
            result = _receive_chromadb_archive(chromadb_payload)
        else:
            # 모드 A: 서버 내부 파이프라인 실행
            from rag_pipeline.pipeline import refresh_trends
            steps = inp.get("steps")
            if isinstance(steps, str):
                steps = [s.strip() for s in steps.split(",")]
            result = refresh_trends(steps=steps)

        result["elapsed_seconds"] = round(time.time() - t0, 2)
        return result
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[handler_sd] refresh_trends 오류: {e}\n{tb}")
        return {"error": f"{type(e).__name__}: {e}", "traceback": tb}


def _receive_chromadb_archive(payload_b64: str) -> Dict[str, Any]:
    """
    base64 인코딩된 tar.gz ChromaDB 아카이브를 수신하여 교체.

    기대하는 아카이브 구조:
        chromadb_trends/
        chromadb_ncs/
        chromadb_styles/    (선택)
    """
    import shutil
    import tarfile
    import tempfile

    stores_dir = PROJECT_ROOT / "data" / "rag" / "stores"
    stores_dir.mkdir(parents=True, exist_ok=True)

    # 디코딩 + 압축 해제
    raw = base64.b64decode(payload_b64)
    size_mb = len(raw) / (1024 * 1024)
    logger.info(f"[handler_sd] ChromaDB 아카이브 수신: {size_mb:.1f} MB")

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "chromadb.tar.gz"
        tar_path.write_bytes(raw)

        with tarfile.open(tar_path, "r:gz") as tar:
            # 보안: 경로 탈출 방지
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"안전하지 않은 경로: {member.name}")
            tar.extractall(path=tmpdir)

        # 추출된 컬렉션 디렉터리 교체
        replaced = []
        for collection_name in ("chromadb_trends", "chromadb_ncs", "chromadb_styles"):
            src = Path(tmpdir) / collection_name
            if not src.is_dir():
                continue
            dst = stores_dir / collection_name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            replaced.append(collection_name)
            logger.info(f"[handler_sd] {collection_name} 교체 완료")

    # 메모리 캐시 무효화 (다음 쿼리 시 재로드)
    _invalidate_collection_caches(replaced)

    return {
        "success": True,
        "mode": "receive_archive",
        "replaced_collections": replaced,
        "archive_size_mb": round(size_mb, 2),
    }


def _invalidate_collection_caches(replaced: list) -> None:
    """교체된 컬렉션의 메모리 캐시를 무효화."""
    if "chromadb_styles" in replaced:
        try:
            import style_recommender
            style_recommender._collection_cache = None
            logger.info("[handler_sd] style_recommender 캐시 무효화")
        except Exception:
            pass

    if "chromadb_trends" in replaced or "chromadb_ncs" in replaced:
        try:
            from rag_pipeline import rag_query
            if hasattr(rag_query, "_get_collection"):
                rag_query._get_collection.cache_clear()
            if hasattr(rag_query, "_get_trend_corpus"):
                rag_query._get_trend_corpus.cache_clear()
            logger.info("[handler_sd] rag_query 캐시 무효화")
        except Exception:
            pass

        try:
            from rag_pipeline import ncs_rag_query
            if hasattr(ncs_rag_query, "_get_collection"):
                ncs_rag_query._get_collection.cache_clear()
            logger.info("[handler_sd] ncs_rag_query 캐시 무효화")
        except Exception:
            pass


# ── 추천 + RAG 컨텍스트 ────────────────────────────────────────────────────────

def _run_recommendation(
    face_ratios: Dict[str, Any],
    preference: Optional[Dict[str, Any]],
    preference_text: str,
    age: Optional[int],
    color_text: str,
    top_k: int,
) -> tuple:
    """
    추천 엔진 실행 → (recommendations_data, rag_context, hairstyle_text, color_text)
    """
    from style_recommender import recommend_top_k, recommend_to_dict

    recommendations = recommend_top_k(
        face_ratios=face_ratios,
        preference=preference,
        preference_text=preference_text or None,
        age=age,
        top_k=top_k,
    )
    recommendations_data = recommend_to_dict(recommendations)

    # 추천된 스타일들로 RAG 트렌드 검색
    rag_context_str = _fetch_rag_context_for_styles(recommendations)

    # 첫 번째 추천 스타일을 기본 hairstyle_text로 설정 (단일 생성 모드 폴백용)
    if recommendations:
        top_style = recommendations[0]
        hairstyle_text = top_style.style_name
        if not color_text and top_style.metadata.get("color_temp"):
            color_text = ""  # 색상은 사용자 지정 우선

    logger.info(
        f"[handler_sd] 추천 완료: {len(recommendations)}개 스타일, "
        f"top='{recommendations[0].style_name if recommendations else 'none'}'"
    )

    return recommendations_data, rag_context_str, hairstyle_text, color_text


def _fetch_rag_context_for_styles(recommendations) -> Optional[str]:
    """추천된 스타일들에 대한 RAG 트렌드 컨텍스트를 검색."""
    try:
        from rag_pipeline.rag_query import retrieve, build_context
    except ImportError:
        logger.warning("[handler_sd] RAG pipeline import 실패, 컨텍스트 없이 진행")
        return None

    all_docs = []
    seen_titles = set()
    for rec in recommendations[:3]:  # 상위 3개 스타일만 검색 (성능)
        try:
            docs = retrieve(rec.style_name, n_results=3, expand=True)
            for doc in docs:
                title = doc.get("title", "")
                if title not in seen_titles:
                    seen_titles.add(title)
                    all_docs.append(doc)
        except Exception as e:
            logger.warning(f"[handler_sd] RAG 검색 실패({rec.style_name}): {e}")

    if not all_docs:
        return None

    # 상위 5개 문서로 제한
    return build_context(all_docs[:5])


def _generate_per_recommendation(
    pipeline,
    img_bgr,
    recommendations: list,
    color_text: str,
    return_intermediates: bool,
    lora_path: Optional[str],
    lora_scale: float,
    rag_context: Optional[str],
) -> list:
    """
    추천된 각 스타일마다 1장씩 생성.
    RAG 컨텍스트가 있으면 프롬프트에 트렌드 정보를 주입.
    """
    all_results = []

    for idx, rec in enumerate(recommendations):
        style_name = rec.get("style_name", "")
        description = rec.get("description", "")

        # RAG 트렌드 컨텍스트에서 해당 스타일 관련 정보 추출
        enriched_prompt = style_name
        if description:
            enriched_prompt = f"{style_name}, {description}"

        # RAG 컨텍스트에서 키워드 추출하여 프롬프트 보강
        if rag_context:
            rag_keywords = _extract_rag_keywords(rag_context, style_name)
            if rag_keywords:
                enriched_prompt = f"{enriched_prompt}, {rag_keywords}"

        logger.info(f"[handler_sd] 추천 #{idx}: '{enriched_prompt}'")

        try:
            results = pipeline.run(
                image=img_bgr,
                hairstyle_text=enriched_prompt,
                color_text=color_text,
                top_k=1,  # 스타일당 1장
                return_intermediates=return_intermediates if idx == 0 else False,
                lora_path=lora_path,
                lora_scale=lora_scale,
            )
            for r in results:
                r.rank = idx
                r.style_meta = {
                    "style_id": rec.get("style_id"),
                    "style_name": style_name,
                    "recommendation_score": rec.get("score"),
                }
                all_results.append(r)
        except Exception as e:
            logger.error(f"[handler_sd] 추천 #{idx} 생성 실패: {e}")

    return all_results


def _extract_rag_keywords(rag_context: str, style_name: str) -> str:
    """RAG 컨텍스트에서 해당 스타일 관련 키워드를 추출."""
    style_lower = style_name.lower()
    keywords = []

    for line in rag_context.split("\n"):
        line_lower = line.lower()
        # 스타일 태그 라인에서 관련 키워드 추출
        if "스타일 태그:" in line:
            tags = line.split(":", 1)[1].strip()
            tag_list = [t.strip() for t in tags.split(",")]
            for tag in tag_list:
                tag_l = tag.lower().strip("[] ")
                if tag_l and (
                    tag_l in style_lower
                    or style_lower in tag_l
                    or any(w in tag_l for w in style_lower.split())
                ):
                    keywords.append(tag)
        # 요약에서 스타일 관련 트렌드 키워드 추출
        elif "요약:" in line and any(w in line_lower for w in style_lower.split()):
            summary = line.split(":", 1)[1].strip()
            if len(summary) < 200:
                keywords.append(summary)

    # 중복 제거, 최대 3개
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return ", ".join(unique[:3])


# ── RunPod Handler ──────────────────────────────────────────────────────────────

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    # import 에러 체크
    if _IMPORT_ERROR:
        return {"error": f"파이프라인 import 실패:\n{_IMPORT_ERROR}"}

    inp = (job or {}).get("input") or {}

    # ── action 라우팅 ─────────────────────────────────────────────────────
    action = str(inp.get("action", "")).strip().lower()

    # 헬스체크
    if action == "health_check" or _coerce_bool(inp.get("health_check")):
        import torch
        return {
            "status": "ok",
            "cuda": {
                "available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }

    # 트렌드 데이터 최신화
    if action == "refresh_trends":
        return _handle_refresh_trends(inp)

    try:
        # ── 입력 파싱 ────────────────────────────────────────────────────────
        hairstyle_text = str(inp.get("hairstyle_text", "")).strip()
        color_text     = str(inp.get("color_text", "")).strip()
        top_k          = max(1, min(5, int(inp.get("top_k", 3))))
        return_base64  = _coerce_bool(inp.get("return_base64"), default=True)
        return_intermediates = _coerce_bool(inp.get("return_intermediates"), default=False)
        mask_debug_only = _coerce_bool(inp.get("mask_debug_only"), default=False)
        bg_fill_mode   = str(inp.get("bg_fill_mode", "cv2")).strip()  # "cv2" | "sd"
        lora_path = str(inp.get("lora_path", "")).strip() or None
        lora_scale = float(inp.get("lora_scale", 1.0))

        # ── 추천 모드 입력 ─────────────────────────────────────────────────
        face_ratios = inp.get("face_ratios")         # EP1에서 받은 얼굴 비율
        preference = inp.get("preference")           # 구조화된 취향 벡터
        preference_text = str(inp.get("preference_text", "")).strip()
        age = inp.get("age")
        if age is not None:
            age = int(age)

        is_recommend_mode = face_ratios is not None
        recommendations_data = None
        rag_context_str = None

        if is_recommend_mode:
            # ── 추천 기반 생성 모드 ──────────────────────────────────────────
            recommendations_data, rag_context_str, hairstyle_text, color_text = (
                _run_recommendation(
                    face_ratios=face_ratios,
                    preference=preference,
                    preference_text=preference_text,
                    age=age,
                    color_text=color_text,
                    top_k=top_k,
                )
            )
        elif not hairstyle_text and not color_text:
            return {"error": "hairstyle_text 또는 color_text 중 하나 이상 필요합니다. "
                           "또는 face_ratios를 전달하여 추천 모드를 사용하세요."}

        # ── 이미지 로드 ──────────────────────────────────────────────────────
        img_bgr = _load_image_from_input(inp)
        h, w = img_bgr.shape[:2]
        logger.info(
            f"[handler_sd] 입력: {w}×{h}, "
            f"hairstyle='{hairstyle_text}', color='{color_text}', top_k={top_k}, "
            f"recommend_mode={is_recommend_mode}"
        )

        # ── 파이프라인 실행 ───────────────────────────────────────────────────
        pipeline = _get_pipeline()
        pipeline.config.bg_fill_mode = bg_fill_mode
        logger.info(f"[handler_sd] bg_fill_mode={bg_fill_mode}")
        logger.info(f"[handler_sd] mask_debug_only={mask_debug_only}")

        if is_recommend_mode and recommendations_data:
            # 추천 모드: 추천된 각 스타일로 1장씩 생성
            all_results = _generate_per_recommendation(
                pipeline=pipeline,
                img_bgr=img_bgr,
                recommendations=recommendations_data,
                color_text=color_text,
                return_intermediates=return_intermediates,
                lora_path=lora_path,
                lora_scale=lora_scale,
                rag_context=rag_context_str,
            )
        else:
            # 기존 모드: 동일 스타일로 top_k장 생성
            all_results = pipeline.run(
                image=img_bgr,
                hairstyle_text=hairstyle_text,
                color_text=color_text,
                top_k=top_k,
                return_intermediates=return_intermediates,
                lora_path=lora_path,
                lora_scale=lora_scale,
            )

        # ── 결과 직렬화 ───────────────────────────────────────────────────────
        output_results = []
        for r in all_results:
            item: Dict[str, Any] = {
                "rank":       r.rank,
                "seed":       r.seed,
                "clip_score": round(float(r.clip_score), 4),
                "mask_used":  r.mask_used,
            }
            if return_base64:
                item["image_base64"] = _image_to_base64(r.image)
                if r.mask is not None:
                    mask_uint8 = (r.mask * 255).astype(np.uint8)
                    mask_rgb = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
                    item["mask_base64"] = _image_to_base64(mask_rgb)

                    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    overlay = orig_rgb.copy()
                    red_layer = np.zeros_like(overlay)
                    red_layer[:, :, 0] = 255
                    alpha = r.mask[..., np.newaxis]
                    overlay = (overlay * (1 - 0.5 * alpha) + red_layer * 0.5 * alpha).astype(np.uint8)
                    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    item["mask_overlay_base64"] = _image_to_base64(overlay_bgr)

                if r.face_bbox is not None:
                    x1, y1, x2, y2 = r.face_bbox
                    item["face_bbox"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

            # 추천 모드: 어떤 스타일로 생성했는지 메타데이터 추가
            if hasattr(r, "style_meta") and r.style_meta:
                item["recommended_style"] = r.style_meta

            output_results.append(item)

        intermediates: Dict[str, str] = {}
        intermediate_data: Dict[str, Any] = {}
        if return_intermediates and all_results:
            debug_images = all_results[0].debug_images or {}
            for name, dbg_bgr in debug_images.items():
                if mask_debug_only and not _is_mask_debug_image(name):
                    continue
                try:
                    intermediates[name] = _image_to_base64(dbg_bgr, quality=90)
                except Exception as e:
                    logger.warning(f"[handler_sd] intermediate 직렬화 실패({name}): {e}")
            debug_data = all_results[0].debug_data or {}
            if isinstance(debug_data, dict) and debug_data:
                intermediate_data = debug_data

        elapsed = time.time() - t0
        logger.info(f"[handler_sd] 완료: {elapsed:.1f}s, {len(all_results)}개 결과")

        response: Dict[str, Any] = {
            "results":         output_results,
            "elapsed_seconds": round(elapsed, 2),
        }
        if recommendations_data:
            response["recommendations"] = recommendations_data
        if rag_context_str:
            response["rag_context"] = rag_context_str
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
