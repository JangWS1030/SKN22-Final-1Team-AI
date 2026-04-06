"""
MirrAI SD Inpainting Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

완전 생성형(Text→Hair) 파이프라인.

아키텍처:
  ┌─────────────────────────────────────────────────────────────┐
  │  입력: 사진 + hairstyle_text + color_text                    │
  ├─────────────────────────────────────────────────────────────┤
  │  [1] MediaPipe FaceDetection → 얼굴 bbox + landmarks        │
  │  [2] SegFace → base hair/face/cloth mask (원본 해상도)      │
  │  [3] SAM2   → 정밀 hair mask 보정 (point + text prompt)     │
  │  [4] Canny edge → ControlNet conditioning (얼굴 구조 보존)  │
  │  [5] face crop → IP-Adapter conditioning (얼굴 identity)    │
  │  [6] SD 1.5 Inpainting + ControlNet → hair 영역 생성        │
  │  [7] Composite → 원본 얼굴 유지 + 생성 헤어 합성             │
  └─────────────────────────────────────────────────────────────┘
  출력: top-k 결과 이미지 (각기 다른 seed)

모델:
  - SegFace(custom): siik/segface_hair_khairstyle
  - SegFace(base):   kartiknarayan/SegFace
  - SAM2:    pretrained_models/sam2.pt  (기존 모델 재사용)
  - SD Inpaint: runwayml/stable-diffusion-inpainting (HF Hub)
  - ControlNet: lllyasviel/control_v11p_sd15_canny   (HF Hub)
  - IP-Adapter: h94/IP-Adapter / ip-adapter-plus-face_sd15.bin (HF Hub)
"""

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

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
try:
    from utils.env_loader import load_project_dotenv
except Exception:
    load_project_dotenv = None
try:
    from utils.trend_prompt import resolve_generation_request
except Exception:
    resolve_generation_request = None

if load_project_dotenv is not None:
    load_project_dotenv()


def _clean_optional_env_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None

# ── HuggingFace 모델 ID ────────────────────────────────────────────────────────
SD_INPAINT_MODEL_ID   = "runwayml/stable-diffusion-inpainting"
CONTROLNET_MODEL_ID   = "lllyasviel/control_v11p_sd15_canny"
IP_ADAPTER_REPO_ID    = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT     = "ip-adapter-plus-face_sd15.bin"
DEFAULT_RUNTIME_LORA_HF_REPO_ID = "siik/mirrai-hair-swap-stage4-garment-reveal-lora-20260330"
DEFAULT_RUNTIME_LORA_HF_FILENAME = "pytorch_lora_weights.safetensors"
DEFAULT_SEGFACE_HF_REPO_ID    = "siik/segface_hair_khairstyle"
DEFAULT_SEGFACE_HF_SUBFOLDER  = ""
DEFAULT_SEGFACE_HF_FILENAME   = "best.pt"
DEFAULT_SEGFACE_BASE_HF_REPO_ID    = "kartiknarayan/SegFace"
DEFAULT_SEGFACE_BASE_HF_SUBFOLDER  = "swinb_celeba_512"
DEFAULT_SEGFACE_BASE_HF_FILENAME   = "model_299.pt"
DEFAULT_SEGFACE_MODEL_VARIANT = "swin_base"
DEFAULT_SEGFACE_INPUT_RES     = 512

# ── SegFace 설정 ───────────────────────────────────────────────────────────────
HAIR_CLASS_IDX   = 14
GLASS_CLASS_IDX  = 15
EARRING_CLASS_IDX = 17
NECKLACE_CLASS_IDX = 18
# 0: bg, 1: neck, 2: face, 3: cloth, 4: r_ear, 5: l_ear, 6: r_bro, 7: l_bro, 
# 8: r_eye, 9: l_eye, 10: nose, 11: inner_mouth, 12: lower_lip, 13: upper_lip
# 얼굴 내부 및 목/귀 클래스 포함
FACE_CLASS_IDXS  = frozenset([1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18])
# 17: earring, 18: necklace → SD가 귀걸이/목걸이 임의 생성하는 문제 방지
CLOTH_CLASS_IDX  = 3   # SegFace class 3 = cloth → hair mask에서 제거해 옷 영역 보호

# ── SD 생성 해상도 ─────────────────────────────────────────────────────────────
SD_SIZE = 512   # SD 1.5 native resolution

# ── 공통 네거티브 프롬프트 ─────────────────────────────────────────────────────
_NEGATIVE_BASE = (
    "ugly, deformed, blurry, low quality, bad anatomy, distorted face, "
    "distorted hair, bald patch, artifacts, watermark, signature, "
    "cartoon, anime, illustration, painting, drawing"
)

_COMMON_STYLE_BLOCK_NEGATIVE = (
    "earring, earrings, drop earrings, dangling earrings, jewelry, ear accessories, necklace, accessories, piercings, "
    "dangling side locks, loose dangling side locks, long dangling side locks, dangling face-framing strands, "
    "dangling lower side tails, loose side tendrils touching clothing, side locks touching shoulders or clothing"
)

# ── 헤어 길이 키워드 ────────────────────────────────────────────────────────────
_SHORT_HAIR_KEYWORDS = frozenset([
    "short", "bob", "pixie", "buzz", "hush", "crop", "cropped",
    "undercut", "crew cut", "crew", "fade", "taper", "bowl", "chin length", "chin-length",
    "above ear", "above shoulder", "ear length", "single",
    "comma hair", "comma", "dandy cut", "dandy",
    "regent cut", "regent", "side part", "side-part", "swept-back", "swept back",
    "단발", "숏컷", "픽시",
])
_MEDIUM_HAIR_KEYWORDS = frozenset([
    "lob", "midi", "medium", "shoulder length", "shoulder-length",
    "collarbone", "clavicle", "mid length", "mid-length",
    "wolf cut", "soft mullet", "mullet", "baby mullet", "mini mullet",
    "two block", "two-block", "comma hair", "comma", "dandy cut", "dandy",
    "regent cut", "regent", "side part", "side-part", "swept-back", "swept back",
    "afro", "rounded afro", "curly afro", "coily", "coils", "tight curl", "tight curls",
    "shorter back and sides", "back and sides",
])

_MALE_SUBJECT_HINTS = frozenset([
    "male", "man", "men", "boy", "masculine", "gentleman", "guy",
    "남자", "남성",
])
_FEMALE_SUBJECT_HINTS = frozenset([
    "female", "woman", "women", "girl", "feminine", "lady",
    "여자", "여성",
])
_MALE_STYLE_HINTS = frozenset([
    "mullet", "wolf cut", "soft mullet", "baby mullet", "mini mullet",
    "crop", "cropped", "buzz", "crew", "fade", "taper", "undercut",
    "two block", "two-block", "comma", "dandy", "regent",
    "barber", "side part", "side-part", "swept-back", "swept back",
    "afro", "coily", "coils", "tight curl", "tight curls",
    "shorter back and sides", "back and sides",
])
_FEMALE_STYLE_HINTS = frozenset([
    "bob", "lob", "bixie", "pixie bob", "hydro bob",
    "ponytail", "braid", "bun", "updo",
])

_NO_COLOR_HINTS = frozenset([
    "", "none", "no color", "same", "original", "default",
    "원본", "기존", "유지", "없음",
])

# RGB 기준 타겟 컬러 (근사값)
_HAIR_COLOR_TARGET_RGB: List[Tuple[str, Tuple[int, int, int]]] = [
    ("ash beige", (173, 158, 136)),
    ("ash brown", (111, 92, 80)),
    ("ash blonde", (192, 176, 146)),
    ("ash black", (58, 58, 62)),
    ("ash gray", (124, 128, 134)),
    ("ash grey", (124, 128, 134)),
    ("ash", (128, 126, 124)),
    ("black", (44, 41, 39)),
    ("dark brown", (82, 62, 50)),
    ("brown", (98, 74, 58)),
    ("beige", (174, 153, 128)),
    ("blonde", (193, 166, 121)),
    ("silver", (170, 174, 182)),
    ("gray", (132, 132, 132)),
    ("grey", (132, 132, 132)),
    ("red", (128, 56, 45)),
    ("auburn", (120, 63, 48)),
    ("pink", (170, 112, 132)),
    ("blue", (82, 95, 138)),
]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SDInpaintConfig:
    """SD Inpainting 파이프라인 설정"""
    # SD 생성 파라미터
    num_inference_steps: int = 30
    controlnet_conditioning_scale: float = 0.3   # 낮춰야 텍스트 프롬프트가 먹힘
    ip_adapter_scale: float = 0.35               # 너무 강하면 원본 헤어 유지해버림

    # Canny edge 파라미터
    canny_low: int  = 80
    canny_high: int = 200

    # hair mask dilate (SD 입력용 — 경계 확장, 잔머리 커버용으로 넉넉하게)
    # 얼굴 내부 보호는 SegFace face_region_mask 로 픽셀 단위 처리함
    mask_dilate_px: int = 30

    # 씨드 리스트 — None 이면 요청마다 랜덤 생성 (권장), 고정값 지정도 가능
    seeds: Optional[List[int]] = None

    # 디바이스 / dtype
    device: str = "cuda"
    dtype: str = "float16"
    lora_path: Optional[str] = None
    lora_scale: float = 1.0

    # SAM2 사용 여부
    use_sam2: bool = True

    # hair mask refinement mode
    #   "sam2": current default refinement
    #   "segface_priority": keep SegFace core and let SAM2 adjust only near the boundary
    #   "segface_only": skip SAM2 refinement and use SegFace mask only
    mask_refine_mode: str = "sam2"

    # 얼굴 랜드마크/메쉬 백엔드

    # 후처리 옵션 (현재 파이프라인에서는 기본 alpha blend 사용)
    use_clip_ranking: bool = False   # 향후 CLIP 랭킹 확장용
    use_color_match:  bool = False   # 향후 LAB 색상 매칭 확장용
    use_poisson_blend: bool = False  # 향후 Poisson blend 확장용

    # 배경 채우기 모드 (short/medium 변환 시 긴머리 제거 방법)
    #   "cv2" : cv2.inpaint(NS+TELEA) 블렌딩 (기본, 빠름)
    #   "sd"  : cv2 1차 + SD 복원 보정 2차 (품질↑, 시간↑)
    bg_fill_mode: str = "cv2"

    enable_post_cloth_refine: bool = True

    # salon-photo style portrait standardization
    enable_input_standardization: bool = True
    standardize_face_height_ratio_min: float = 0.22
    standardize_face_width_ratio_min: float = 0.18
    standardized_width: int = 768
    standardized_height: int = 1024
    enable_portrait_reframe: bool = False
    portrait_reframe_face_height_ratio_max: float = 0.40
    portrait_reframe_top_gap_ratio_min: float = 0.06


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SDInpaintResult:
    image: np.ndarray       # H×W×3 BGR (원본 해상도)
    image_pil: Image.Image  # PIL RGB
    seed: int
    rank: int
    mask_used: str          # "sam2" | "sam2_soft" | "segface"
    mask_refine_mode: str = "sam2"
    clip_score: float = 0.0 # CLIP 점수 (현재는 rank 순서, 향후 CLIP 랭킹 확장용)
    mask: Optional[np.ndarray] = None       # H×W float32 디버그용 마스크
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    debug_images: Optional[Dict[str, np.ndarray]] = None    # 디버그용 중간 산출물 (BGR)
    debug_data: Optional[Dict[str, Any]] = None             # 디버그용 중간 메타데이터(JSON)
    style_meta: Optional[Dict[str, Any]] = None             # 추천 모드: 스타일 메타데이터


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MirrAISDPipeline:
    """
    SAM2 + SD Inpainting + ControlNet(Canny) + IP-Adapter 기반 헤어 변환 파이프라인.

    - hair segmentation: SegFace + SAM2 refinement
    - 생성:              SD 1.5 Inpainting + ControlNet canny + IP-Adapter face
    """

    def __init__(self, config: Optional[SDInpaintConfig] = None) -> None:
        self.config = config or SDInpaintConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.dtype = (
            torch.float16 if self.config.dtype == "float16" else torch.bfloat16
        )

        self._segface    = None   # SegFace (Swin-B) custom hair parsing
        self._segface_base = None # SegFace (Swin-B) base parsing for face/cloth
        self._sam2_factory = None  # SAM2 predictor factory (callable)
        self._sd_pipe    = None   # StableDiffusionControlNetInpaintPipeline
        self._mp_face    = None   # MediaPipe FaceDetection
        self._mp_face_mesh = None # MediaPipe FaceMesh
        self._lama       = None   # LaMa large mask inpainting
        self._lama_device: Optional[str] = None
        self._lama_error: Optional[str] = None
        self._segface_load_info: Dict[str, Any] = {}
        self._segface_base_load_info: Dict[str, Any] = {}
        self._segface_custom_binary_hair = False
        self._segface_custom_hair_threshold = 0.5
        self._last_segface_mask_debug: Optional[Dict[str, Any]] = None
        self._active_lora_path: Optional[str] = None
        self._active_lora_scale: float = 1.0
        self._active_lora_adapter_name: Optional[str] = None
        self._loaded     = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """모델 로드 (cold start). 이미 로드된 경우 no-op."""
        if self._loaded:
            return
        logger.info("[SDPipeline] 모델 로딩 시작...")
        self._load_segface()
        self._load_sam2()
        self._load_mediapipe()
        self._load_sd_pipeline()
        self._load_lama()
        self._loaded = True
        logger.info("[SDPipeline] 모든 모델 로드 완료")

    def run(
        self,
        image: np.ndarray,     # BGR, any resolution
        hairstyle_text: str,
        color_text: str,
        top_k: int = 3,
        return_intermediates: bool = False,
        mask_refine_mode: Optional[str] = None,
        subject_gender: Optional[str] = None,
        lora_path: Optional[str] = None,
        lora_scale: Optional[float] = None,
        sd_prompt_data: Optional[Dict[str, Any]] = None,
    ) -> List[SDInpaintResult]:
        """
        헤어 스타일 변환 실행.

        Args:
            image:          입력 이미지 (BGR numpy)
            hairstyle_text: 사용자 헤어스타일 텍스트 (런타임에 llm_refined_trends 기반 보강)
            color_text:     헤어 컬러 텍스트
            top_k:          반환 결과 수 (기본 3)
            return_intermediates: 중간 산출물 디버그 이미지 포함 여부
            sd_prompt_data: DB에서 가져온 SD 프롬프트 데이터
                            {"sd_positive", "sd_negative", "sd_guidance"}

        Returns:
            SDInpaintResult 리스트 (rank 0이 first)
        """
        if not self._loaded:
            self.load()
        self._apply_runtime_lora(lora_path=lora_path, lora_scale=lora_scale)

        requested_top_k = top_k

        # 시드 결정: config에 고정값 있으면 사용, 없으면 매 요청마다 랜덤 생성
        if self.config.seeds:
            seeds = self.config.seeds[:top_k]
        else:
            seeds = [random.randint(0, 2**31 - 1) for _ in range(top_k)]
        logger.info(f"[SDPipeline] seeds={seeds}")

        image_bgr = image.copy()
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        requested_hairstyle_text = " ".join(str(hairstyle_text or "").strip().split())
        requested_color_text = self._normalize_color_text(color_text)
        requested_gender_mode = self._infer_subject_gender(
            requested_hairstyle_text,
            subject_gender=subject_gender,
        )
        trend_request = None
        effective_hairstyle_text = requested_hairstyle_text
        effective_color_text = requested_color_text
        if resolve_generation_request is not None and (requested_hairstyle_text or requested_color_text):
            try:
                trend_request = resolve_generation_request(
                    requested_hairstyle_text,
                    requested_color_text,
                    top_k=3,
                )
                resolved_style = " ".join(
                    str(trend_request.resolved_hairstyle_text or "").strip().split()
                )
                resolved_color = self._normalize_color_text(trend_request.resolved_color_text)
                resolved_gender_mode = self._infer_subject_gender(
                    resolved_style,
                    subject_gender=None,
                ) if resolved_style else requested_gender_mode
                use_resolved_style = bool(resolved_style)
                if use_resolved_style and requested_gender_mode == "male" and resolved_gender_mode != "male":
                    use_resolved_style = False
                    logger.info(
                        "[SDPipeline] trend resolved style ignored due to gender mismatch: "
                        "requested_gender=%s resolved_gender=%s resolved_style='%s'",
                        requested_gender_mode,
                        resolved_gender_mode,
                        resolved_style,
                    )
                if use_resolved_style:
                    effective_hairstyle_text = resolved_style
                if resolved_color:
                    effective_color_text = resolved_color
                logger.info(
                    "[SDPipeline] trend resolution: requested_style='%s' -> resolved_style='%s', matches=%d",
                    requested_hairstyle_text,
                    effective_hairstyle_text,
                    len(trend_request.matches),
                )
                if trend_request.matches:
                    logger.info(
                        "[SDPipeline] top trend match: %s (score=%.3f, source=%s)",
                        trend_request.matches[0].trend_name,
                        trend_request.matches[0].score,
                        trend_request.matches[0].source,
                    )
            except Exception as e:
                logger.warning(f"[SDPipeline] trend resolution skipped: {e}")
                trend_request = None
        normalized_color_text = self._normalize_color_text(effective_color_text)
        subject_gender_mode = self._infer_subject_gender(
            effective_hairstyle_text,
            subject_gender=subject_gender,
        )
        has_color_request = bool(normalized_color_text)
        target_hair_lab = self._resolve_target_hair_lab(normalized_color_text) if has_color_request else None
        if not has_color_request:
            logger.info("[SDPipeline] color_text 미지정 → 원본 머리 톤 유지 모드")
        elif target_hair_lab is None:
            logger.info("[SDPipeline] color_text 파싱 실패 → 색상 재정렬은 스킵")

        H, W = image_bgr.shape[:2]
        debug_images_common: Optional[Dict[str, np.ndarray]] = {} if return_intermediates else None
        debug_data_common: Optional[Dict[str, Any]] = {} if return_intermediates else None
        if debug_data_common is not None:
            debug_data_common["subject_gender"] = subject_gender_mode
        if debug_data_common is not None and trend_request is not None:
            debug_data_common["trend_resolution"] = trend_request.to_debug_dict()

        def _store_mask(name: str, mask: Optional[np.ndarray]) -> None:
            if debug_images_common is None or mask is None:
                return
            m = np.clip(mask, 0.0, 1.0)
            m_u8 = (m * 255).astype(np.uint8)
            debug_images_common[name] = cv2.cvtColor(m_u8, cv2.COLOR_GRAY2BGR)

        def _store_rgb(name: str, rgb_img: np.ndarray) -> None:
            if debug_images_common is None:
                return
            debug_images_common[name] = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        if debug_images_common is not None:
            debug_images_common["pipeline_input_image"] = image_bgr.copy()

        # ── Step 1: 얼굴 검출 ────────────────────────────────────────────────
        face_obs = self._detect_face(img_rgb)
        if face_obs is None:
            raise ValueError("얼굴을 검출할 수 없습니다.")
        face_bbox = face_obs  # (x1, y1, x2, y2)
        standardized_meta = self._maybe_standardize_input_portrait(img_rgb, face_bbox)
        if standardized_meta.get("applied"):
            std_rgb = standardized_meta.get("image_rgb")
            if isinstance(std_rgb, np.ndarray):
                img_rgb = std_rgb
                image_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                H, W = image_bgr.shape[:2]
                face_obs = self._detect_face(img_rgb)
                if face_obs is None:
                    raise ValueError("표준화 후 얼굴을 다시 검출할 수 없습니다.")
                face_bbox = face_obs
                if debug_images_common is not None:
                    debug_images_common["pipeline_standardized_input_image"] = image_bgr.copy()
                if debug_data_common is not None:
                    debug_data_common["input_standardization"] = {
                        k: v for k, v in standardized_meta.items() if k != "image_rgb"
                    }
        logger.info(f"[SDPipeline] 얼굴 검출: {face_bbox}")
        landmark_obs = self._detect_landmark_data(img_rgb, face_bbox)
        landmark_face_mask = landmark_obs.get("face_mask")

        if landmark_obs.get("detected"):
            logger.info(
                "[SDPipeline] mediapipe landmarks, points=%s",
                landmark_obs.get("landmarks_count"),
            )
            landmark_debug_images = landmark_obs.get("debug_images")
            if debug_images_common is not None and isinstance(landmark_debug_images, dict):
                debug_images_common.update(landmark_debug_images)

            landmark_debug_data = landmark_obs.get("debug_data")
            if debug_data_common is not None and isinstance(landmark_debug_data, dict):
                debug_data_common["mediapipe_face_mesh"] = landmark_debug_data

            if debug_images_common is not None and isinstance(landmark_face_mask, np.ndarray):
                _store_mask("mediapipe_face_protect_mask", landmark_face_mask)
        elif debug_data_common is not None:
            debug_data_common["mediapipe_face_mesh"] = {"detected": False}

        # ── Step 2: SegFace base hair mask + 얼굴 픽셀 마스크 + 옷 마스크 ──────
        hair_mask_base, face_region_mask, cloth_mask = self._segface_hair_mask(img_rgb, face_bbox)
        face_region_mask = self._sanitize_face_region_mask(
            face_region_mask,
            face_bbox,
            landmark_face_mask=landmark_face_mask,
        )
        cloth_mask = self._sanitize_cloth_mask(img_rgb, cloth_mask, hair_mask_base, face_bbox)
        _store_mask("segface_hair_mask", hair_mask_base)
        _store_mask("segface_face_region_mask", face_region_mask)
        _store_mask("segface_cloth_mask", cloth_mask)
        segface_debug = self._last_segface_mask_debug or {}
        custom_hair_mask = segface_debug.get("custom_hair_mask")
        base_hair_mask = segface_debug.get("base_hair_mask")
        base_hair_support_mask = segface_debug.get("base_hair_support_mask")
        glasses_mask = segface_debug.get("glasses_mask")
        earring_mask = segface_debug.get("earring_mask")
        necklace_mask = segface_debug.get("necklace_mask")
        sparse_dark_cloth_support_mask = segface_debug.get("sparse_dark_cloth_support_mask")
        subject_cloth_anchor_mask = segface_debug.get("subject_cloth_anchor_mask")
        subject_cloth_filtered_mask = segface_debug.get("subject_cloth_filtered_mask")
        if isinstance(custom_hair_mask, np.ndarray):
            _store_mask("segface_custom_hair_mask", custom_hair_mask)
        if isinstance(base_hair_mask, np.ndarray):
            _store_mask("segface_base_hair_mask", base_hair_mask)
        if isinstance(base_hair_support_mask, np.ndarray):
            _store_mask("segface_base_hair_support_mask", base_hair_support_mask)
        if isinstance(glasses_mask, np.ndarray):
            _store_mask("segface_glasses_mask", glasses_mask)
        if isinstance(earring_mask, np.ndarray):
            _store_mask("segface_earring_mask", earring_mask)
        if isinstance(necklace_mask, np.ndarray):
            _store_mask("segface_necklace_mask", necklace_mask)
        if isinstance(sparse_dark_cloth_support_mask, np.ndarray):
            _store_mask("segface_sparse_dark_cloth_support_mask", sparse_dark_cloth_support_mask)
        if isinstance(subject_cloth_anchor_mask, np.ndarray):
            _store_mask("segface_subject_cloth_anchor_mask", subject_cloth_anchor_mask)
        if isinstance(subject_cloth_filtered_mask, np.ndarray):
            _store_mask("segface_subject_cloth_filtered_mask", subject_cloth_filtered_mask)
        subject_cloth_anchor_for_post = None
        if (
            isinstance(subject_cloth_anchor_mask, np.ndarray)
            and subject_cloth_anchor_mask.shape == (H, W)
        ):
            subject_cloth_anchor_for_post = np.clip(
                subject_cloth_anchor_mask.astype(np.float32),
                0.0,
                1.0,
            )
        if debug_data_common is not None and segface_debug.get("meta"):
            debug_data_common["segface_mask_debug"] = segface_debug["meta"]

        # ── Step 3: SAM2 refinement ───────────────────────────────────────────
        hair_mask, mask_source, mask_refine_mode_used = self._refine_with_sam2(
            img_rgb,
            hair_mask_base,
            face_bbox,
            effective_hairstyle_text,
            mask_refine_mode=mask_refine_mode,
        )
        logger.info(
            f"[SDPipeline] hair mask source={mask_source}, refine_mode={mask_refine_mode_used}, "
            f"pixels={hair_mask.sum():.0f}"
        )
        _store_mask(f"{mask_source}_refined_hair_mask", hair_mask)
        if debug_data_common is not None:
            debug_data_common["mask_refine_mode"] = {
                "requested": self._resolve_mask_refine_mode(mask_refine_mode),
                "used": mask_refine_mode_used,
                "mask_used": mask_source,
            }

        if hair_mask.sum() < 300:
            raise ValueError("머리카락 영역이 너무 작습니다.")

        # ── Step 3-b: 헤어 길이 분류 ─────────────────────────────────────────
        hair_length = self._classify_hair_length(effective_hairstyle_text)
        if debug_data_common is not None:
            debug_data_common["hair_length"] = hair_length
        logger.info(
            f"[SDPipeline] 헤어 길이 분류: {hair_length}, subject_gender={subject_gender_mode}"
        )
        if hair_length == "short" and len(seeds) < 5:
            extra = 5 - len(seeds)
            seeds.extend(random.randint(0, 2**31 - 1) for _ in range(extra))
            logger.info(
                f"[SDPipeline] short internal candidate expansion: requested={requested_top_k}, internal={len(seeds)}"
            )
        elif subject_gender_mode == "male" and hair_length in ("short", "medium") and len(seeds) < 3:
            extra = 3 - len(seeds)
            seeds.extend(random.randint(0, 2**31 - 1) for _ in range(extra))
            logger.info(
                f"[SDPipeline] male internal candidate expansion: requested={requested_top_k}, internal={len(seeds)}"
            )
        landmark_debug_data = landmark_obs.get("debug_data")
        if not isinstance(landmark_debug_data, dict):
            landmark_debug_data = None

        lower_tail_support_mask = self._build_lower_hair_tail_support_mask(
            img_rgb=img_rgb,
            hair_mask=hair_mask,
            cloth_mask=cloth_mask,
            face_bbox=face_bbox,
            hair_length=hair_length,
        )
        if float(lower_tail_support_mask.sum()) > 0.0:
            hair_mask = np.maximum(hair_mask, lower_tail_support_mask).astype(np.float32)
            logger.info(
                "[SDPipeline] lower tail support added: pixels=%.0f merged_pixels=%.0f",
                lower_tail_support_mask.sum(),
                hair_mask.sum(),
            )
        _store_mask("pipeline_lower_tail_support_mask", lower_tail_support_mask)

        protect_mask_for_sd = self._build_generation_protect_mask(
            face_region_mask,
            face_bbox=face_bbox,
            hair_length=hair_length,
        )
        protect_mask_for_removal = self._build_removal_protect_mask(
            face_region_mask,
            face_bbox=face_bbox,
            hair_length=hair_length,
        )
        accessory_protect_mask = self._build_accessory_protect_mask(
            face_bbox=face_bbox,
            earring_mask=earring_mask,
            necklace_mask=necklace_mask,
            hair_length=hair_length,
        )
        if float(accessory_protect_mask.sum()) > 0.0:
            protect_mask_for_sd = np.maximum(protect_mask_for_sd, accessory_protect_mask).astype(np.float32)
            protect_mask_for_removal = np.maximum(protect_mask_for_removal, accessory_protect_mask).astype(np.float32)
            _store_mask("pipeline_accessory_protect_mask", accessory_protect_mask)
        _store_mask("pipeline_generation_protect_mask", protect_mask_for_sd)
        _store_mask("pipeline_removal_protect_mask", protect_mask_for_removal)

        # ── Step 3-c: SegFace 얼굴 픽셀 제거 (bbox 직사각형 대신 픽셀 단위 보정) ─
        hair_mask_before_face_protect = hair_mask.copy()
        hair_mask_for_removal = np.clip(hair_mask - protect_mask_for_removal, 0.0, 1.0)
        hair_mask = np.clip(hair_mask - protect_mask_for_sd, 0.0, 1.0)
        bangs_restore_for_removal = self._build_bangs_recovery_mask(
            hair_mask_before_face_protect,
            protect_mask_for_removal,
            face_bbox,
            landmark_debug_data=landmark_debug_data,
            hair_length=hair_length,
        )
        bangs_restore_for_sd = self._build_bangs_recovery_mask(
            hair_mask_before_face_protect,
            protect_mask_for_sd,
            face_bbox,
            landmark_debug_data=landmark_debug_data,
            hair_length=hair_length,
        )
        if float(bangs_restore_for_removal.sum()) > 0.0:
            hair_mask_for_removal = np.maximum(hair_mask_for_removal, bangs_restore_for_removal).astype(np.float32)
        if float(bangs_restore_for_sd.sum()) > 0.0:
            hair_mask = np.maximum(hair_mask, bangs_restore_for_sd).astype(np.float32)
        # short/medium 긴머리 제거 단계에서는 "옷 위로 떨어진 머리카락"도 지워야 하므로
        # cloth 제거 전 마스크를 별도로 보관한다.
        _store_mask("pipeline_bangs_recovery_mask_removal", bangs_restore_for_removal)
        _store_mask("pipeline_bangs_recovery_mask_generation", bangs_restore_for_sd)
        _store_mask("pipeline_hair_mask_face_protected", hair_mask_for_removal)
        logger.info(
            f"[SDPipeline] 얼굴 픽셀 제거 완료, gen_px={hair_mask.sum():.0f}, removal_px={hair_mask_for_removal.sum():.0f}"
        )

        # ── Step 3-d: SegFace 옷 픽셀 제거 (옷이 바뀌는 문제 방지) ────────────
        # expand 전에 먼저 제거해야 옷 영역이 마스크 확장에 영향받지 않음
        cloth_dilate_px = max(
            5,
            int(
                self.config.mask_dilate_px
                * (0.20 if hair_length == "short" else 0.28 if hair_length == "medium" else 0.40)
            ),
        )
        cloth_mask_dilated = self._dilate_mask_with_px(
            cloth_mask,
            cloth_dilate_px,
        )
        cloth_ratio_limit = 0.10 if hair_length == "short" else 0.14 if hair_length == "medium" else 0.18
        if self._mask_ratio(cloth_mask_dilated) > cloth_ratio_limit:
            logger.info(
                "[SDPipeline] dilated cloth mask disabled: ratio=%.4f",
                self._mask_ratio(cloth_mask_dilated),
            )
            cloth_mask_dilated = np.zeros_like(cloth_mask_dilated, dtype=np.float32)
        cloth_restore_mask_for_post = cloth_mask_dilated.astype(np.float32)
        use_short_dark_cloth_anchor_fallback = False
        if (
            hair_length == "short"
            and isinstance(subject_cloth_anchor_mask, np.ndarray)
            and subject_cloth_anchor_mask.shape == cloth_mask_dilated.shape
        ):
            anchor_mask_f = np.clip(subject_cloth_anchor_mask.astype(np.float32), 0.0, 1.0)
            cloth_ratio = self._mask_ratio(cloth_mask_dilated)
            anchor_ratio = self._mask_ratio(anchor_mask_f)
            cloth_visible_u8 = (np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8)
            dark_cloth_ready = False
            if int(cloth_visible_u8.sum()) >= 80:
                source_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                source_sat = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
                cloth_gray_median = float(np.median(source_gray[cloth_visible_u8 > 0]))
                cloth_sat_median = float(np.median(source_sat[cloth_visible_u8 > 0]))
                dark_cloth_ready = cloth_gray_median <= 132.0 and cloth_sat_median <= 160.0
            if (
                dark_cloth_ready
                and cloth_ratio < 0.07
                and anchor_ratio > max(0.08, cloth_ratio + 0.05)
            ):
                anchor_u8 = cv2.erode(
                    (anchor_mask_f > 0.04).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                    iterations=1,
                )
                cloth_restore_mask_for_post = np.maximum(
                    cloth_restore_mask_for_post,
                    anchor_u8.astype(np.float32) / 255.0,
                ).astype(np.float32)
                logger.info(
                    "[SDPipeline] short restore cloth fallback enabled: cloth_ratio=%.4f anchor_ratio=%.4f",
                    cloth_ratio,
                    anchor_ratio,
                )
                use_short_dark_cloth_anchor_fallback = True
        hair_mask = np.clip(hair_mask - cloth_mask_dilated, 0.0, 1.0)
        _store_mask("segface_cloth_mask_dilated", cloth_mask_dilated)
        if hair_length == "short":
            _store_mask("pipeline_short_restore_cloth_mask", cloth_restore_mask_for_post)
        _store_mask("pipeline_hair_mask_cloth_protected", hair_mask)
        logger.info(
            f"[SDPipeline] 옷 픽셀 제거 완료, pixels={hair_mask.sum():.0f}"
        )

        # ── Step 3-e: 숏컷/중단발 — 전략 2 (Post-Inpainting) ────────────────
        # 단발/숏컷에서 기존 긴머리 prior가 강하면, 생성 후 잔여 long-hair만
        # 차집합 마스크 기반으로 후처리하는 쪽이 더 안정적이다.
        cutoff_y_for_post: Optional[int] = None
        removal_mask_for_post: Optional[np.ndarray] = None
        shoulder_protect_for_post: Optional[np.ndarray] = None
        neckline_preserve_for_post: Optional[np.ndarray] = None
        lateral_neck_preserve_for_post: Optional[np.ndarray] = None
        torso_cloth_preserve_for_post: Optional[np.ndarray] = None
        bright_cloth_preserve_for_post: Optional[np.ndarray] = None
        shoulder_cloth_release_for_post: Optional[np.ndarray] = None
        artifact_cleanup_mask_for_post: Optional[np.ndarray] = None
        lower_tail_support_for_post: Optional[np.ndarray] = None
        below_bob_generation_block_for_post: Optional[np.ndarray] = None
        below_bob_cloth_restore_for_post: Optional[np.ndarray] = None
        composite_bangs_release_mask = np.zeros((H, W), dtype=np.float32)
        center_chest_strand_mask = np.zeros((H, W), dtype=np.float32)
        center_chest_strand_removal_mask = np.zeros((H, W), dtype=np.float32)
        center_cloth_restore_exclusion_mask = np.zeros((H, W), dtype=np.float32)
        if hair_length in ("short", "medium"):
            head_x1, head_y1, head_x2, head_y2, cutoff_y = self._estimate_head_generation_box(
                image_shape=(H, W),
                face_bbox=face_bbox,
                landmark_face_mask=landmark_face_mask,
                landmark_debug_data=landmark_debug_data,
                hair_length=hair_length,
            )
            cutoff_y_for_post = cutoff_y
            shoulder_protect_for_post = self._build_shoulder_protect_mask(
                cloth_mask=cloth_mask_dilated,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
            )
            neckline_preserve_for_post = self._build_neckline_preserve_mask(
                face_bbox=face_bbox,
                cloth_mask=cloth_mask_dilated,
                hair_length=hair_length,
            )
            lateral_neck_preserve_for_post = self._build_short_lateral_neck_preserve_mask(
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                cloth_mask=cloth_mask_dilated,
                hair_length=hair_length,
            )
            torso_cloth_preserve_for_post = self._build_torso_cloth_preserve_mask(
                face_bbox=face_bbox,
                cloth_mask=cloth_mask_dilated,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            )
            lower_tail_support_for_post = self._build_lower_tail_post_support_mask(
                support_mask=lower_tail_support_mask,
                cloth_mask=cloth_mask_dilated,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            )
            center_chest_strand_mask = self._build_center_chest_strand_support_mask(
                img_rgb=img_rgb,
                cloth_mask=cloth_mask_dilated,
                support_mask=lower_tail_support_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
                anchor_mask=subject_cloth_anchor_for_post,
            )
            if float(center_chest_strand_mask.sum()) > 0.0:
                lower_tail_support_for_post = np.maximum(
                    lower_tail_support_for_post,
                    center_chest_strand_mask,
                ).astype(np.float32)
            if shoulder_protect_for_post.sum() > 0:
                logger.info(
                    "[SDPipeline] 어깨 보호 마스크 적용: "
                    f"pixels={shoulder_protect_for_post.sum():.0f}"
                )
            _store_mask("pipeline_shoulder_protect_mask", shoulder_protect_for_post)
            _store_mask("pipeline_neckline_preserve_mask", neckline_preserve_for_post)
            _store_mask("pipeline_lateral_neck_preserve_mask", lateral_neck_preserve_for_post)
            _store_mask("pipeline_torso_cloth_preserve_mask", torso_cloth_preserve_for_post)
            _store_mask("pipeline_lower_tail_support_post_mask", lower_tail_support_for_post)
            _store_mask("pipeline_center_chest_strand_mask", center_chest_strand_mask)

            # v4 쪽이 더 안정적이었던 핵심:
            # 1) cutoff 아래 long hair를 먼저 실제 hair 기반으로 비운 뒤
            # 2) 생성은 head box 안에서만 제한적으로 수행한다.
            if hair_length == "short":
                base_prior = np.clip(
                    hair_mask_base - face_region_mask,
                    0.0,
                    1.0,
                ).astype(np.float32)
                prior_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                base_prior = cv2.dilate(
                    (base_prior > 0.5).astype(np.uint8) * 255,
                    prior_k,
                    iterations=1,
                )
                base_prior = (base_prior > 0).astype(np.float32)

                removal_seed = np.clip(hair_mask_for_removal * base_prior, 0.0, 1.0)
                tail_below_cutoff = hair_mask_for_removal.copy()
                tail_below_cutoff[:cutoff_y, :] = 0.0
                tail_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 7))
                tail_below_cutoff = cv2.dilate(
                    (tail_below_cutoff > 0.5).astype(np.uint8) * 255,
                    tail_k,
                    iterations=1,
                ).astype(np.float32) / 255.0

                if removal_seed.sum() > 120:
                    removal_mask = np.clip(
                        np.maximum(removal_seed, tail_below_cutoff),
                        0.0,
                        1.0,
                    )
                else:
                    removal_mask = hair_mask_for_removal.copy()
            else:
                removal_mask = hair_mask_for_removal.copy()

            removal_mask[:cutoff_y, :] = 0.0
            bright_cloth_preserve_for_post = self._build_bright_cloth_preserve_mask(
                img_rgb=img_rgb,
                removal_mask=removal_mask,
                cloth_mask=cloth_mask_dilated,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            )
            _store_mask("pipeline_bright_cloth_preserve_mask", bright_cloth_preserve_for_post)

            hair_below = (hair_mask_for_removal > 0.5).astype(np.uint8)
            hair_below[:cutoff_y, :] = 0
            face_x1, _, face_x2, _ = [int(v) for v in face_bbox]
            face_w_for_tail = max(int(face_x2 - face_x1), 1)
            if hair_length == "short":
                tail_left = max(0, int(face_x1 - face_w_for_tail * 1.42))
                tail_right = min(W, int(face_x2 + face_w_for_tail * 1.42))
            else:
                tail_left = max(0, head_x1 - 20)
                tail_right = min(W, head_x2 + 20)
            hair_below[:, :tail_left] = 0
            hair_below[:, tail_right:] = 0
            if int((hair_below > 0).sum()) > 0:
                expand_k = (
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 7))
                    if hair_length == "short"
                    else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 13))
                )
                hair_below_expanded = cv2.dilate(
                    hair_below,
                    expand_k,
                    iterations=1,
                ).astype(np.float32)
                removal_mask = np.maximum(removal_mask, hair_below_expanded)
                logger.info(
                    f"[SDPipeline] removal_mask hair기반 확장({hair_length}): "
                    f"pixels={removal_mask.sum():.0f}"
                )

            close_size = 5 if hair_length == "short" else 11
            remove_close_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (close_size, close_size),
            )
            removal_mask = cv2.morphologyEx(
                removal_mask,
                cv2.MORPH_CLOSE,
                remove_close_k,
            )
            removal_mask = np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

            if shoulder_protect_for_post is not None and shoulder_protect_for_post.shape == (H, W):
                shoulder_protect_weight = 0.30 if hair_length == "short" else 0.85
                removal_mask = np.clip(
                    removal_mask - shoulder_protect_for_post * shoulder_protect_weight,
                    0.0,
                    1.0,
                )
            if neckline_preserve_for_post is not None and neckline_preserve_for_post.shape == (H, W):
                removal_mask = np.clip(
                    removal_mask - neckline_preserve_for_post * (0.28 if hair_length == "short" else 0.26),
                    0.0,
                    1.0,
                )
            if torso_cloth_preserve_for_post is not None and torso_cloth_preserve_for_post.shape == (H, W):
                removal_mask = np.clip(
                    removal_mask - torso_cloth_preserve_for_post * (0.76 if hair_length == "short" else 0.34),
                    0.0,
                    1.0,
                )
            if bright_cloth_preserve_for_post is not None and bright_cloth_preserve_for_post.shape == (H, W):
                removal_mask = np.clip(
                    removal_mask - bright_cloth_preserve_for_post * (0.92 if hair_length == "short" else 0.44),
                    0.0,
                    1.0,
                )
            if hair_length == "short":
                removal_mask = self._filter_short_torso_box_mask(
                    img_rgb=img_rgb,
                    removal_mask=removal_mask,
                    cloth_mask=cloth_mask_dilated,
                    support_mask=lower_tail_support_for_post,
                    center_support_mask=center_chest_strand_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                )
                shoulder_cloth_release_for_post = self._build_shoulder_cloth_restore_mask(
                    removal_mask=removal_mask,
                    cloth_mask=cloth_mask_dilated,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                    protect_mask=protect_mask_for_sd,
                )
                if shoulder_cloth_release_for_post is not None and shoulder_cloth_release_for_post.shape == (H, W):
                    removal_mask = np.clip(
                        removal_mask - shoulder_cloth_release_for_post * 0.98,
                        0.0,
                        1.0,
                    )
            if float(center_chest_strand_mask.sum()) > 0.0:
                center_chest_strand_removal_mask = cv2.dilate(
                    (center_chest_strand_mask > 0.08).astype(np.uint8) * 255,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (7, 21) if hair_length == "short" else (5, 15),
                    ),
                    iterations=1,
                ).astype(np.float32) / 255.0
                removal_mask = np.maximum(removal_mask, center_chest_strand_removal_mask).astype(np.float32)
                if (
                    hair_length == "short"
                    and subject_gender_mode != "male"
                    and cloth_restore_mask_for_post is not None
                    and cloth_restore_mask_for_post.shape == (H, W)
                ):
                    face_x1, face_y1, face_x2, face_y2 = [int(v) for v in face_bbox]
                    face_w = max(face_x2 - face_x1, 1)
                    face_h = max(face_y2 - face_y1, 1)
                    center_cloth_restore_exclusion_u8 = cv2.dilate(
                        (center_chest_strand_removal_mask > 0.08).astype(np.uint8) * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 45)),
                        iterations=1,
                    )
                    center_cloth_restore_gate_u8 = np.zeros((H, W), dtype=np.uint8)
                    center_gate_half_w = max(24, int(face_w * 0.35))
                    center_gate_left = max(0, int(0.5 * (face_x1 + face_x2)) - center_gate_half_w)
                    center_gate_right = min(W, int(0.5 * (face_x1 + face_x2)) + center_gate_half_w)
                    center_gate_top = max(0, int(cutoff_y + face_h * 0.02))
                    center_gate_bottom = min(H, int(cutoff_y + face_h * 1.28))
                    if center_gate_top < center_gate_bottom and center_gate_left < center_gate_right:
                        center_cloth_restore_gate_u8[
                            center_gate_top:center_gate_bottom,
                            center_gate_left:center_gate_right,
                        ] = 255
                        center_cloth_restore_exclusion_u8 = cv2.bitwise_and(
                            center_cloth_restore_exclusion_u8,
                            center_cloth_restore_gate_u8,
                        )
                    if int((center_cloth_restore_exclusion_u8 > 0).sum()) >= 36:
                        center_cloth_restore_exclusion = cv2.GaussianBlur(
                            center_cloth_restore_exclusion_u8.astype(np.float32) / 255.0,
                            (0, 0),
                            sigmaX=3.8,
                            sigmaY=7.0,
                        ).astype(np.float32)
                        center_cloth_restore_exclusion_mask = np.maximum(
                            center_cloth_restore_exclusion_mask,
                            center_cloth_restore_exclusion,
                        ).astype(np.float32)
                        cloth_restore_mask_for_post = np.clip(
                            cloth_restore_mask_for_post.astype(np.float32) * (1.0 - center_cloth_restore_exclusion),
                            0.0,
                            1.0,
                        )
                        _store_mask(
                            "pipeline_center_cloth_restore_exclusion_mask",
                            center_cloth_restore_exclusion,
                        )
                        if hair_length == "short":
                            _store_mask("pipeline_short_restore_cloth_mask", cloth_restore_mask_for_post)
            lower_tail_removal_extension = np.zeros((H, W), dtype=np.float32)
            if lower_tail_support_for_post is not None and lower_tail_support_for_post.shape == (H, W):
                lower_tail_removal_extension = self._build_lower_tail_removal_extension_mask(
                    support_mask=lower_tail_support_for_post,
                    removal_mask=removal_mask,
                    cloth_mask=cloth_mask_dilated,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                )
                if float(lower_tail_removal_extension.sum()) > 0.0:
                    removal_mask = np.maximum(removal_mask, lower_tail_removal_extension).astype(np.float32)
                    logger.info(
                        "[SDPipeline] lower tail removal extension added: pixels=%.0f merged_pixels=%.0f",
                        lower_tail_removal_extension.sum(),
                        removal_mask.sum(),
                    )
            if hair_length == "short":
                removal_mask = self._filter_short_torso_box_mask(
                    img_rgb=img_rgb,
                    removal_mask=removal_mask,
                    cloth_mask=cloth_mask_dilated,
                    support_mask=lower_tail_support_for_post,
                    center_support_mask=center_chest_strand_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                )
                shoulder_cloth_release_for_post = self._build_shoulder_cloth_restore_mask(
                    removal_mask=removal_mask,
                    cloth_mask=cloth_mask_dilated,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                    protect_mask=protect_mask_for_sd,
                )
                if shoulder_cloth_release_for_post is not None and shoulder_cloth_release_for_post.shape == (H, W):
                    removal_mask = np.clip(
                        removal_mask - shoulder_cloth_release_for_post,
                        0.0,
                        1.0,
                    )
                removal_mask = self._restrict_short_removal_to_tail_lanes(
                    removal_mask=removal_mask,
                    support_mask=lower_tail_support_for_post,
                    center_support_mask=center_chest_strand_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                )
                if (
                    lateral_neck_preserve_for_post is not None
                    and lateral_neck_preserve_for_post.shape == (H, W)
                ):
                    removal_mask = np.clip(
                        removal_mask - lateral_neck_preserve_for_post * 0.92,
                        0.0,
                        1.0,
                    )
            removal_mask_for_post = removal_mask.copy()

            _store_mask("pipeline_lower_tail_removal_extension_mask", lower_tail_removal_extension)
            _store_mask("pipeline_center_chest_strand_removal_mask", center_chest_strand_removal_mask)
            _store_mask("pipeline_shoulder_cloth_release_mask", shoulder_cloth_release_for_post)
            gen_mask = np.zeros((H, W), dtype=np.float32)
            short_generation_seed_mask_for_debug: Optional[np.ndarray] = None
            if hair_length == "short":
                x1f, y1f, x2f, y2f = [int(v) for v in face_bbox]
                face_w = max(int(x2f - x1f), 1)
                face_h = max(int(y2f - y1f), 1)
                face_cx = int(0.5 * (x1f + x2f))
                male_short_compact_style = (
                    subject_gender_mode == "male"
                    and self._is_compact_male_short_style(effective_hairstyle_text)
                )
                male_short_volume_boost = subject_gender_mode == "male" and not male_short_compact_style
                male_short_dominant_side: Optional[str] = None
                seed_top = max(0, int(head_y1))
                seed_bottom = min(H, int(min(cutoff_y + face_h * 0.06, y2f + face_h * 0.26)))
                seed_left = max(0, int(x1f - face_w * (0.56 if male_short_compact_style else 0.72)))
                seed_right = min(W, int(x2f + face_w * (0.56 if male_short_compact_style else 0.72)))
                short_seed_u8 = np.zeros((H, W), dtype=np.uint8)

                if seed_top < seed_bottom and seed_left < seed_right:
                    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
                    corridor_u8[seed_top:seed_bottom, seed_left:seed_right] = 255

                    crown_center_y = int(
                        max(
                            seed_top + 1,
                            min(
                                seed_bottom - 1,
                                y1f + face_h * (
                                    0.17 if male_short_compact_style else (0.08 if male_short_volume_boost else 0.12)
                                ),
                            ),
                        )
                    )
                    crown_axes_y = max(
                        26,
                        int(
                            (seed_bottom - seed_top)
                            * (0.34 if male_short_compact_style else (0.52 if male_short_volume_boost else 0.44))
                        ),
                    )
                    crown_axes_x = max(
                        24,
                        int(face_w * (0.62 if male_short_compact_style else (0.78 if male_short_volume_boost else 0.72))),
                    )
                    cv2.ellipse(
                        short_seed_u8,
                        (face_cx, crown_center_y),
                        (crown_axes_x, crown_axes_y),
                        0,
                        0,
                        360,
                        255,
                        thickness=-1,
                    )

                    side_top = max(
                        seed_top,
                        int(
                            y1f
                            + face_h
                            * (0.08 if male_short_compact_style else (0.02 if male_short_volume_boost else 0.06))
                        ),
                    )
                    side_bottom = min(seed_bottom, int(y2f + face_h * 0.10))
                    side_inner_gap = max(16, int(face_w * (0.20 if male_short_compact_style else 0.18)))
                    side_outer_span = max(
                        22,
                        int(face_w * (0.48 if male_short_compact_style else (0.60 if male_short_volume_boost else 0.56))),
                    )
                    left_outer = max(0, int(face_cx - side_outer_span))
                    left_inner = max(left_outer + 1, int(face_cx - side_inner_gap))
                    right_inner = min(W - 1, int(face_cx + side_inner_gap))
                    right_outer = min(W, int(face_cx + side_outer_span))
                    if side_top < side_bottom:
                        short_seed_u8[side_top:side_bottom, left_outer:left_inner] = 255
                        short_seed_u8[side_top:side_bottom, right_inner:right_outer] = 255

                    upper_prior_u8 = cv2.dilate(
                        (np.clip(base_prior.astype(np.float32), 0.0, 1.0) > 0.10).astype(np.uint8) * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                        iterations=1,
                    )
                    prior_cap_y = min(
                        seed_bottom,
                        int(
                            y1f
                            + face_h
                            * (0.28 if male_short_compact_style else (0.38 if male_short_volume_boost else 0.32))
                        ),
                    )
                    if prior_cap_y < H:
                        upper_prior_u8[prior_cap_y:, :] = 0
                    if male_short_volume_boost:
                        probe_bottom = min(seed_bottom, int(y1f + face_h * 0.26))
                        if seed_top < probe_bottom and seed_left < face_cx < seed_right:
                            left_prior_px = int((upper_prior_u8[seed_top:probe_bottom, seed_left:face_cx] > 0).sum())
                            right_prior_px = int((upper_prior_u8[seed_top:probe_bottom, face_cx:seed_right] > 0).sum())
                            dominant_side_boost = max(10, int(face_w * 0.12))
                            if left_prior_px >= max(24, int(right_prior_px * 1.08)):
                                male_short_dominant_side = "left"
                                extra_left = max(0, left_outer - dominant_side_boost)
                                short_seed_u8[seed_top:side_bottom, extra_left:left_inner] = 255
                            elif right_prior_px >= max(24, int(left_prior_px * 1.08)):
                                male_short_dominant_side = "right"
                                extra_right = min(W, right_outer + dominant_side_boost)
                                short_seed_u8[seed_top:side_bottom, right_inner:extra_right] = 255
                    short_seed_u8 = cv2.bitwise_or(short_seed_u8, upper_prior_u8)
                    short_seed_u8 = cv2.bitwise_and(short_seed_u8, corridor_u8)
                    short_seed_u8 = cv2.morphologyEx(
                        short_seed_u8,
                        cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (7, 11) if male_short_compact_style else (9, 13),
                        ),
                    )
                    short_seed_u8 = cv2.erode(
                        short_seed_u8,
                        cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (7, 9) if male_short_compact_style else (5, 7),
                        ),
                        iterations=1,
                    )
                    short_seed_u8 = cv2.dilate(
                        short_seed_u8,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)),
                        iterations=1,
                    )

                if int((short_seed_u8 > 0).sum()) >= 120:
                    gen_mask = short_seed_u8.astype(np.float32) / 255.0
                else:
                    fallback_bottom = min(H, int(cutoff_y + face_h * 0.12))
                    fallback_left = max(0, int(x1f - face_w * 0.78))
                    fallback_right = min(W, int(x2f + face_w * 0.78))
                    if seed_top < fallback_bottom and fallback_left < fallback_right:
                        gen_mask[seed_top:fallback_bottom, fallback_left:fallback_right] = 1.0
                short_generation_seed_mask_for_debug = gen_mask.copy()
            else:
                gen_mask[head_y1:head_y2, head_x1:head_x2] = 1.0
            gen_mask = np.clip(gen_mask - protect_mask_for_sd, 0.0, 1.0)
            gen_mask = np.clip(gen_mask - cloth_mask_dilated, 0.0, 1.0)
            if hair_length == "short":
                below_bob_generation_block_for_post = self._build_short_below_bob_generation_block_mask(
                    removal_mask=removal_mask_for_post,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                    support_mask=lower_tail_support_for_post,
                    anchor_mask=subject_cloth_anchor_for_post if use_short_dark_cloth_anchor_fallback else None,
                )
                below_bob_cloth_restore_for_post = self._build_short_below_bob_cloth_restore_mask(
                    removal_mask=removal_mask_for_post,
                    cloth_mask=cloth_restore_mask_for_post,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                    subject_gender_mode=subject_gender_mode,
                    support_mask=lower_tail_support_for_post,
                    anchor_mask=subject_cloth_anchor_for_post if use_short_dark_cloth_anchor_fallback else None,
                )
                if (
                    hair_length == "short"
                    and below_bob_cloth_restore_for_post is not None
                    and below_bob_cloth_restore_for_post.shape == (H, W)
                    and float(center_cloth_restore_exclusion_mask.sum()) > 0.0
                ):
                    below_bob_center_exclusion_weight = (
                        0.34 if subject_gender_mode != "male" else 1.08
                    )
                    below_bob_cloth_restore_for_post = np.clip(
                        below_bob_cloth_restore_for_post.astype(np.float32)
                        * (
                            1.0
                            - np.clip(
                                center_cloth_restore_exclusion_mask * below_bob_center_exclusion_weight,
                                0.0,
                                1.0,
                            )
                        ),
                        0.0,
                        1.0,
                    )
                if (
                    below_bob_generation_block_for_post is not None
                    and below_bob_generation_block_for_post.shape == (H, W)
                ):
                    gen_mask = np.clip(
                        gen_mask - below_bob_generation_block_for_post * 1.45,
                        0.0,
                        1.0,
                    )
                if (
                    below_bob_cloth_restore_for_post is not None
                    and below_bob_cloth_restore_for_post.shape == (H, W)
                ):
                    gen_mask = np.clip(
                        gen_mask - below_bob_cloth_restore_for_post * 1.30,
                        0.0,
                        1.0,
                    )
            if shoulder_protect_for_post is not None and shoulder_protect_for_post.shape == (H, W):
                gen_mask = np.clip(
                    gen_mask - (shoulder_protect_for_post * (0.42 if hair_length == "short" else 0.28)),
                    0.0,
                    1.0,
                )
            if neckline_preserve_for_post is not None and neckline_preserve_for_post.shape == (H, W):
                gen_mask = np.clip(
                    gen_mask - neckline_preserve_for_post * (0.34 if hair_length == "short" else 0.18),
                    0.0,
                    1.0,
                )
            if hair_length == "short":
                gen_mask = cv2.erode(
                    gen_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    iterations=1,
                )
                short_volume_cap_u8 = np.zeros((H, W), dtype=np.uint8)
                cap_center_x = face_cx
                if male_short_dominant_side == "left":
                    cap_center_x = max(0, face_cx - max(6, int(face_w * 0.05)))
                elif male_short_dominant_side == "right":
                    cap_center_x = min(W - 1, face_cx + max(6, int(face_w * 0.05)))
                cap_center_y = int(
                    max(
                        seed_top + 1,
                        min(
                            H - 1,
                            y1f
                            + face_h
                            * (0.20 if male_short_compact_style else (0.14 if male_short_volume_boost else 0.18)),
                        ),
                    )
                )
                cap_axes_x = max(
                    24,
                    int(face_w * (0.52 if male_short_compact_style else (0.64 if male_short_volume_boost else 0.58))),
                )
                cap_axes_y = max(
                    30,
                    int(face_h * (0.60 if male_short_compact_style else (0.84 if male_short_volume_boost else 0.72))),
                )
                cv2.ellipse(
                    short_volume_cap_u8,
                    (cap_center_x, cap_center_y),
                    (cap_axes_x, cap_axes_y),
                    0,
                    0,
                    360,
                    255,
                    thickness=-1,
                )
                cap_side_top = max(
                    seed_top,
                    int(
                        y1f
                        + face_h
                        * (0.14 if male_short_compact_style else (0.08 if male_short_volume_boost else 0.12))
                    ),
                )
                cap_side_bottom = min(
                    H,
                    int(y2f + face_h * (0.14 if male_short_compact_style else (0.22 if male_short_volume_boost else 0.18))),
                )
                cap_inner_gap = max(10, int(face_w * (0.21 if male_short_compact_style else 0.18)))
                cap_outer_span = max(
                    18,
                    int(face_w * (0.44 if male_short_compact_style else (0.58 if male_short_volume_boost else 0.52))),
                )
                cap_left_outer = max(0, int(face_cx - cap_outer_span))
                cap_left_inner = max(cap_left_outer + 1, int(face_cx - cap_inner_gap))
                cap_right_inner = min(W - 1, int(face_cx + cap_inner_gap))
                cap_right_outer = min(W, int(face_cx + cap_outer_span))
                if male_short_dominant_side == "left":
                    cap_left_outer = max(0, cap_left_outer - max(10, int(face_w * 0.12)))
                elif male_short_dominant_side == "right":
                    cap_right_outer = min(W, cap_right_outer + max(10, int(face_w * 0.12)))
                if cap_side_top < cap_side_bottom:
                    short_volume_cap_u8[cap_side_top:cap_side_bottom, cap_left_outer:cap_left_inner] = 255
                    short_volume_cap_u8[cap_side_top:cap_side_bottom, cap_right_inner:cap_right_outer] = 255
                short_volume_cap = cv2.GaussianBlur(
                    short_volume_cap_u8.astype(np.float32) / 255.0,
                    (0, 0),
                    sigmaX=3.0,
                    sigmaY=3.4,
                ).astype(np.float32)
                gen_mask = np.clip(
                    gen_mask
                    * np.clip(
                        short_volume_cap
                        * (
                            1.14
                            if male_short_compact_style
                            else (1.34 if male_short_volume_boost else 1.24)
                        ),
                        0.0,
                        1.0,
                    ),
                    0.0,
                    1.0,
                )
            composite_bangs_release_mask = np.zeros((H, W), dtype=np.float32)
            if float(bangs_restore_for_sd.sum()) > 0.0:
                soft_bangs_generation_mask = self._build_soft_bangs_generation_mask(
                    bangs_restore_for_sd,
                    face_bbox=face_bbox,
                    hair_length=hair_length,
                )
                composite_bangs_release_mask = np.clip(
                    soft_bangs_generation_mask.astype(np.float32) * (1.28 if has_color_request else 1.15),
                    0.0,
                    1.0,
                )
                gen_mask = np.maximum(
                    gen_mask,
                    np.clip(
                        soft_bangs_generation_mask.astype(np.float32) * (1.12 if has_color_request else 1.0),
                        0.0,
                        1.0,
                    ),
                )
                _store_mask("pipeline_bangs_generation_soft_mask", soft_bangs_generation_mask)
                _store_mask("pipeline_bangs_composite_release_mask", composite_bangs_release_mask)

            _store_mask("pipeline_short_removal_mask", removal_mask_for_post)
            _store_mask("pipeline_short_generation_seed_mask", short_generation_seed_mask_for_debug)
            _store_mask("pipeline_short_generation_mask", gen_mask)
            _store_mask("pipeline_short_below_bob_generation_block_mask", below_bob_generation_block_for_post)
            _store_mask("pipeline_short_below_bob_cloth_restore_mask", below_bob_cloth_restore_for_post)

            logger.info(
                f"[SDPipeline] 전략2: removal_px={removal_mask_for_post.sum():.0f}, "
                f"gen_px={gen_mask.sum():.0f}, cutoff_y={cutoff_y}, "
                f"head_box=({head_x1},{head_y1})-({head_x2},{head_y2})"
            )

            bg_mode_requested = self.config.bg_fill_mode
            bg_mode = self._resolve_background_fill_mode(
                bg_mode_requested,
                removal_mask=removal_mask,
                cloth_mask=cloth_mask_dilated,
                face_bbox=face_bbox,
                hair_length=hair_length,
                lower_tail_support_mask=lower_tail_support_for_post,
                center_support_mask=center_chest_strand_removal_mask,
            )
            logger.info(
                f"[SDPipeline] bg_fill_mode requested={bg_mode_requested} resolved={bg_mode}"
            )
            tail_core_mask = np.zeros((H, W), dtype=np.float32)

            if removal_mask.sum() > 50:
                removal_u8 = (removal_mask > 0.5).astype(np.uint8) * 255
                removal_u8_dilated = cv2.dilate(
                    removal_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                    iterations=1,
                )

                inpaint_radius = 8 if hair_length == "short" else 10
                img_rgb_ns = cv2.inpaint(
                    img_rgb,
                    removal_u8_dilated,
                    inpaintRadius=inpaint_radius,
                    flags=cv2.INPAINT_NS,
                )
                img_rgb_telea = cv2.inpaint(
                    img_rgb,
                    removal_u8_dilated,
                    inpaintRadius=max(3, inpaint_radius),
                    flags=cv2.INPAINT_TELEA,
                )
                img_rgb_filled = cv2.addWeighted(img_rgb_ns, 0.25, img_rgb_telea, 0.75, 0.0)

                removal_px = int((removal_u8 > 0).sum())
                max_clone_px = int(H * W * 0.18)
                if 50 <= removal_px <= max_clone_px:
                    ys, xs = np.where(removal_u8 > 0)
                    c_x = int((xs.min() + xs.max()) * 0.5)
                    c_y = int((ys.min() + ys.max()) * 0.5)
                    src_bgr = cv2.cvtColor(img_rgb_filled, cv2.COLOR_RGB2BGR)
                    dst_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    try:
                        cloned_bgr = cv2.seamlessClone(
                            src_bgr,
                            dst_bgr,
                            removal_u8,
                            (c_x, c_y),
                            cv2.NORMAL_CLONE,
                        )
                        img_rgb_cleaned = cv2.cvtColor(cloned_bgr, cv2.COLOR_BGR2RGB)
                    except Exception:
                        soft_alpha = removal_u8.astype(np.float32) / 255.0
                        soft_alpha = cv2.GaussianBlur(soft_alpha, (0, 0), sigmaX=1.2)[..., np.newaxis]
                        soft_alpha = np.clip((soft_alpha - 0.28) / 0.72, 0.0, 1.0)
                        img_rgb_cleaned = (
                            img_rgb_filled.astype(np.float32) * soft_alpha
                            + img_rgb.astype(np.float32) * (1.0 - soft_alpha)
                        ).clip(0, 255).astype(np.uint8)
                else:
                    soft_alpha = removal_u8.astype(np.float32) / 255.0
                    soft_alpha = cv2.GaussianBlur(soft_alpha, (0, 0), sigmaX=1.2)[..., np.newaxis]
                    soft_alpha = np.clip((soft_alpha - 0.28) / 0.72, 0.0, 1.0)
                    img_rgb_cleaned = (
                        img_rgb_filled.astype(np.float32) * soft_alpha
                        + img_rgb.astype(np.float32) * (1.0 - soft_alpha)
                    ).clip(0, 255).astype(np.uint8)

                cloth_u8 = (cloth_mask_dilated > 0.45).astype(np.uint8) * 255
                cloth_overlap = cv2.bitwise_and(removal_u8, cloth_u8)
                protect_u8 = ((protect_mask_for_sd > 0.2).astype(np.uint8) * 255)
                cloth_overlap = cv2.bitwise_and(cloth_overlap, cv2.bitwise_not(protect_u8))
                if int((cloth_overlap > 0).sum()) >= 80:
                    cloth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    cloth_overlap = cv2.dilate(cloth_overlap, cloth_k, iterations=1)
                    cloth_ns = cv2.inpaint(img_rgb, cloth_overlap, inpaintRadius=4, flags=cv2.INPAINT_NS)
                    cloth_te = cv2.inpaint(img_rgb, cloth_overlap, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                    cloth_refill = cv2.addWeighted(cloth_ns, 0.40, cloth_te, 0.60, 0.0)
                    cloth_alpha = cloth_overlap.astype(np.float32) / 255.0
                    cloth_alpha = cv2.GaussianBlur(
                        cloth_alpha,
                        (0, 0),
                        sigmaX=2.2,
                        sigmaY=2.2,
                    )[..., np.newaxis]
                    cloth_alpha = np.clip(cloth_alpha * 0.92, 0.0, 1.0)
                    img_rgb_cleaned = (
                        cloth_refill.astype(np.float32) * cloth_alpha
                        + img_rgb_cleaned.astype(np.float32) * (1.0 - cloth_alpha)
                    ).clip(0, 255).astype(np.uint8)
                    logger.info(
                        f"[SDPipeline] cloth overlap 복원 적용: pixels={int((cloth_overlap > 0).sum())}"
                    )

                if hair_length == "short":
                    try:
                        tail_core_mask = self._build_short_tail_core_mask(
                            removal_mask=removal_mask,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y,
                            hair_length=hair_length,
                        )
                        if shoulder_protect_for_post is not None and shoulder_protect_for_post.shape == (H, W):
                            tail_core_mask = np.clip(
                                tail_core_mask - shoulder_protect_for_post * 0.16,
                                0.0,
                                1.0,
                            )
                        if neckline_preserve_for_post is not None and neckline_preserve_for_post.shape == (H, W):
                            tail_core_mask = np.clip(
                                tail_core_mask - neckline_preserve_for_post * 0.18,
                                0.0,
                                1.0,
                            )
                        tail_core_u8 = cv2.dilate(
                            (tail_core_mask > 0.10).astype(np.uint8) * 255,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
                            iterations=1,
                        )
                        if int((tail_core_u8 > 0).sum()) >= 60:
                            img_rgb_cleaned = self._lama_inpaint(img_rgb_cleaned, tail_core_u8)
                            logger.info(
                                f"[SDPipeline] short tail-core LaMa preclean applied: pixels={int((tail_core_u8 > 0).sum())}"
                            )
                        _store_mask("pipeline_short_tail_core_mask", tail_core_mask)

                        artifact_cleanup_mask = self._build_dark_tail_residual_mask(
                            img_rgb=img_rgb_cleaned,
                            removal_mask=removal_mask,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y,
                            hair_length=hair_length,
                        )
                        front_strand_cleanup_mask = self._build_front_strand_cleanup_mask(
                            removal_mask=removal_mask,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y,
                            hair_length=hair_length,
                            anchor_mask=center_chest_strand_removal_mask,
                        )
                        front_strand_cleanup_u8 = (
                            (np.clip(front_strand_cleanup_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                        )
                        artifact_cleanup_mask = np.maximum(
                            artifact_cleanup_mask,
                            np.clip(tail_core_mask * 0.55, 0.0, 1.0),
                        )
                        artifact_cleanup_mask = np.maximum(
                            artifact_cleanup_mask,
                            np.clip(front_strand_cleanup_mask * 0.46, 0.0, 1.0),
                        )
                        if float(center_chest_strand_removal_mask.sum()) > 0.0:
                            artifact_cleanup_mask = np.maximum(
                                artifact_cleanup_mask,
                                np.clip(center_chest_strand_removal_mask * 0.62, 0.0, 1.0),
                            )
                        if shoulder_protect_for_post is not None and shoulder_protect_for_post.shape == (H, W):
                            artifact_cleanup_mask = np.clip(
                                artifact_cleanup_mask - shoulder_protect_for_post * 0.08,
                                0.0,
                                1.0,
                            )
                        if neckline_preserve_for_post is not None and neckline_preserve_for_post.shape == (H, W):
                            artifact_cleanup_mask = np.clip(
                                artifact_cleanup_mask - neckline_preserve_for_post * 0.10,
                                0.0,
                                1.0,
                            )
                        if torso_cloth_preserve_for_post is not None and torso_cloth_preserve_for_post.shape == (H, W):
                            artifact_cleanup_mask = np.clip(
                                artifact_cleanup_mask - torso_cloth_preserve_for_post * 0.12,
                                0.0,
                                1.0,
                            )
                        artifact_cleanup_mask = np.clip(
                            artifact_cleanup_mask - cloth_mask_dilated * 0.04,
                            0.0,
                            1.0,
                        )
                        artifact_anchor_u8 = np.zeros((H, W), dtype=np.uint8)
                        artifact_anchor_u8 = cv2.bitwise_or(
                            artifact_anchor_u8,
                            (np.clip(tail_core_mask.astype(np.float32), 0.0, 1.0) > 0.10).astype(np.uint8) * 255,
                        )
                        artifact_anchor_u8 = cv2.bitwise_or(
                            artifact_anchor_u8,
                            (np.clip(front_strand_cleanup_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
                        )
                        if float(center_chest_strand_removal_mask.sum()) > 0.0:
                            artifact_anchor_u8 = cv2.bitwise_or(
                                artifact_anchor_u8,
                                (np.clip(center_chest_strand_removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
                            )
                        artifact_cleanup_mask_for_post = np.clip(
                            artifact_cleanup_mask,
                            0.0,
                            1.0,
                        ).astype(np.float32)
                        artifact_cleanup_u8 = (artifact_cleanup_mask_for_post > 0.08).astype(np.uint8) * 255
                        artifact_cleanup_u8 = cv2.bitwise_and(
                            artifact_cleanup_u8,
                            (np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                        )
                        artifact_cleanup_u8 = self._filter_short_center_cleanup_mask(
                            artifact_cleanup_u8,
                            face_bbox,
                            cutoff_y,
                            anchor_u8=artifact_anchor_u8,
                            top_scale=0.00,
                            bottom_scale=0.78,
                            half_w_scale=0.25,
                            shrink_half_w_scale=0.18,
                            max_area_scale=0.10,
                            max_width_scale=0.34,
                            min_height_scale=0.10,
                            center_allow_scale=0.18,
                            max_total_scale=0.05,
                        )
                        artifact_cleanup_mask_for_post = np.where(
                            artifact_cleanup_u8 > 0,
                            artifact_cleanup_mask_for_post,
                            0.0,
                        ).astype(np.float32)
                        artifact_cleanup_mask = artifact_cleanup_mask_for_post.copy()
                        artifact_cleanup_u8 = cv2.dilate(
                            artifact_cleanup_u8,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
                            iterations=1,
                        )
                        artifact_cleanup_u8 = self._filter_short_center_cleanup_mask(
                            artifact_cleanup_u8,
                            face_bbox,
                            cutoff_y,
                            anchor_u8=artifact_anchor_u8,
                            top_scale=0.00,
                            bottom_scale=0.80,
                            half_w_scale=0.26,
                            shrink_half_w_scale=0.19,
                            max_area_scale=0.11,
                            max_width_scale=0.36,
                            min_height_scale=0.10,
                            center_allow_scale=0.19,
                            max_total_scale=0.06,
                        )
                        if int((artifact_cleanup_u8 > 0).sum()) >= 80:
                            img_rgb_cleaned = self._lama_inpaint(img_rgb_cleaned, artifact_cleanup_u8)
                            img_rgb_cleaned = self._cv2_cleanup_dark_tail_blob(img_rgb_cleaned, artifact_cleanup_u8)
                            if int((front_strand_cleanup_u8 > 0).sum()) >= 20:
                                img_rgb_cleaned = self._cv2_cleanup_dark_tail_blob(img_rgb_cleaned, front_strand_cleanup_u8)
                            logger.info(
                                f"[SDPipeline] short artifact preclean applied: pixels={int((artifact_cleanup_u8 > 0).sum())}"
                            )
                        _store_mask("pipeline_front_strand_cleanup_mask", front_strand_cleanup_mask)
                        _store_mask("pipeline_short_artifact_cleanup_mask", artifact_cleanup_mask)
                    except Exception as e:
                        logger.warning(f"[SDPipeline] short tail-core LaMa preclean 실패(무시): {e}")

                logger.info("[SDPipeline] cv2.inpaint 2-way 블렌딩 완료")
                if bg_mode == "sd":
                    try:
                        fill_seed = int(seeds[0]) if seeds else random.randint(0, 2**31 - 1)
                        face_crop_fill = self._crop_face(Image.fromarray(img_rgb), face_bbox)
                        fill_refine_mode = "short_tail" if hair_length == "short" else "generic"
                        img_rgb_cleaned = self._sd_refine_removed_region(
                            base_rgb=img_rgb_cleaned,
                            removal_mask=removal_mask,
                            face_bbox=face_bbox,
                            face_crop_pil=face_crop_fill,
                            protect_mask=protect_mask_for_sd,
                            cloth_mask=cloth_mask_dilated,
                            hair_length=hair_length,
                            seed=fill_seed,
                            refine_mode=fill_refine_mode,
                        )
                        logger.info("[SDPipeline] bg_fill_mode=sd: 제거 영역 SD 보정 완료")
                    except Exception as e:
                        logger.warning(f"[SDPipeline] bg_fill_mode=sd 실패, cv2 결과 사용: {e}")

                try:
                    img_rgb_cleaned = self._remove_residual_hair_below_cutoff(
                        img_rgb=img_rgb_cleaned,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        removal_mask=removal_mask,
                        shoulder_protect=shoulder_protect_for_post,
                        neckline_preserve=neckline_preserve_for_post,
                        lateral_preserve=lateral_neck_preserve_for_post,
                        hair_length=hair_length,
                        center_anchor_mask=center_chest_strand_removal_mask,
                    )
                except Exception as e:
                    logger.warning(f"[SDPipeline] residual hair 정리 실패(무시): {e}")
            else:
                img_rgb_cleaned = img_rgb

            if hair_length == "short":
                try:
                    preclean_cloth_hair_cleanup_mask = self._build_preclean_cloth_hair_cleanup_mask(
                        img_rgb=img_rgb_cleaned,
                        removal_mask=removal_mask_for_post,
                        cloth_mask=cloth_mask_dilated,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        hair_length=hair_length,
                    )
                    preclean_cloth_hair_cleanup_u8 = (
                        (np.clip(preclean_cloth_hair_cleanup_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                    )
                    if int((preclean_cloth_hair_cleanup_u8 > 0).sum()) >= 120:
                        img_rgb_cleaned = self._lama_inpaint(
                            img_rgb_cleaned,
                            preclean_cloth_hair_cleanup_u8,
                        )
                        img_rgb_cleaned = self._cv2_cleanup_dark_tail_blob(
                            img_rgb_cleaned,
                            preclean_cloth_hair_cleanup_u8,
                        )
                        if cloth_mask_dilated is not None and cloth_mask_dilated.shape == (H, W):
                            img_rgb_cleaned = self._blend_neighbor_cloth_tone(
                                img_rgb_cleaned,
                                preclean_cloth_hair_cleanup_mask,
                                cloth_mask=cloth_mask_dilated,
                            )
                        if hair_length != "short":
                            img_rgb_cleaned = self._restore_cloth_overlap_from_source(
                                source_rgb=img_rgb,
                                current_rgb=img_rgb_cleaned,
                                restore_mask=preclean_cloth_hair_cleanup_mask,
                                final_hair_mask=None,
                            )
                        img_rgb_cleaned = self._cv2_refine_cloth_region(
                            img_rgb_cleaned,
                            preclean_cloth_hair_cleanup_mask,
                        )
                    _store_mask("pipeline_preclean_cloth_hair_cleanup_mask", preclean_cloth_hair_cleanup_mask)
                except Exception as e:
                    logger.warning(f"[SDPipeline] preclean cloth hair cleanup failed (ignored): {e}")

                try:
                    preclean_side_candidate_mask = np.clip(
                        removal_mask_for_post.astype(np.float32),
                        0.0,
                        1.0,
                    )
                    if (
                        artifact_cleanup_mask_for_post is not None
                        and artifact_cleanup_mask_for_post.shape == (H, W)
                    ):
                        preclean_side_candidate_mask = np.maximum(
                            preclean_side_candidate_mask,
                            np.clip(artifact_cleanup_mask_for_post.astype(np.float32), 0.0, 1.0),
                        ).astype(np.float32)
                    if (
                        shoulder_cloth_release_for_post is not None
                        and shoulder_cloth_release_for_post.shape == (H, W)
                    ):
                        preclean_side_candidate_mask = np.maximum(
                            preclean_side_candidate_mask,
                            np.clip(shoulder_cloth_release_for_post.astype(np.float32), 0.0, 1.0) * 0.88,
                        ).astype(np.float32)
                    if (
                        below_bob_cloth_restore_for_post is not None
                        and below_bob_cloth_restore_for_post.shape == (H, W)
                    ):
                        preclean_side_candidate_mask = np.maximum(
                            preclean_side_candidate_mask,
                            np.clip(below_bob_cloth_restore_for_post.astype(np.float32), 0.0, 1.0),
                        ).astype(np.float32)
                    preclean_side_restore_mask = self._build_side_column_cloth_restore_mask(
                        img_rgb=img_rgb_cleaned,
                        cloth_mask=cloth_restore_mask_for_post,
                        candidate_mask=preclean_side_candidate_mask,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        hair_length=hair_length,
                        final_hair_mask=None,
                        subject_gender_mode=subject_gender_mode,
                    )
                    direct_preclean_side_restore_mask = self._build_direct_short_column_restore_mask(
                        removal_mask=removal_mask_for_post,
                        cloth_mask=cloth_restore_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        hair_length=hair_length,
                        subject_gender_mode=subject_gender_mode,
                    )
                    if hair_length == "short" and float(center_cloth_restore_exclusion_mask.sum()) > 0.0:
                        preclean_side_exclusion_weight = 0.28 if subject_gender_mode != "male" else 0.96
                        direct_preclean_exclusion_weight = 0.14 if subject_gender_mode != "male" else 1.12
                        preclean_side_restore_mask = np.clip(
                            preclean_side_restore_mask.astype(np.float32)
                            * (
                                1.0
                                - np.clip(
                                    center_cloth_restore_exclusion_mask * preclean_side_exclusion_weight,
                                    0.0,
                                    1.0,
                                )
                            ),
                            0.0,
                            1.0,
                        )
                        direct_preclean_side_restore_mask = np.clip(
                            direct_preclean_side_restore_mask.astype(np.float32)
                            * (
                                1.0
                                - np.clip(
                                    center_cloth_restore_exclusion_mask * direct_preclean_exclusion_weight,
                                    0.0,
                                    1.0,
                                )
                            ),
                            0.0,
                            1.0,
                        )
                    if float(direct_preclean_side_restore_mask.sum()) > 0.0:
                        preclean_side_restore_mask = np.maximum(
                            preclean_side_restore_mask,
                            direct_preclean_side_restore_mask,
                        ).astype(np.float32)
                    if (
                        below_bob_cloth_restore_for_post is not None
                        and below_bob_cloth_restore_for_post.shape == (H, W)
                    ):
                        preclean_side_restore_mask = np.maximum(
                            preclean_side_restore_mask,
                            np.clip(below_bob_cloth_restore_for_post.astype(np.float32), 0.0, 1.0),
                        ).astype(np.float32)
                    if hair_length == "short" and float(center_cloth_restore_exclusion_mask.sum()) > 0.0:
                        merged_preclean_exclusion_weight = 0.36 if subject_gender_mode != "male" else 1.16
                        preclean_side_restore_mask = np.clip(
                            preclean_side_restore_mask.astype(np.float32)
                            * (
                                1.0
                                - np.clip(
                                    center_cloth_restore_exclusion_mask * merged_preclean_exclusion_weight,
                                    0.0,
                                    1.0,
                                )
                            ),
                            0.0,
                            1.0,
                        )
                    preclean_side_restore_u8 = (
                        (np.clip(preclean_side_restore_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                    )
                    preclean_side_cleanup_mask = self._build_preclean_side_column_cleanup_mask(
                        img_rgb=img_rgb_cleaned,
                        cloth_mask=cloth_mask_dilated,
                        base_mask=preclean_side_restore_mask,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        hair_length=hair_length,
                    )
                    preclean_side_cleanup_u8 = (
                        (np.clip(preclean_side_cleanup_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                    )
                    if int((preclean_side_cleanup_u8 > 0).sum()) >= 48:
                        img_rgb_cleaned = self._cleanup_region_with_cloth_restore(
                            source_rgb=img_rgb,
                            current_rgb=img_rgb_cleaned,
                            cleanup_mask=preclean_side_cleanup_mask,
                            cloth_mask=cloth_mask_dilated,
                            ignore_final_hair_for_cloth_restore=True,
                            cleanup_dark_tail=True,
                        )
                    if int((preclean_side_restore_u8 > 0).sum()) >= 140:
                        if hair_length == "short":
                            img_rgb_cleaned = self._cleanup_region_with_cloth_restore(
                                source_rgb=img_rgb,
                                current_rgb=img_rgb_cleaned,
                                cleanup_mask=preclean_side_restore_mask,
                                cloth_mask=cloth_mask_dilated,
                                ignore_final_hair_for_cloth_restore=True,
                                cleanup_dark_tail=True,
                            )
                        else:
                            img_rgb_cleaned = self._restore_cloth_overlap_from_source(
                                source_rgb=img_rgb,
                                current_rgb=img_rgb_cleaned,
                                restore_mask=preclean_side_restore_mask,
                                final_hair_mask=None,
                            )
                            img_rgb_cleaned = self._cv2_refine_cloth_region(
                                img_rgb_cleaned,
                                preclean_side_restore_mask,
                            )
                    _store_mask("pipeline_preclean_side_column_cleanup_mask", preclean_side_cleanup_mask)
                    _store_mask("pipeline_preclean_side_column_cloth_restore_mask", preclean_side_restore_mask)
                    _store_mask("pipeline_preclean_direct_short_column_restore_mask", direct_preclean_side_restore_mask)
                except Exception as e:
                    logger.warning(f"[SDPipeline] preclean side column cloth restore failed (ignored): {e}")
                try:
                    preclean_dark_lane_mask = self._build_dark_lane_cleanup_mask(
                        img_rgb=img_rgb_cleaned,
                        cloth_mask=cloth_restore_mask_for_post,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        hair_length=hair_length,
                        anchor_mask=subject_cloth_anchor_for_post if use_short_dark_cloth_anchor_fallback else None,
                    )
                    preclean_dark_lane_u8 = (
                        (np.clip(preclean_dark_lane_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                    )
                    if int((preclean_dark_lane_u8 > 0).sum()) >= 40:
                        img_rgb_cleaned = self._lama_inpaint(img_rgb_cleaned, preclean_dark_lane_u8)
                        img_rgb_cleaned = self._cv2_refine_cloth_region(
                            img_rgb_cleaned,
                            preclean_dark_lane_mask,
                        )
                    _store_mask("pipeline_preclean_dark_lane_cleanup_mask", preclean_dark_lane_mask)
                except Exception as e:
                    logger.warning(f"[SDPipeline] preclean dark lane cleanup failed (ignored): {e}")
                regen_tail_mask = self._build_short_regen_tail_mask(
                    img_rgb=img_rgb_cleaned,
                    removal_mask=removal_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                )
                face_w = max(int(face_bbox[2] - face_bbox[0]), 1)
                face_h = max(int(face_bbox[3] - face_bbox[1]), 1)
                if shoulder_protect_for_post is not None and shoulder_protect_for_post.shape == (H, W):
                    regen_tail_mask = np.clip(
                        regen_tail_mask - shoulder_protect_for_post * 0.32,
                        0.0,
                        1.0,
                    )
                if neckline_preserve_for_post is not None and neckline_preserve_for_post.shape == (H, W):
                    regen_tail_mask = np.clip(
                        regen_tail_mask - neckline_preserve_for_post * 0.38,
                        0.0,
                        1.0,
                    )
                regen_tail_mask = np.clip(regen_tail_mask - cloth_mask_dilated * 0.48, 0.0, 1.0)
                regen_tail_u8 = cv2.dilate(
                    (regen_tail_mask > 0.08).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
                    iterations=1,
                )
                if hair_length == "short":
                    regen_cap_y = min(H, int(cutoff_y + face_h * 0.88))
                    regen_center_x = int(0.5 * (face_bbox[0] + face_bbox[2]))
                    regen_x1 = max(0, int(face_bbox[0] - face_w * 1.16))
                    regen_x2 = min(W, int(face_bbox[2] + face_w * 1.16))
                    regen_corridor_u8 = np.zeros((H, W), dtype=np.uint8)
                    regen_top = max(0, int(cutoff_y - face_h * 0.02))
                    if regen_top < regen_cap_y and regen_x1 < regen_x2:
                        regen_corridor_u8[regen_top:regen_cap_y, regen_x1:regen_x2] = 255
                        regen_tail_u8 = cv2.bitwise_and(regen_tail_u8, regen_corridor_u8)
                    if regen_cap_y < H:
                        regen_tail_u8[regen_cap_y:, :] = 0
                cloth_block_u8 = cv2.dilate(
                    (cloth_mask_dilated > 0.10).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                    iterations=1,
                )
                regen_tail_u8 = cv2.bitwise_and(regen_tail_u8, cv2.bitwise_not(cloth_block_u8))
                if shoulder_protect_for_post is not None and shoulder_protect_for_post.shape == (H, W):
                    shoulder_block_u8 = cv2.dilate(
                        (shoulder_protect_for_post > 0.04).astype(np.uint8) * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 19)),
                        iterations=1,
                    )
                    regen_tail_u8 = cv2.bitwise_and(regen_tail_u8, cv2.bitwise_not(shoulder_block_u8))
                if neckline_preserve_for_post is not None and neckline_preserve_for_post.shape == (H, W):
                    neckline_block_u8 = cv2.dilate(
                        (neckline_preserve_for_post > 0.04).astype(np.uint8) * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 15)),
                        iterations=1,
                    )
                    regen_tail_u8 = cv2.bitwise_and(regen_tail_u8, cv2.bitwise_not(neckline_block_u8))
                if hair_length == "short" and int((regen_tail_u8 > 0).sum()) > 0:
                    filtered_regen_u8 = np.zeros((H, W), dtype=np.uint8)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(regen_tail_u8, 8)
                    center_half = max(18, int(face_w * 0.28))
                    side_offset = max(14, int(face_w * 0.14))
                    side_max_width = max(56, int(face_w * 0.88))
                    center_max_width = max(28, int(face_w * 0.42))
                    min_component_height = max(14, int(face_h * 0.10))
                    side_max_area = max(520, int(face_w * face_h * 0.22))
                    center_max_area = max(220, int(face_w * face_h * 0.09))
                    for idx in range(1, num_labels):
                        x = int(stats[idx, cv2.CC_STAT_LEFT])
                        y = int(stats[idx, cv2.CC_STAT_TOP])
                        w = int(stats[idx, cv2.CC_STAT_WIDTH])
                        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
                        area = int(stats[idx, cv2.CC_STAT_AREA])
                        bottom = y + h
                        if area < 18 or h < min_component_height or bottom > regen_cap_y:
                            continue
                        comp_cx = float(centroids[idx][0])
                        if abs(comp_cx - regen_center_x) <= center_half:
                            if w > center_max_width or area > center_max_area:
                                continue
                        elif abs(comp_cx - regen_center_x) >= side_offset:
                            if w > side_max_width or area > side_max_area:
                                continue
                        else:
                            continue
                        filtered_regen_u8[labels == idx] = 255
                    regen_tail_u8 = filtered_regen_u8
                if (
                    hair_length == "short"
                    and below_bob_generation_block_for_post is not None
                    and below_bob_generation_block_for_post.shape == (H, W)
                    and int((regen_tail_u8 > 0).sum()) > 0
                ):
                    below_bob_generation_block_u8 = (
                        (np.clip(below_bob_generation_block_for_post.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8)
                        * 255
                    )
                    regen_tail_u8 = cv2.bitwise_and(
                        regen_tail_u8,
                        cv2.bitwise_not(below_bob_generation_block_u8),
                    )
                if (
                    hair_length == "short"
                    and below_bob_cloth_restore_for_post is not None
                    and below_bob_cloth_restore_for_post.shape == (H, W)
                    and int((regen_tail_u8 > 0).sum()) > 0
                ):
                    below_bob_block_u8 = (
                        (np.clip(below_bob_cloth_restore_for_post.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8)
                        * 255
                    )
                    regen_tail_u8 = cv2.bitwise_and(
                        regen_tail_u8,
                        cv2.bitwise_not(below_bob_block_u8),
                    )
                regen_tail_mask = regen_tail_u8.astype(np.float32) / 255.0
                if int((regen_tail_u8 > 0).sum()) > 60:
                    gen_mask = np.maximum(gen_mask, regen_tail_mask)
                if (
                    hair_length == "short"
                    and below_bob_generation_block_for_post is not None
                    and below_bob_generation_block_for_post.shape == (H, W)
                ):
                    gen_mask = np.clip(
                        gen_mask - below_bob_generation_block_for_post * 1.45,
                        0.0,
                        1.0,
                    )
                if (
                    hair_length == "short"
                    and below_bob_cloth_restore_for_post is not None
                    and below_bob_cloth_restore_for_post.shape == (H, W)
                ):
                    gen_mask = np.clip(
                        gen_mask - below_bob_cloth_restore_for_post * 1.35,
                        0.0,
                        1.0,
                    )
                _store_mask("pipeline_short_regen_tail_mask", regen_tail_mask)

            _store_rgb("cv2_background_cleaned_rgb", img_rgb_cleaned)
            _store_mask("pipeline_short_generation_mask", gen_mask)

            hair_mask_for_sd = gen_mask.astype(np.float32)
            img_rgb_for_sd   = img_rgb_cleaned
        else:
            # long 헤어는 기존 단일 패스 유지
            hair_mask_for_sd = hair_mask
            img_rgb_for_sd   = img_rgb
            img_rgb_cleaned  = img_rgb
            if float(bangs_restore_for_sd.sum()) > 0.0:
                long_soft_bangs_mask = self._build_soft_bangs_generation_mask(
                    bangs_restore_for_sd,
                    face_bbox=face_bbox,
                    hair_length=hair_length,
                )
                if float(long_soft_bangs_mask.sum()) > 0.0:
                    long_soft_bangs_u8 = cv2.dilate(
                        (np.clip(long_soft_bangs_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
                        iterations=1,
                    )
                    long_soft_bangs_mask = cv2.GaussianBlur(
                        long_soft_bangs_u8.astype(np.float32) / 255.0,
                        (0, 0),
                        sigmaX=1.9,
                        sigmaY=2.2,
                    ).astype(np.float32)
                    long_soft_bangs_mask = np.clip(
                        long_soft_bangs_mask * (1.02 if has_color_request else 0.98),
                        0.0,
                        1.0,
                    )
                    hair_mask_for_sd = np.maximum(
                        hair_mask_for_sd.astype(np.float32),
                        long_soft_bangs_mask,
                    ).astype(np.float32)
                    composite_bangs_release_mask = np.maximum(
                        composite_bangs_release_mask,
                        np.clip(
                            long_soft_bangs_mask * (1.12 if has_color_request else 1.04),
                            0.0,
                            1.0,
                        ),
                    ).astype(np.float32)
                    _store_mask("pipeline_bangs_generation_soft_mask", long_soft_bangs_mask)
                    _store_mask("pipeline_bangs_composite_release_mask", composite_bangs_release_mask)

        _store_mask("sd_inpaint_mask", hair_mask_for_sd)
        _store_rgb("sd_input_rgb", img_rgb_for_sd)

        # ── Step 4: SD 입력 준비 ─────────────────────────────────────────────
        img_pil = Image.fromarray(img_rgb_cleaned)
        # short/medium에서는 기존 long-hair 윤곽도 억제해 ControlNet이
        # 원본 긴머리 edge를 새 단발 형상으로 따라가지 않게 한다.
        canny_suppress = None
        if hair_length in ("short", "medium"):
            canny_suppress = hair_mask_for_removal.astype(np.float32)
            if hair_length == "short":
                canny_suppress = np.maximum(
                    canny_suppress,
                    self._dilate_mask_with_px(hair_mask_for_removal.astype(np.float32), 33),
                )
                canny_suppress = np.maximum(
                    canny_suppress,
                    self._dilate_mask_with_px(hair_mask_for_sd.astype(np.float32), 25),
                )
                if (
                    lower_tail_support_mask is not None
                    and lower_tail_support_mask.shape == canny_suppress.shape
                ):
                    canny_suppress = np.maximum(
                        canny_suppress,
                        self._dilate_mask_with_px(lower_tail_support_mask.astype(np.float32), 21),
                    )
        img_512, mask_512, canny_512, scale, pad = self._prepare_sd_inputs(
            img_rgb_for_sd, hair_mask_for_sd,
            canny_suppress_mask=canny_suppress,
        )
        if debug_images_common is not None:
            debug_images_common["sd_input_512"] = cv2.cvtColor(
                np.array(img_512), cv2.COLOR_RGB2BGR
            )
            debug_images_common["sd_inpaint_mask_512"] = cv2.cvtColor(
                np.array(mask_512).astype(np.uint8), cv2.COLOR_GRAY2BGR
            )
            debug_images_common["controlnet_canny_512"] = cv2.cvtColor(
                np.array(canny_512), cv2.COLOR_RGB2BGR
            )

        # ── Step 5: 얼굴 crop (IP-Adapter) ───────────────────────────────────
        face_crop_pil = self._crop_face(img_pil, face_bbox)

        # ── Step 6: 프롬프트 ─────────────────────────────────────────────────
        prompt, neg_prompt, guidance = self._build_prompt(
            effective_hairstyle_text,
            normalized_color_text,
            hair_length,
            subject_gender=subject_gender_mode,
            sd_prompt_data=sd_prompt_data,
        )
        logger.info(f"[SDPipeline] 프롬프트: {prompt}")
        logger.info(f"[SDPipeline] 네거티브: {neg_prompt}")
        logger.info(f"[SDPipeline] guidance_scale: {guidance}")

        # ── Step 7: SD Inpainting ─────────────────────────────────────────────
        gen_images = self._generate(
            img_512, mask_512, canny_512, face_crop_pil, prompt, neg_prompt, guidance, seeds,
            hair_length=hair_length,
        )

        # ── Step 8: Composite → 원본 해상도 ───────────────────────────────────
        # 전략 2는 원본 위에 short 생성물을 합성한 뒤, cutoff 아래 잔여 긴머리만 정리한다.
        composite_base_rgb = img_rgb_cleaned
        composite_base_bgr = cv2.cvtColor(composite_base_rgb, cv2.COLOR_RGB2BGR)
        male_medium_source_profile: Optional[Dict[str, float]] = None
        male_short_source_profile: Optional[Dict[str, float]] = None
        if hair_length == "medium" and subject_gender_mode == "male":
            male_medium_source_profile = self._estimate_hair_shape_profile(
                hair_mask_base,
                face_bbox,
                hair_length="medium",
            )
        if hair_length == "short" and subject_gender_mode == "male":
            male_short_source_profile = self._estimate_hair_shape_profile(
                hair_mask_base,
                face_bbox,
                hair_length="short",
            )

        candidates: List[Dict[str, Any]] = []
        for gen_idx, (gen_pil, seed) in enumerate(zip(gen_images, seeds)):
            gen_preview_bgr = cv2.cvtColor(np.array(gen_pil), cv2.COLOR_RGB2BGR)
            composited_bgr = self._composite(
                composite_base_bgr, composite_base_rgb,
                gen_pil, hair_mask_for_sd, scale, pad, (W, H),
                protect_mask=protect_mask_for_sd,   # 얼굴 영역 alpha 침범 방지
                protect_release_mask=composite_bangs_release_mask if float(composite_bangs_release_mask.sum()) > 0.0 else None,
                hair_length=hair_length,
            )

            if (
                hair_length in ("short", "medium")
                and cutoff_y_for_post is not None
                and removal_mask_for_post is not None
            ):
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    post_rgb = self._final_cutoff_cleanup(
                        post_rgb,
                        face_bbox=face_bbox,
                        removal_mask=removal_mask_for_post,
                        cutoff_y=cutoff_y_for_post,
                        shoulder_protect=shoulder_protect_for_post,
                        neckline_preserve=neckline_preserve_for_post,
                        lateral_preserve=lateral_neck_preserve_for_post,
                        hair_length=hair_length,
                        center_anchor_mask=center_chest_strand_removal_mask,
                    )
                    post_rgb = self._remove_residual_hair_below_cutoff(
                        post_rgb,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        removal_mask=removal_mask_for_post,
                        shoulder_protect=shoulder_protect_for_post,
                        neckline_preserve=neckline_preserve_for_post,
                        lateral_preserve=lateral_neck_preserve_for_post,
                        hair_length=hair_length,
                        center_anchor_mask=center_chest_strand_removal_mask,
                    )
                    composited_bgr = cv2.cvtColor(post_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"[SDPipeline] short/medium 잔여물 cleanup 실패(무시): {e}")

            if not has_color_request:
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    post_rgb = self._preserve_original_hair_tone(
                        source_rgb=img_rgb,
                        target_rgb=post_rgb,
                        face_bbox=face_bbox,
                    )
                    composited_bgr = cv2.cvtColor(post_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"[SDPipeline] 원본 컬러 유지 보정 실패(무시): {e}")

            if has_color_request and hair_length in {"short", "long"}:
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    bangs_tone_mask = (
                        composite_bangs_release_mask
                        if float(composite_bangs_release_mask.sum()) > 0.0
                        else bangs_restore_for_sd
                    )
                    post_rgb = self._harmonize_short_bangs_tone(
                        img_rgb=post_rgb,
                        face_bbox=face_bbox,
                        bangs_mask=bangs_tone_mask,
                        target_lab=target_hair_lab,
                    )
                    composited_bgr = cv2.cvtColor(post_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"[SDPipeline] bangs tone harmonization failed (ignored): {e}")

            color_distance: Optional[float] = None
            color_score = 0.0
            if has_color_request and target_hair_lab is not None:
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    color_distance = self._estimate_hair_color_distance(
                        img_rgb=post_rgb,
                        face_bbox=face_bbox,
                        target_lab=target_hair_lab,
                    )
                    if color_distance is not None:
                        color_score = float(np.clip(1.0 - (color_distance / 80.0), 0.0, 1.0))
                except Exception as e:
                    logger.warning(f"[SDPipeline] 색상 거리 계산 실패(무시): {e}")

            tail_penalty: Optional[float] = None
            if (
                hair_length == "short"
                and cutoff_y_for_post is not None
                and removal_mask_for_post is not None
            ):
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    tail_penalty = self._estimate_short_tail_penalty(
                        img_rgb=post_rgb,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        removal_mask=removal_mask_for_post,
                    )
                except Exception as e:
                    logger.warning(f"[SDPipeline] short tail penalty 계산 실패(무시): {e}")

            accessory_penalty: Optional[float] = None
            try:
                post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                accessory_penalty = self._estimate_accessory_penalty(
                    img_rgb=post_rgb,
                    face_bbox=face_bbox,
                )
            except Exception as e:
                logger.warning(f"[SDPipeline] accessory penalty 계산 실패(무시): {e}")

            male_medium_fit_penalty: Optional[float] = None
            if hair_length == "medium" and subject_gender_mode == "male":
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    male_medium_fit_penalty = self._estimate_male_medium_fit_penalty(
                        img_rgb=post_rgb,
                        face_bbox=face_bbox,
                        source_profile=male_medium_source_profile,
                    )
                except Exception as e:
                    logger.warning(f"[SDPipeline] male medium fit penalty 怨꾩궛 ?ㅽ뙣(臾댁떆): {e}")

            male_short_fit_penalty: Optional[float] = None
            if hair_length == "short" and subject_gender_mode == "male":
                try:
                    post_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
                    male_short_fit_penalty = self._estimate_male_short_fit_penalty(
                        img_rgb=post_rgb,
                        face_bbox=face_bbox,
                        source_profile=male_short_source_profile,
                        hairstyle_text=effective_hairstyle_text,
                    )
                except Exception as e:
                    logger.warning(f"[SDPipeline] male short fit penalty calculation failed (ignored): {e}")

            candidates.append({
                "seed": seed,
                "image_bgr": composited_bgr,
                "preview_bgr": gen_preview_bgr,
                "color_distance": color_distance,
                "color_score": color_score,
                "tail_penalty": tail_penalty,
                "accessory_penalty": accessory_penalty,
                "male_medium_fit_penalty": male_medium_fit_penalty,
                "male_short_fit_penalty": male_short_fit_penalty,
                "gen_idx": gen_idx,
            })

        prefer_short_color_first_ranking = (
            hair_length == "short"
            and subject_gender_mode != "male"
            and has_color_request
            and target_hair_lab is not None
        )

        if has_color_request and target_hair_lab is not None and len(candidates) > 1:
            sortable_count = sum(c["color_distance"] is not None for c in candidates)
            if sortable_count >= 2:
                if prefer_short_color_first_ranking:
                    candidates.sort(
                        key=lambda c: (
                            c["color_distance"] is None,
                            c["color_distance"] if c["color_distance"] is not None else 1e9,
                            c["tail_penalty"] is None,
                            c["tail_penalty"] if c["tail_penalty"] is not None else 1e9,
                            c["accessory_penalty"] is None,
                            c["accessory_penalty"] if c["accessory_penalty"] is not None else 1e9,
                            c["gen_idx"],
                        )
                    )
                else:
                    candidates.sort(
                        key=lambda c: (
                            c["male_medium_fit_penalty"] is None,
                            c["male_medium_fit_penalty"] if c["male_medium_fit_penalty"] is not None else 1e9,
                            c["male_short_fit_penalty"] is None,
                            c["male_short_fit_penalty"] if c["male_short_fit_penalty"] is not None else 1e9,
                            c["color_distance"] is None,
                            c["color_distance"] if c["color_distance"] is not None else 1e9,
                            c["accessory_penalty"] is None,
                            c["accessory_penalty"] if c["accessory_penalty"] is not None else 1e9,
                            c["gen_idx"],
                        )
                    )
                logger.info("[SDPipeline] 컬러 유사도 기준으로 결과 재정렬 완료")
            else:
                logger.info("[SDPipeline] 컬러 유사도 재정렬 스킵 (유효 샘플 부족)")
        elif len(candidates) > 1:
            accessory_sortable = sum(c["accessory_penalty"] is not None for c in candidates)
            fit_sortable = sum(c["male_medium_fit_penalty"] is not None for c in candidates)
            short_fit_sortable = sum(c["male_short_fit_penalty"] is not None for c in candidates)
            if accessory_sortable >= 2 or fit_sortable >= 2 or short_fit_sortable >= 2:
                candidates.sort(
                    key=lambda c: (
                        c["accessory_penalty"] is None,
                        c["accessory_penalty"] if c["accessory_penalty"] is not None else 1e9,
                        c["male_medium_fit_penalty"] is None,
                        c["male_medium_fit_penalty"] if c["male_medium_fit_penalty"] is not None else 1e9,
                        c["male_short_fit_penalty"] is None,
                        c["male_short_fit_penalty"] if c["male_short_fit_penalty"] is not None else 1e9,
                        c["gen_idx"],
                    )
                )
                if fit_sortable >= 2:
                    logger.info("[SDPipeline] male medium fit ranking applied")
                elif short_fit_sortable >= 2:
                    logger.info("[SDPipeline] male short fit ranking applied")
                else:
                    logger.info("[SDPipeline] accessory penalty ranking applied")

        if hair_length == "short" and len(candidates) > 1:
            tail_sortable = sum(c["tail_penalty"] is not None for c in candidates)
            short_fit_sortable = sum(c["male_short_fit_penalty"] is not None for c in candidates)
            if prefer_short_color_first_ranking and tail_sortable >= 2:
                candidates.sort(
                    key=lambda c: (
                        c["color_distance"] is None,
                        c["color_distance"] if c["color_distance"] is not None else 1e9,
                        c["tail_penalty"] is None,
                        c["tail_penalty"] if c["tail_penalty"] is not None else 1e9,
                        c["accessory_penalty"] is None,
                        c["accessory_penalty"] if c["accessory_penalty"] is not None else 1e9,
                        c["gen_idx"],
                    )
                )
                logger.info("[SDPipeline] short female color-first + short tail ranking applied")
            elif tail_sortable >= 2 or short_fit_sortable >= 2:
                candidates.sort(
                    key=lambda c: (
                        c["accessory_penalty"] is None,
                        c["accessory_penalty"] if c["accessory_penalty"] is not None else 1e9,
                        c["male_short_fit_penalty"] is None,
                        c["male_short_fit_penalty"] if c["male_short_fit_penalty"] is not None else 1e9,
                        c["tail_penalty"] is None,
                        c["tail_penalty"] if c["tail_penalty"] is not None else 1e9,
                        c["color_distance"] is None,
                        c["color_distance"] if c["color_distance"] is not None else 1e9,
                        c["gen_idx"],
                    )
                )
                if short_fit_sortable >= 2:
                    logger.info("[SDPipeline] male short fit + short tail ranking applied")
                else:
                    logger.info("[SDPipeline] short tail penalty ranking applied")

        results: List[SDInpaintResult] = []
        for rank, cand in enumerate(candidates[:requested_top_k]):
            if debug_images_common is not None and rank == 0:
                debug_images_common["sd_generated_rank0_512"] = cand["preview_bgr"]
            final_bgr = cand["image_bgr"]
            pre_post_cloth_center_residual_mask = np.zeros(final_bgr.shape[:2], dtype=np.float32)
            if (
                self.config.enable_post_cloth_refine
                and hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                    pre_post_cloth_center_residual_mask = np.clip(
                        self._build_center_residual_detector_mask(
                            current_rgb=final_rgb,
                            source_rgb=img_rgb,
                            cloth_mask=cloth_restore_mask_for_post,
                            removal_mask=removal_mask_for_post,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y_for_post,
                            hair_length=hair_length,
                            final_hair_mask=final_hair_mask,
                            center_support_mask=center_chest_strand_removal_mask,
                            anchor_mask=(
                                subject_cloth_anchor_for_post if use_short_dark_cloth_anchor_fallback else None
                            ),
                        ).astype(np.float32),
                        0.0,
                        1.0,
                    )
                    cloth_refine_mask = self._build_post_cloth_refine_mask(
                        removal_mask=removal_mask_for_post,
                        cloth_mask=cloth_restore_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        subject_gender_mode=subject_gender_mode,
                        protect_mask=protect_mask_for_sd,
                        final_hair_mask=final_hair_mask,
                        artifact_cleanup_mask=artifact_cleanup_mask_for_post,
                        exclusion_mask=pre_post_cloth_center_residual_mask,
                    )
                    cloth_refine_u8 = (cloth_refine_mask > 0.08).astype(np.uint8) * 255
                    if debug_images_common is not None and rank == 0:
                        pre_post_cloth_center_residual_u8 = (
                            (np.clip(pre_post_cloth_center_residual_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(
                                np.uint8
                            )
                            * 255
                        )
                        debug_images_common["pipeline_pre_post_cloth_center_residual_exclusion_mask"] = cv2.cvtColor(
                            pre_post_cloth_center_residual_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                    if int((cloth_refine_u8 > 0).sum()) >= 100:
                        final_rgb = self._sd_refine_removed_region(
                            base_rgb=final_rgb,
                            removal_mask=cloth_refine_mask,
                            face_bbox=face_bbox,
                            face_crop_pil=face_crop_pil,
                            protect_mask=protect_mask_for_sd,
                            cloth_mask=cloth_restore_mask_for_post,
                            hair_length=hair_length,
                            seed=int(cand["seed"]) + 1701,
                            reference_rgb=img_rgb,
                            refine_mode="cloth",
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        if debug_images_common is not None and rank == 0:
                            debug_images_common["pipeline_post_cloth_refine_mask"] = cv2.cvtColor(
                                cloth_refine_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                    if int((cloth_refine_u8 > 0).sum()) >= 80:
                        final_rgb = self._cv2_refine_cloth_region(
                            final_rgb,
                            cloth_refine_mask,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_restore_mask_for_post,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"[SDPipeline] post cloth refine 실패(무시): {e}")
            if (
                hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and artifact_cleanup_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                    x1, y1, x2, y2 = face_bbox
                    face_w = max(int(x2 - x1), 1)
                    face_h = max(int(y2 - y1), 1)
                    final_dark_tail_u8 = np.zeros((H, W), dtype=np.uint8)

                    cleanup_u8 = (
                        (np.clip(artifact_cleanup_mask_for_post.astype(np.float32), 0.0, 1.0) > 0.18).astype(np.uint8) * 255
                    )
                    cleanup_u8 = cv2.bitwise_and(
                        cleanup_u8,
                        (np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0) > 0.10).astype(np.uint8) * 255,
                    )
                    cleanup_corridor_u8 = np.zeros((H, W), dtype=np.uint8)
                    cleanup_top = max(0, int(cutoff_y_for_post - face_h * 0.02))
                    cleanup_bottom = min(H, int(cutoff_y_for_post + face_h * (0.78 if hair_length == "short" else 0.92)))
                    cleanup_left = max(0, int(x1 - face_w * (0.84 if hair_length == "short" else 1.12)))
                    cleanup_right = min(W, int(x2 + face_w * (0.84 if hair_length == "short" else 1.12)))
                    if cleanup_top < cleanup_bottom and cleanup_left < cleanup_right:
                        cleanup_corridor_u8[cleanup_top:cleanup_bottom, cleanup_left:cleanup_right] = 255
                        cleanup_u8 = cv2.bitwise_and(cleanup_u8, cleanup_corridor_u8)
                    cleanup_anchor_u8 = cleanup_u8.copy()
                    if hair_length == "short":
                        cleanup_u8 = self._filter_short_center_cleanup_mask(
                            cleanup_u8,
                            face_bbox,
                            cutoff_y_for_post,
                            anchor_u8=cleanup_anchor_u8,
                            top_scale=0.00,
                            bottom_scale=0.78,
                            half_w_scale=0.25,
                            shrink_half_w_scale=0.18,
                            max_area_scale=0.10,
                            max_width_scale=0.34,
                            min_height_scale=0.10,
                            center_allow_scale=0.18,
                            max_total_scale=0.05,
                        )
                    support_cleanup_u8 = np.zeros((H, W), dtype=np.uint8)
                    support_cleanup_mask = lower_tail_support_for_post
                    if support_cleanup_mask is None or support_cleanup_mask.shape != (H, W):
                        support_cleanup_mask = lower_tail_support_mask
                    support_corridor_u8 = np.zeros((H, W), dtype=np.uint8)
                    support_top = max(0, int(cutoff_y_for_post - face_h * 0.03))
                    support_bottom = min(H, int(cutoff_y_for_post + face_h * (0.92 if hair_length == "short" else 1.18)))
                    support_left = max(0, int(x1 - face_w * (0.62 if hair_length == "short" else 0.92)))
                    support_right = min(W, int(x2 + face_w * (0.62 if hair_length == "short" else 0.92)))
                    if support_top < support_bottom and support_left < support_right:
                        support_corridor_u8[support_top:support_bottom, support_left:support_right] = 255
                    center_anchor_u8 = np.zeros((H, W), dtype=np.uint8)
                    if float(center_chest_strand_mask.sum()) > 0.0:
                        center_anchor_u8 = cv2.dilate(
                            (np.clip(center_chest_strand_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13) if hair_length == "short" else (11, 25)),
                            iterations=1,
                        )
                        center_anchor_u8 = cv2.bitwise_and(center_anchor_u8, support_corridor_u8)
                        center_anchor_u8 = cv2.bitwise_and(
                            center_anchor_u8,
                            (np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                        )
                        cleanup_anchor_u8 = cv2.bitwise_or(cleanup_anchor_u8, center_anchor_u8)
                    if support_cleanup_mask is not None and support_cleanup_mask.shape == (H, W):
                        support_cleanup_u8 = (
                            (np.clip(support_cleanup_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                        )
                        support_cleanup_u8 = cv2.bitwise_and(support_cleanup_u8, support_corridor_u8)
                        support_cleanup_u8 = cv2.bitwise_and(
                            support_cleanup_u8,
                            (np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                        )
                        support_cleanup_u8 = cv2.dilate(
                            support_cleanup_u8,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13) if hair_length == "short" else (9, 17)),
                            iterations=1,
                        )
                        if hair_length == "short":
                            support_cleanup_u8 = self._filter_short_center_cleanup_mask(
                                support_cleanup_u8,
                                face_bbox,
                                cutoff_y_for_post,
                                anchor_u8=cleanup_anchor_u8,
                                top_scale=0.00,
                                bottom_scale=0.86,
                                half_w_scale=0.24,
                                shrink_half_w_scale=0.18,
                                max_area_scale=0.10,
                                max_width_scale=0.34,
                                min_height_scale=0.10,
                                center_allow_scale=0.18,
                                max_total_scale=0.05,
                            )
                        cleanup_u8 = cv2.bitwise_or(cleanup_u8, support_cleanup_u8)
                    if float(center_chest_strand_mask.sum()) > 0.0:
                        center_cleanup_u8 = center_anchor_u8.copy()
                        if hair_length == "short":
                            center_cleanup_u8 = self._filter_short_center_cleanup_mask(
                                center_cleanup_u8,
                                face_bbox,
                                cutoff_y_for_post,
                                anchor_u8=center_anchor_u8,
                                top_scale=0.00,
                                bottom_scale=0.86,
                                half_w_scale=0.22,
                                shrink_half_w_scale=0.17,
                                max_area_scale=0.08,
                                max_width_scale=0.30,
                                min_height_scale=0.12,
                                center_allow_scale=0.16,
                                max_total_scale=0.04,
                            )
                        cleanup_u8 = cv2.bitwise_or(cleanup_u8, center_cleanup_u8)
                    final_dark_tail = self._build_dark_tail_residual_mask(
                        img_rgb=final_rgb,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                    )
                    final_dark_tail_u8 = (final_dark_tail > 0.08).astype(np.uint8) * 255
                    if int((final_dark_tail_u8 > 0).sum()) > 0:
                        final_dark_tail_u8 = cv2.bitwise_and(final_dark_tail_u8, cleanup_corridor_u8)
                        final_dark_tail_u8 = cv2.bitwise_and(
                            final_dark_tail_u8,
                            (np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
                        )
                        if hair_length == "short":
                            final_dark_tail_u8 = self._filter_short_center_cleanup_mask(
                                final_dark_tail_u8,
                                face_bbox,
                                cutoff_y_for_post,
                                anchor_u8=cleanup_anchor_u8,
                                top_scale=0.00,
                                bottom_scale=0.82,
                                half_w_scale=0.25,
                                shrink_half_w_scale=0.18,
                                max_area_scale=0.10,
                                max_width_scale=0.34,
                                min_height_scale=0.10,
                                center_allow_scale=0.18,
                                max_total_scale=0.05,
                            )
                        cleanup_u8 = cv2.bitwise_or(cleanup_u8, final_dark_tail_u8)
                    final_hair_u8 = cv2.dilate(
                        (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24).astype(np.uint8) * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13)),
                        iterations=1,
                    )
                    cleanup_u8 = cv2.bitwise_and(cleanup_u8, cv2.bitwise_not(final_hair_u8))
                    if hair_length == "short":
                        cleanup_u8 = self._filter_short_center_cleanup_mask(
                            cleanup_u8,
                            face_bbox,
                            cutoff_y_for_post,
                            anchor_u8=cleanup_anchor_u8,
                            top_scale=0.00,
                            bottom_scale=0.84,
                            half_w_scale=0.26,
                            shrink_half_w_scale=0.18,
                            max_area_scale=0.11,
                            max_width_scale=0.36,
                            min_height_scale=0.10,
                            center_allow_scale=0.18,
                            max_total_scale=0.05,
                        )
                    cleanup_u8 = cv2.dilate(
                        cleanup_u8,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13) if hair_length == "short" else (11, 19)),
                        iterations=1,
                    )
                    if hair_length == "short":
                        cleanup_u8 = self._filter_short_center_cleanup_mask(
                            cleanup_u8,
                            face_bbox,
                            cutoff_y_for_post,
                            anchor_u8=cleanup_anchor_u8,
                            top_scale=0.00,
                            bottom_scale=0.84,
                            half_w_scale=0.27,
                            shrink_half_w_scale=0.19,
                            max_area_scale=0.12,
                            max_width_scale=0.38,
                            min_height_scale=0.10,
                            center_allow_scale=0.19,
                            max_total_scale=0.06,
                        )
                    micro_cleanup_u8 = (
                        self._build_micro_cloth_artifact_mask(
                            img_rgb=final_rgb,
                            cloth_mask=cloth_mask_dilated,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y_for_post,
                            hair_length=hair_length,
                            final_hair_mask=final_hair_mask,
                        ) > 0.08
                    ).astype(np.uint8) * 255
                    if int((micro_cleanup_u8 > 0).sum()) > 0:
                        cleanup_u8 = cv2.bitwise_or(cleanup_u8, micro_cleanup_u8)
                        if debug_images_common is not None and rank == 0:
                            debug_images_common["pipeline_micro_cloth_artifact_mask"] = cv2.cvtColor(
                                micro_cleanup_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                    if hair_length == "short":
                        cleanup_u8 = self._filter_short_center_cleanup_mask(
                            cleanup_u8,
                            face_bbox,
                            cutoff_y_for_post,
                            anchor_u8=cleanup_anchor_u8,
                            top_scale=0.02,
                            bottom_scale=0.82,
                            half_w_scale=0.24,
                            shrink_half_w_scale=0.17,
                            max_area_scale=0.10,
                            max_width_scale=0.34,
                            min_height_scale=0.10,
                            center_allow_scale=0.18,
                            max_total_scale=0.045,
                        )
                    if int((cleanup_u8 > 0).sum()) >= 80:
                        final_rgb = self._lama_inpaint(final_rgb, cleanup_u8)
                        if int((final_dark_tail_u8 > 0).sum()) >= 40:
                            final_rgb = self._cv2_cleanup_dark_tail_blob(final_rgb, final_dark_tail_u8)
                        if hair_length != "short":
                            final_rgb = self._restore_cloth_overlap_from_source(
                                source_rgb=img_rgb,
                                current_rgb=final_rgb,
                                restore_mask=cleanup_u8.astype(np.float32) / 255.0,
                                final_hair_mask=final_hair_mask,
                            )
                        final_rgb = self._cv2_refine_cloth_region(
                            final_rgb,
                            cleanup_u8.astype(np.float32) / 255.0,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_mask_dilated,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        if debug_images_common is not None and rank == 0:
                            debug_images_common["pipeline_final_artifact_cloth_cleanup_mask"] = cv2.cvtColor(
                                cleanup_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                except Exception as e:
                    logger.warning(f"[SDPipeline] final artifact cloth cleanup ?ㅽ뙣(臾댁떆): {e}")
            if (
                self.config.enable_post_cloth_refine
                and hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                    _, shoulder_y1, _, shoulder_y2 = face_bbox
                    shoulder_face_h = max(int(shoulder_y2 - shoulder_y1), 1)
                    shoulder_cloth_refine_mask = self._build_shoulder_cloth_restore_mask(
                        removal_mask=removal_mask_for_post,
                        cloth_mask=cloth_mask_dilated,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        protect_mask=protect_mask_for_sd,
                        final_hair_mask=final_hair_mask,
                    )
                    if (
                        hair_length == "short"
                        and shoulder_cloth_release_for_post is not None
                        and shoulder_cloth_release_for_post.shape == (H, W)
                    ):
                        shoulder_source_restore_mask = np.clip(
                            shoulder_cloth_release_for_post.astype(np.float32),
                            0.0,
                            1.0,
                        )
                        shoulder_source_restore_mask[:max(0, int(cutoff_y_for_post - shoulder_face_h * 0.02)), :] = 0.0
                        shoulder_source_restore_mask = np.clip(
                            shoulder_source_restore_mask * np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0),
                            0.0,
                            1.0,
                        )
                        if float(shoulder_cloth_refine_mask.sum()) > 0.0:
                            shoulder_cloth_refine_mask = np.maximum(
                                shoulder_cloth_refine_mask,
                                shoulder_source_restore_mask * 0.92,
                            ).astype(np.float32)
                        else:
                            shoulder_cloth_refine_mask = shoulder_source_restore_mask.astype(np.float32)
                    shoulder_cloth_refine_u8 = (
                        (np.clip(shoulder_cloth_refine_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                    )
                    shoulder_refine_px = int((shoulder_cloth_refine_u8 > 0).sum())
                    if shoulder_refine_px >= 180:
                        final_rgb = self._sd_refine_removed_region(
                            base_rgb=final_rgb,
                            removal_mask=shoulder_cloth_refine_mask,
                            face_bbox=face_bbox,
                            face_crop_pil=face_crop_pil,
                            protect_mask=protect_mask_for_sd,
                            cloth_mask=cloth_mask_dilated,
                            hair_length=hair_length,
                            seed=int(cand["seed"]) + 1739,
                            reference_rgb=img_rgb,
                            refine_mode="cloth",
                        )
                    if 100 <= shoulder_refine_px < 8000:
                        final_rgb = self._cv2_refine_cloth_region(
                            final_rgb,
                            shoulder_cloth_refine_mask,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_mask_dilated,
                        )
                    if shoulder_refine_px >= 120:
                        final_rgb = self._restore_cloth_overlap_from_source(
                            source_rgb=img_rgb,
                            current_rgb=final_rgb,
                            restore_mask=shoulder_cloth_refine_mask,
                            final_hair_mask=final_hair_mask,
                        )
                    if shoulder_refine_px >= 100:
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_shoulder_cloth_refine_mask"] = cv2.cvtColor(
                            shoulder_cloth_refine_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                except Exception as e:
                    logger.warning(f"[SDPipeline] shoulder cloth refine failed (ignored): {e}")
            if (
                hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and cutoff_y_for_post is not None
            ):
                direct_side_column_restore_mask_for_post = np.zeros(final_bgr.shape[:2], dtype=np.float32)
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                    side_restore_cloth_mask = np.clip(
                        cloth_restore_mask_for_post.astype(np.float32),
                        0.0,
                        1.0,
                    )
                    if (
                        hair_length == "short"
                        and subject_cloth_anchor_for_post is not None
                        and subject_cloth_anchor_for_post.shape == (H, W)
                        and float(subject_cloth_anchor_for_post.sum()) > 0.0
                    ):
                        anchor_weight = 0.98 if subject_gender_mode != "male" else 0.90
                        side_restore_cloth_mask = np.maximum(
                            side_restore_cloth_mask,
                            np.clip(subject_cloth_anchor_for_post.astype(np.float32), 0.0, 1.0) * anchor_weight,
                        ).astype(np.float32)
                    side_column_candidate_mask = np.zeros((H, W), dtype=np.float32)
                    if removal_mask_for_post is not None and removal_mask_for_post.shape == (H, W):
                        side_column_candidate_mask = np.maximum(
                            side_column_candidate_mask,
                            np.clip(removal_mask_for_post.astype(np.float32), 0.0, 1.0) * 0.34,
                        ).astype(np.float32)
                    if artifact_cleanup_mask_for_post is not None and artifact_cleanup_mask_for_post.shape == (H, W):
                        side_column_candidate_mask = np.maximum(
                            side_column_candidate_mask,
                            np.clip(artifact_cleanup_mask_for_post.astype(np.float32), 0.0, 1.0),
                        ).astype(np.float32)
                    if shoulder_cloth_release_for_post is not None and shoulder_cloth_release_for_post.shape == (H, W):
                        side_column_candidate_mask = np.maximum(
                            side_column_candidate_mask,
                            np.clip(shoulder_cloth_release_for_post.astype(np.float32), 0.0, 1.0) * 0.96,
                        ).astype(np.float32)
                    if (
                        below_bob_cloth_restore_for_post is not None
                        and below_bob_cloth_restore_for_post.shape == (H, W)
                    ):
                        side_column_candidate_mask = np.maximum(
                            side_column_candidate_mask,
                            np.clip(below_bob_cloth_restore_for_post.astype(np.float32), 0.0, 1.0),
                        ).astype(np.float32)
                    if (
                        hair_length == "short"
                        and subject_cloth_anchor_for_post is not None
                        and subject_cloth_anchor_for_post.shape == (H, W)
                        and float(subject_cloth_anchor_for_post.sum()) > 0.0
                    ):
                        side_column_candidate_mask = np.maximum(
                            side_column_candidate_mask,
                            np.clip(subject_cloth_anchor_for_post.astype(np.float32), 0.0, 1.0) * 0.92,
                        ).astype(np.float32)
                    side_column_restore_mask = self._build_side_column_cloth_restore_mask(
                        img_rgb=final_rgb,
                        cloth_mask=side_restore_cloth_mask,
                        candidate_mask=side_column_candidate_mask,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        final_hair_mask=final_hair_mask,
                        subject_gender_mode=subject_gender_mode,
                    )
                    direct_side_column_restore_mask = self._build_direct_short_column_restore_mask(
                        removal_mask=removal_mask_for_post,
                        cloth_mask=side_restore_cloth_mask,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        subject_gender_mode=subject_gender_mode,
                    )
                    if hair_length == "short" and float(center_cloth_restore_exclusion_mask.sum()) > 0.0:
                        side_column_exclusion_weight = 0.30 if subject_gender_mode != "male" else 0.98
                        direct_side_exclusion_weight = 0.16 if subject_gender_mode != "male" else 1.12
                        side_column_restore_mask = np.clip(
                            side_column_restore_mask.astype(np.float32)
                            * (
                                1.0
                                - np.clip(
                                    center_cloth_restore_exclusion_mask * side_column_exclusion_weight,
                                    0.0,
                                    1.0,
                                )
                            ),
                            0.0,
                            1.0,
                        )
                        direct_side_column_restore_mask = np.clip(
                            direct_side_column_restore_mask.astype(np.float32)
                            * (
                                1.0
                                - np.clip(
                                    center_cloth_restore_exclusion_mask * direct_side_exclusion_weight,
                                    0.0,
                                    1.0,
                                )
                            ),
                            0.0,
                            1.0,
                        )
                    direct_side_column_restore_mask_for_post = np.clip(
                        direct_side_column_restore_mask.astype(np.float32),
                        0.0,
                        1.0,
                    )
                    if float(direct_side_column_restore_mask.sum()) > 0.0:
                        side_column_restore_mask = np.maximum(
                            side_column_restore_mask,
                            direct_side_column_restore_mask,
                        ).astype(np.float32)
                    if (
                        below_bob_cloth_restore_for_post is not None
                        and below_bob_cloth_restore_for_post.shape == (H, W)
                    ):
                        side_column_restore_mask = np.maximum(
                            side_column_restore_mask,
                            np.clip(below_bob_cloth_restore_for_post.astype(np.float32), 0.0, 1.0),
                        ).astype(np.float32)
                    if hair_length == "short" and float(center_cloth_restore_exclusion_mask.sum()) > 0.0:
                        merged_side_exclusion_weight = 0.40 if subject_gender_mode != "male" else 1.18
                        side_column_restore_mask = np.clip(
                            side_column_restore_mask.astype(np.float32)
                            * (
                                1.0
                                - np.clip(
                                    center_cloth_restore_exclusion_mask * merged_side_exclusion_weight,
                                    0.0,
                                    1.0,
                                )
                            ),
                            0.0,
                            1.0,
                        )
                    side_column_restore_u8 = (
                        (np.clip(side_column_restore_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                    )
                    side_column_px = int((side_column_restore_u8 > 0).sum())
                    if side_column_px >= 160:
                        if hair_length != "short":
                            final_rgb = self._sd_refine_removed_region(
                                base_rgb=final_rgb,
                                removal_mask=side_column_restore_mask,
                                face_bbox=face_bbox,
                                face_crop_pil=face_crop_pil,
                                protect_mask=protect_mask_for_sd,
                                cloth_mask=cloth_mask_dilated,
                                hair_length=hair_length,
                                seed=int(cand["seed"]) + 1787,
                                reference_rgb=img_rgb,
                                refine_mode="cloth",
                            )
                        if hair_length == "short":
                            final_rgb = self._cleanup_region_with_cloth_restore(
                                source_rgb=img_rgb,
                                current_rgb=final_rgb,
                                cleanup_mask=side_column_restore_mask,
                                cloth_mask=side_restore_cloth_mask,
                                final_hair_mask=final_hair_mask,
                                ignore_final_hair_for_cloth_restore=True,
                                cleanup_dark_tail=True,
                            )
                        else:
                            final_rgb = self._restore_cloth_overlap_from_source(
                                source_rgb=img_rgb,
                                current_rgb=final_rgb,
                                restore_mask=side_column_restore_mask,
                                final_hair_mask=final_hair_mask,
                            )
                            final_rgb = self._cv2_refine_cloth_region(final_rgb, side_column_restore_mask)
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_side_column_cloth_restore_mask"] = cv2.cvtColor(
                            side_column_restore_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                        debug_images_common["pipeline_direct_short_column_restore_mask"] = cv2.cvtColor(
                            (
                                (np.clip(direct_side_column_restore_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8)
                                * 255
                            ),
                            cv2.COLOR_GRAY2BGR,
                        )
                except Exception as e:
                    logger.warning(f"[SDPipeline] side column cloth restore failed (ignored): {e}")
            if (
                hair_length == "short"
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                    short_side_lane_refine_mask = self._build_short_final_side_lane_refine_mask(
                        img_rgb=final_rgb,
                        final_hair_mask=final_hair_mask,
                        cloth_mask=cloth_mask_dilated,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                    )
                    short_side_lane_refine_u8 = (
                        (np.clip(short_side_lane_refine_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8)
                        * 255
                    )
                    short_side_lane_px = int((short_side_lane_refine_u8 > 0).sum())
                    if short_side_lane_px >= 140:
                        final_rgb = self._sd_refine_removed_region(
                            base_rgb=final_rgb,
                            removal_mask=short_side_lane_refine_mask,
                            face_bbox=face_bbox,
                            face_crop_pil=face_crop_pil,
                            protect_mask=protect_mask_for_sd,
                            cloth_mask=cloth_mask_dilated,
                            hair_length=hair_length,
                            seed=int(cand["seed"]) + 1823,
                            reference_rgb=img_rgb,
                            refine_mode="short_tail",
                        )
                        short_side_lane_cloth_mask = np.clip(
                            short_side_lane_refine_mask.astype(np.float32)
                            * np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0),
                            0.0,
                            1.0,
                        )
                        if float(short_side_lane_cloth_mask.sum()) >= 80.0:
                            final_rgb = self._restore_cloth_overlap_from_source(
                                source_rgb=img_rgb,
                                current_rgb=final_rgb,
                                restore_mask=short_side_lane_cloth_mask,
                                final_hair_mask=None,
                            )
                            final_rgb = self._blend_neighbor_cloth_tone(
                                final_rgb,
                                short_side_lane_cloth_mask,
                                cloth_mask=cloth_mask_dilated,
                                reference_rgb=img_rgb,
                            )
                            final_rgb = self._cv2_refine_cloth_region(
                                final_rgb,
                                short_side_lane_cloth_mask,
                                reference_rgb=img_rgb,
                                reference_mask=cloth_mask_dilated,
                            )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_short_final_side_lane_refine_mask"] = cv2.cvtColor(
                            short_side_lane_refine_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                except Exception as e:
                    logger.warning(f"[SDPipeline] short side lane refine failed (ignored): {e}")
            if (
                hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                if hair_length == "short":
                    try:
                        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                        final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                        short_lower_tail_mask = self._build_short_lower_tail_cleanup_mask(
                            img_rgb=final_rgb,
                            cloth_mask=cloth_mask_dilated,
                            removal_mask=removal_mask_for_post,
                            final_hair_mask=final_hair_mask,
                            face_bbox=face_bbox,
                            cutoff_y=cutoff_y_for_post,
                            hair_length=hair_length,
                        )
                        short_lower_tail_u8 = (
                            (np.clip(short_lower_tail_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                        )
                        short_lower_tail_px = int((short_lower_tail_u8 > 0).sum())
                        if short_lower_tail_px >= 80:
                            final_rgb = self._cleanup_region_with_cloth_restore(
                                source_rgb=img_rgb,
                                current_rgb=final_rgb,
                                cleanup_mask=short_lower_tail_mask,
                                cloth_mask=cloth_mask_dilated,
                                ignore_final_hair_for_cloth_restore=True,
                                cleanup_dark_tail=True,
                            )
                            final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                        if debug_images_common is not None and rank == 0:
                            debug_images_common["pipeline_short_lower_tail_cleanup_mask"] = cv2.cvtColor(
                                short_lower_tail_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                    except Exception as e:
                        logger.warning(f"[SDPipeline] short lower tail cleanup failed (ignored): {e}")
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    dark_lane_mask = self._build_dark_lane_cleanup_mask(
                        img_rgb=final_rgb,
                        cloth_mask=cloth_restore_mask_for_post,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        anchor_mask=subject_cloth_anchor_for_post if use_short_dark_cloth_anchor_fallback else None,
                    )
                    dark_lane_u8 = (
                        (np.clip(dark_lane_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                    )
                    dark_lane_px = int((dark_lane_u8 > 0).sum())
                    if dark_lane_px >= 40:
                        final_rgb = self._lama_inpaint(final_rgb, dark_lane_u8)
                        final_rgb = self._cv2_refine_cloth_region(
                            final_rgb,
                            dark_lane_mask,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_mask_dilated,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_dark_lane_cleanup_mask"] = cv2.cvtColor(
                            dark_lane_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                except Exception as e:
                    logger.warning(f"[SDPipeline] dark lane cleanup failed (ignored): {e}")
            if (
                hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                    final_hair_lane_mask = self._build_final_hair_lane_cleanup_mask(
                        final_hair_mask=final_hair_mask,
                        cloth_mask=cloth_restore_mask_for_post,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        anchor_mask=subject_cloth_anchor_for_post if use_short_dark_cloth_anchor_fallback else None,
                    )
                    final_hair_lane_u8 = (
                        (np.clip(final_hair_lane_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                    )
                    final_hair_lane_px = int((final_hair_lane_u8 > 0).sum())
                    if final_hair_lane_px >= 40:
                        final_rgb = self._lama_inpaint(final_rgb, final_hair_lane_u8)
                        final_rgb = self._cv2_cleanup_dark_tail_blob(final_rgb, final_hair_lane_u8)
                        final_rgb = self._cv2_refine_cloth_region(
                            final_rgb,
                            final_hair_lane_mask,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_mask_dilated,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_final_hair_lane_cleanup_mask"] = cv2.cvtColor(
                            final_hair_lane_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                except Exception as e:
                    logger.warning(f"[SDPipeline] final hair lane cleanup failed (ignored): {e}")
            if (
                hair_length == "short"
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                    short_bob_tail_mask = self._build_short_bob_tail_suppress_mask(
                        img_rgb=final_rgb,
                        final_hair_mask=final_hair_mask,
                        cloth_mask=cloth_mask_dilated,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                    )
                    short_bob_tail_u8 = (
                        (np.clip(short_bob_tail_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                    )
                    short_bob_tail_px = int((short_bob_tail_u8 > 0).sum())
                    if short_bob_tail_px >= 60:
                        final_rgb = self._cleanup_region_with_cloth_restore(
                            source_rgb=img_rgb,
                            current_rgb=final_rgb,
                            cleanup_mask=short_bob_tail_mask,
                            cloth_mask=cloth_mask_dilated,
                            ignore_final_hair_for_cloth_restore=True,
                            cleanup_dark_tail=True,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_short_bob_tail_suppress_mask"] = cv2.cvtColor(
                            short_bob_tail_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                except Exception as e:
                    logger.warning(f"[SDPipeline] short bob tail suppress failed (ignored): {e}")
            if (
                hair_length == "short"
                and subject_gender_mode != "male"
                and cloth_mask_dilated is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                    deep_short_cloth_hair_cleanup_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                    if final_hair_mask is not None and final_hair_mask.shape == final_bgr.shape[:2]:
                        final_hair_u8 = cv2.dilate(
                            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8) * 255,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
                            iterations=1,
                        )
                        cloth_u8 = cv2.dilate(
                            (np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
                            iterations=1,
                        )
                        deep_gate_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                        deep_top = max(0, int(cutoff_y_for_post + face_h * 0.18))
                        deep_bottom = min(H, int(cutoff_y_for_post + face_h * 1.42))
                        deep_left = max(0, int(x1 - face_w * 1.06))
                        deep_right = min(W, int(x2 + face_w * 1.06))
                        if deep_top < deep_bottom and deep_left < deep_right:
                            deep_gate_u8[deep_top:deep_bottom, deep_left:deep_right] = 255
                        candidate_u8 = cv2.bitwise_and(final_hair_u8, cloth_u8)
                        candidate_u8 = cv2.bitwise_and(candidate_u8, deep_gate_u8)
                        if int((candidate_u8 > 0).sum()) >= 40:
                            keep_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
                            min_area = max(40, int(face_w * face_h * 0.002))
                            max_area = max(18000, int(face_w * face_h * 0.40))
                            min_height = max(34, int(face_h * 0.18))
                            max_width = max(172, int(face_w * 0.92))
                            deep_bottom_threshold = int(cutoff_y_for_post + face_h * 0.72)
                            center_accept_offset = max(24, int(face_w * 0.24))
                            max_offset = max(230, int(face_w * 1.02))
                            for idx in range(1, num_labels):
                                x = int(stats[idx, cv2.CC_STAT_LEFT])
                                y = int(stats[idx, cv2.CC_STAT_TOP])
                                w = int(stats[idx, cv2.CC_STAT_WIDTH])
                                h = int(stats[idx, cv2.CC_STAT_HEIGHT])
                                area = int(stats[idx, cv2.CC_STAT_AREA])
                                bottom_y = y + h
                                comp_cx = float(centroids[idx][0])
                                if area < min_area or area > max_area:
                                    continue
                                if h < min_height or w > max_width:
                                    continue
                                if bottom_y < deep_bottom_threshold:
                                    continue
                                if abs(comp_cx - face_cx) > max_offset:
                                    continue
                                fill_ratio = float(area) / float(max(w * h, 1))
                                if (
                                    abs(comp_cx - face_cx) > center_accept_offset
                                    and fill_ratio > 0.88
                                    and w > max(70, int(face_w * 0.36))
                                ):
                                    continue
                                keep_u8[labels == idx] = 255
                            if int((keep_u8 > 0).sum()) >= 40:
                                keep_u8 = cv2.morphologyEx(
                                    keep_u8,
                                    cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21)),
                                )
                                keep_u8 = cv2.dilate(
                                    keep_u8,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
                                    iterations=1,
                                )
                                deep_short_cloth_hair_cleanup_u8 = cv2.bitwise_and(keep_u8, deep_gate_u8)
                    if int((deep_short_cloth_hair_cleanup_u8 > 0).sum()) >= 60:
                        deep_short_cloth_hair_cleanup_mask = cv2.GaussianBlur(
                            deep_short_cloth_hair_cleanup_u8.astype(np.float32) / 255.0,
                            (0, 0),
                            sigmaX=3.8,
                            sigmaY=6.4,
                        ).astype(np.float32)
                        final_rgb = self._cleanup_region_with_cloth_restore(
                            source_rgb=img_rgb,
                            current_rgb=final_rgb,
                            cleanup_mask=deep_short_cloth_hair_cleanup_mask,
                            cloth_mask=cloth_mask_dilated,
                            ignore_final_hair_for_cloth_restore=True,
                            cleanup_dark_tail=True,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_deep_short_cloth_hair_cleanup_mask"] = cv2.cvtColor(
                            deep_short_cloth_hair_cleanup_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                except Exception as e:
                    logger.warning(f"[SDPipeline] deep short cloth hair cleanup failed (ignored): {e}")
            short_lower_garment_cleanup_mask_for_post = np.zeros(final_bgr.shape[:2], dtype=np.float32)
            center_residual_cleanup_mask_for_post = np.zeros(final_bgr.shape[:2], dtype=np.float32)
            final_source_cloth_rescue_mask_for_post = np.zeros(final_bgr.shape[:2], dtype=np.float32)
            if (
                hair_length == "short"
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                    short_lower_garment_cleanup_mask = self._build_short_lower_garment_cleanup_mask(
                        current_rgb=final_rgb,
                        source_rgb=img_rgb,
                        removal_mask=removal_mask_for_post,
                        cloth_mask=cloth_restore_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        protect_mask=protect_mask_for_sd,
                        final_hair_mask=final_hair_mask,
                        anchor_mask=subject_cloth_anchor_for_post if use_short_dark_cloth_anchor_fallback else None,
                    )
                    short_lower_garment_cleanup_u8 = (
                        (np.clip(short_lower_garment_cleanup_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8)
                        * 255
                    )
                    short_lower_garment_cleanup_px = int((short_lower_garment_cleanup_u8 > 0).sum())
                    short_lower_garment_cleanup_mask_for_post = np.clip(
                        short_lower_garment_cleanup_mask.astype(np.float32),
                        0.0,
                        1.0,
                    )
                    if short_lower_garment_cleanup_px >= 100:
                        final_rgb = self._cleanup_region_with_cloth_restore(
                            source_rgb=img_rgb,
                            current_rgb=final_rgb,
                            cleanup_mask=short_lower_garment_cleanup_mask,
                            cloth_mask=cloth_restore_mask_for_post,
                            ignore_final_hair_for_cloth_restore=True,
                            cleanup_dark_tail=True,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_short_lower_garment_cleanup_mask"] = cv2.cvtColor(
                            short_lower_garment_cleanup_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                except Exception as e:
                    logger.warning(f"[SDPipeline] short lower garment cleanup failed (ignored): {e}")
            center_residual_mask_for_post = np.zeros(final_bgr.shape[:2], dtype=np.float32)
            residual_strand_cleanup_mask_for_post = np.zeros(final_bgr.shape[:2], dtype=np.float32)
            if (
                hair_length in ("short", "medium")
                and cloth_restore_mask_for_post is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                    center_residual_mask = self._build_center_residual_detector_mask(
                        current_rgb=final_rgb,
                        source_rgb=img_rgb,
                        cloth_mask=cloth_restore_mask_for_post,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        final_hair_mask=final_hair_mask,
                        center_support_mask=center_chest_strand_removal_mask,
                        anchor_mask=subject_cloth_anchor_for_post,
                    )
                    center_residual_mask_for_post = np.clip(center_residual_mask.astype(np.float32), 0.0, 1.0)
                    center_residual_u8 = (
                        (np.clip(center_residual_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                    )
                    center_residual_px = int((center_residual_u8 > 0).sum())
                    if center_residual_px >= 24:
                        center_residual_cleanup_u8 = cv2.dilate(
                            center_residual_u8,
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (25, 41) if hair_length == "short" else (21, 33),
                            ),
                            iterations=1,
                        )
                        if (
                            hair_length == "short"
                            and center_chest_strand_removal_mask is not None
                            and center_chest_strand_removal_mask.shape == final_bgr.shape[:2]
                        ):
                            center_support_u8 = cv2.dilate(
                                (
                                    np.clip(center_chest_strand_removal_mask.astype(np.float32), 0.0, 1.0) > 0.05
                                ).astype(np.uint8)
                                * 255,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 33)),
                                iterations=1,
                            )
                            center_cleanup_gate_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                            center_gate_half_w = max(24, int((x2 - x1) * 0.22))
                            center_gate_left = max(0, int(0.5 * (x1 + x2)) - center_gate_half_w)
                            center_gate_right = min(W, int(0.5 * (x1 + x2)) + center_gate_half_w)
                            center_gate_top = max(0, int(cutoff_y_for_post + max(y2 - y1, 1) * 0.04))
                            center_gate_bottom = min(H, int(cutoff_y_for_post + max(y2 - y1, 1) * 1.42))
                            if center_gate_top < center_gate_bottom and center_gate_left < center_gate_right:
                                center_cleanup_gate_u8[
                                    center_gate_top:center_gate_bottom,
                                    center_gate_left:center_gate_right,
                                ] = 255
                                center_support_u8 = cv2.bitwise_and(center_support_u8, center_cleanup_gate_u8)
                                center_residual_cleanup_u8 = cv2.bitwise_or(
                                    center_residual_cleanup_u8,
                                    center_support_u8,
                                )
                        if cloth_restore_mask_for_post is not None and cloth_restore_mask_for_post.shape == final_bgr.shape[:2]:
                            center_residual_cleanup_u8 = cv2.bitwise_and(
                                center_residual_cleanup_u8,
                                cv2.dilate(
                                    (
                                        np.clip(cloth_restore_mask_for_post.astype(np.float32), 0.0, 1.0) > 0.04
                                    ).astype(np.uint8)
                                    * 255,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                                    iterations=1,
                                ),
                            )
                        center_residual_cleanup_mask = cv2.GaussianBlur(
                            center_residual_cleanup_u8.astype(np.float32) / 255.0,
                            (0, 0),
                            sigmaX=3.8,
                            sigmaY=6.6,
                        ).astype(np.float32)
                        center_residual_cleanup_mask_for_post = np.clip(
                            center_residual_cleanup_mask.astype(np.float32),
                            0.0,
                            1.0,
                        )
                        final_rgb = self._cleanup_region_with_cloth_restore(
                            source_rgb=img_rgb,
                            current_rgb=final_rgb,
                            cleanup_mask=center_residual_cleanup_mask,
                            cloth_mask=cloth_restore_mask_for_post,
                            ignore_final_hair_for_cloth_restore=True,
                            cleanup_dark_tail=True,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_center_residual_detector_mask"] = cv2.cvtColor(
                            center_residual_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                        if center_residual_px >= 24:
                            debug_images_common["pipeline_center_residual_cleanup_mask"] = cv2.cvtColor(
                                center_residual_cleanup_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                except Exception as e:
                    logger.warning(f"[SDPipeline] center residual cleanup failed (ignored): {e}")
            if (
                hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                    residual_strand_mask = self._build_residual_strand_cleanup_mask(
                        img_rgb=final_rgb,
                        cloth_mask=cloth_mask_dilated,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        final_hair_mask=final_hair_mask,
                    )
                    residual_strand_u8 = (
                        (np.clip(residual_strand_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
                    )
                    residual_strand_cleanup_mask_for_post = np.clip(
                        residual_strand_mask.astype(np.float32),
                        0.0,
                        1.0,
                    )
                    residual_strand_px = int((residual_strand_u8 > 0).sum())
                    if residual_strand_px >= 16:
                        final_rgb = self._lama_inpaint(final_rgb, residual_strand_u8)
                        final_rgb = self._cv2_cleanup_dark_tail_blob(final_rgb, residual_strand_u8)
                        final_rgb = self._cv2_refine_cloth_region(
                            final_rgb,
                            residual_strand_mask,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_mask_dilated,
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_residual_strand_cleanup_mask"] = cv2.cvtColor(
                            residual_strand_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                except Exception as e:
                    logger.warning(f"[SDPipeline] residual strand cleanup failed (ignored): {e}")
            if (
                hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                    final_source_cloth_rescue_mask = self._build_final_source_cloth_rescue_mask(
                        current_rgb=final_rgb,
                        source_rgb=img_rgb,
                        removal_mask=removal_mask_for_post,
                        cloth_mask=cloth_restore_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        protect_mask=protect_mask_for_sd,
                        final_hair_mask=final_hair_mask,
                        center_support_mask=center_chest_strand_removal_mask,
                        anchor_mask=subject_cloth_anchor_for_post if use_short_dark_cloth_anchor_fallback else None,
                        exclusion_mask=center_residual_mask_for_post,
                        subject_gender_mode=subject_gender_mode,
                    )
                    center_residual_rescue_exclusion_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                    if hair_length == "short" and float(center_residual_mask_for_post.sum()) > 0.0:
                        center_residual_rescue_exclusion_u8 = cv2.dilate(
                            (np.clip(center_residual_mask_for_post.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8)
                            * 255,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 47)),
                            iterations=1,
                        )
                        center_exclusion_gate_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                        center_gate_half_w = max(28, int((x2 - x1) * 0.42))
                        center_gate_left = max(0, int(0.5 * (x1 + x2)) - center_gate_half_w)
                        center_gate_right = min(W, int(0.5 * (x1 + x2)) + center_gate_half_w)
                        center_gate_top = max(0, int(cutoff_y_for_post + max(y2 - y1, 1) * 0.04))
                        center_gate_bottom = min(H, int(cutoff_y_for_post + max(y2 - y1, 1) * 1.30))
                        if center_gate_top < center_gate_bottom and center_gate_left < center_gate_right:
                            center_exclusion_gate_u8[
                                center_gate_top:center_gate_bottom,
                                center_gate_left:center_gate_right,
                            ] = 255
                            center_residual_rescue_exclusion_u8 = cv2.bitwise_and(
                                center_residual_rescue_exclusion_u8,
                                center_exclusion_gate_u8,
                            )
                        if int((center_residual_rescue_exclusion_u8 > 0).sum()) >= 24:
                            exclusion_mask = cv2.GaussianBlur(
                                center_residual_rescue_exclusion_u8.astype(np.float32) / 255.0,
                                (0, 0),
                                sigmaX=3.8,
                                sigmaY=6.4,
                            ).astype(np.float32)
                            final_source_cloth_rescue_mask = np.clip(
                                final_source_cloth_rescue_mask.astype(np.float32) * (1.0 - exclusion_mask),
                                0.0,
                                1.0,
                            )
                    final_source_cloth_rescue_u8 = (
                        (np.clip(final_source_cloth_rescue_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8)
                        * 255
                    )
                    final_source_cloth_rescue_px = int((final_source_cloth_rescue_u8 > 0).sum())
                    final_source_cloth_rescue_mask_for_post = np.clip(
                        final_source_cloth_rescue_mask.astype(np.float32),
                        0.0,
                        1.0,
                    )
                    if final_source_cloth_rescue_px >= 120:
                        final_rgb = self._cleanup_region_with_cloth_restore(
                            source_rgb=img_rgb,
                            current_rgb=final_rgb,
                            cleanup_mask=final_source_cloth_rescue_mask,
                            cloth_mask=cloth_restore_mask_for_post,
                            final_hair_mask=final_hair_mask,
                            ignore_final_hair_for_cloth_restore=(hair_length == "short"),
                            cleanup_dark_tail=(hair_length == "short"),
                        )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_final_source_cloth_rescue_mask"] = cv2.cvtColor(
                            final_source_cloth_rescue_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                        debug_images_common["pipeline_center_residual_rescue_exclusion_mask"] = cv2.cvtColor(
                            center_residual_rescue_exclusion_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                except Exception as e:
                    logger.warning(f"[SDPipeline] final source cloth rescue failed (ignored): {e}")
                if (
                    hair_length == "short"
                    and subject_gender_mode != "male"
                    and cloth_restore_mask_for_post is not None
                    and cloth_restore_mask_for_post.shape == final_bgr.shape[:2]
                ):
                    try:
                        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                        final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                        female_short_cloth_reference_mask = np.clip(
                            cloth_restore_mask_for_post.astype(np.float32),
                            0.0,
                            1.0,
                        )
                        if (
                            cloth_mask_dilated is not None
                            and cloth_mask_dilated.shape == final_bgr.shape[:2]
                            and float(cloth_mask_dilated.sum()) > 0.0
                        ):
                            female_short_cloth_reference_mask = np.maximum(
                                female_short_cloth_reference_mask,
                                np.clip(cloth_mask_dilated.astype(np.float32) * 0.88, 0.0, 1.0),
                            )
                        if (
                            torso_cloth_preserve_for_post is not None
                            and torso_cloth_preserve_for_post.shape == final_bgr.shape[:2]
                            and float(torso_cloth_preserve_for_post.sum()) > 0.0
                        ):
                            female_short_cloth_reference_mask = np.maximum(
                                female_short_cloth_reference_mask,
                                np.clip(torso_cloth_preserve_for_post.astype(np.float32) * 0.82, 0.0, 1.0),
                            )
                        if (
                            bright_cloth_preserve_for_post is not None
                            and bright_cloth_preserve_for_post.shape == final_bgr.shape[:2]
                            and float(bright_cloth_preserve_for_post.sum()) > 0.0
                        ):
                            female_short_cloth_reference_mask = np.maximum(
                                female_short_cloth_reference_mask,
                                np.clip(bright_cloth_preserve_for_post.astype(np.float32) * 0.96, 0.0, 1.0),
                            )
                        female_short_direct_cloth_restore_mask = np.maximum(
                            short_lower_garment_cleanup_mask_for_post,
                            final_source_cloth_rescue_mask_for_post,
                        ).astype(np.float32)
                        female_short_direct_cloth_restore_mask = np.maximum(
                            female_short_direct_cloth_restore_mask,
                            np.clip(center_residual_cleanup_mask_for_post * 0.92, 0.0, 1.0),
                        ).astype(np.float32)
                        if (
                            residual_strand_cleanup_mask_for_post is not None
                            and residual_strand_cleanup_mask_for_post.shape == final_bgr.shape[:2]
                            and float(residual_strand_cleanup_mask_for_post.sum()) > 0.0
                        ):
                            female_short_direct_cloth_restore_mask = np.maximum(
                                female_short_direct_cloth_restore_mask,
                                np.clip(residual_strand_cleanup_mask_for_post * 0.90, 0.0, 1.0),
                            ).astype(np.float32)
                        female_short_reference_preserve_mask = np.zeros(
                            final_bgr.shape[:2],
                            dtype=np.float32,
                        )
                        if (
                            neckline_preserve_for_post is not None
                            and neckline_preserve_for_post.shape == final_bgr.shape[:2]
                        ):
                            female_short_reference_preserve_mask = np.maximum(
                                female_short_reference_preserve_mask,
                                np.clip(neckline_preserve_for_post.astype(np.float32), 0.0, 1.0),
                            ).astype(np.float32)
                        if (
                            lateral_neck_preserve_for_post is not None
                            and lateral_neck_preserve_for_post.shape == final_bgr.shape[:2]
                        ):
                            female_short_reference_preserve_mask = np.maximum(
                                female_short_reference_preserve_mask,
                                np.clip(lateral_neck_preserve_for_post.astype(np.float32), 0.0, 1.0) * 0.92,
                            ).astype(np.float32)
                        source_cloth_reference_mask = np.clip(
                            female_short_cloth_reference_mask.astype(np.float32),
                            0.0,
                            1.0,
                        )
                        source_cloth_reference_rgb = self._build_source_conditioned_cloth_base(
                            current_rgb=final_rgb,
                            source_rgb=img_rgb,
                            fill_mask=female_short_direct_cloth_restore_mask,
                            cloth_mask=source_cloth_reference_mask,
                            hair_length="short",
                            preserve_mask=female_short_reference_preserve_mask,
                        )
                        female_short_direct_cloth_restore_u8 = (
                            (
                                np.clip(female_short_direct_cloth_restore_mask.astype(np.float32), 0.0, 1.0) > 0.06
                            ).astype(np.uint8)
                            * 255
                        )
                        direct_female_short_restore_u8 = (
                            (
                                np.clip(direct_side_column_restore_mask_for_post.astype(np.float32), 0.0, 1.0) > 0.06
                            ).astype(np.uint8)
                            * 255
                        )
                        if int((female_short_direct_cloth_restore_u8 > 0).sum()) >= 120:
                            x1, y1, x2, y2 = face_bbox
                            face_w = max(int(x2 - x1), 1)
                            face_h = max(int(y2 - y1), 1)
                            cx = int(0.5 * (x1 + x2))
                            torso_gate_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                            gate_top = max(0, int(cutoff_y_for_post + face_h * 0.04))
                            gate_bottom = min(H, int(cutoff_y_for_post + face_h * 1.48))
                            gate_left = max(0, int(x1 - face_w * 1.28))
                            gate_right = min(W, int(x2 + face_w * 1.28))
                            if gate_top < gate_bottom and gate_left < gate_right:
                                torso_gate_u8[gate_top:gate_bottom, gate_left:gate_right] = 255
                            cloth_gate_u8 = cv2.dilate(
                                (
                                    np.clip(cloth_restore_mask_for_post.astype(np.float32), 0.0, 1.0) > 0.04
                                ).astype(np.uint8)
                                * 255,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
                                iterations=1,
                            )
                            female_short_direct_cloth_restore_u8 = cv2.bitwise_and(
                                female_short_direct_cloth_restore_u8,
                                torso_gate_u8,
                            )
                            direct_female_short_restore_u8 = cv2.bitwise_and(
                                direct_female_short_restore_u8,
                                torso_gate_u8,
                            )
                            female_short_direct_cloth_restore_u8 = cv2.bitwise_and(
                                female_short_direct_cloth_restore_u8,
                                cloth_gate_u8,
                            )
                            direct_female_short_restore_u8 = cv2.bitwise_and(
                                direct_female_short_restore_u8,
                                cloth_gate_u8,
                            )
                            if (
                                center_chest_strand_removal_mask is not None
                                and center_chest_strand_removal_mask.shape == final_bgr.shape[:2]
                            ):
                                center_support_u8 = cv2.dilate(
                                    (
                                        np.clip(center_chest_strand_removal_mask.astype(np.float32), 0.0, 1.0) > 0.05
                                    ).astype(np.uint8)
                                    * 255,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 37)),
                                    iterations=1,
                                )
                                center_gate_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                                center_half_w = max(26, int(face_w * 0.30))
                                center_left = max(0, cx - center_half_w)
                                center_right = min(W, cx + center_half_w)
                                center_top = max(0, int(cutoff_y_for_post + face_h * 0.04))
                                center_bottom = min(H, int(cutoff_y_for_post + face_h * 1.40))
                                if center_top < center_bottom and center_left < center_right:
                                    center_gate_u8[center_top:center_bottom, center_left:center_right] = 255
                                    center_support_u8 = cv2.bitwise_and(center_support_u8, center_gate_u8)
                                    center_support_u8 = cv2.bitwise_and(center_support_u8, cloth_gate_u8)
                                    female_short_direct_cloth_restore_u8 = cv2.bitwise_or(
                                        female_short_direct_cloth_restore_u8,
                                        center_support_u8,
                                    )
                            female_short_source_seed_u8 = cv2.bitwise_or(
                                female_short_direct_cloth_restore_u8,
                                direct_female_short_restore_u8,
                            )
                            source_gray = None
                            current_gray = None
                            source_sat = None
                            current_sat = None
                            diff_rgb = None
                            if int((female_short_source_seed_u8 > 0).sum()) >= 48:
                                source_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                                current_gray = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                                gray_delta = np.abs(current_gray - source_gray)
                                current_sat = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
                                source_sat = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
                                diff_rgb = np.abs(
                                    final_rgb.astype(np.float32) - img_rgb.astype(np.float32)
                                ).mean(axis=2)
                                female_short_source_support_u8 = cv2.dilate(
                                    female_short_source_seed_u8,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 23)),
                                    iterations=1,
                                )
                                female_short_source_rescue_u8 = (
                                    (
                                        (
                                            (diff_rgb > 12.0)
                                            | (gray_delta > 10.0)
                                            | (current_sat > source_sat + 12.0)
                                            | (current_gray + 10.0 < source_gray)
                                            | (current_gray > source_gray + 14.0)
                                        ).astype(np.uint8)
                                    )
                                    * 255
                                )
                                female_short_source_rescue_u8 = cv2.bitwise_and(
                                    female_short_source_rescue_u8,
                                    female_short_source_support_u8,
                                )
                                female_short_source_rescue_u8 = cv2.bitwise_and(
                                    female_short_source_rescue_u8,
                                    torso_gate_u8,
                                )
                                female_short_source_rescue_u8 = cv2.bitwise_and(
                                    female_short_source_rescue_u8,
                                    cloth_gate_u8,
                                )
                                female_short_direct_cloth_restore_u8 = cv2.bitwise_or(
                                    female_short_direct_cloth_restore_u8,
                                    female_short_source_rescue_u8,
                                )
                            if final_hair_mask is not None and final_hair_mask.shape == final_bgr.shape[:2]:
                                final_hair_u8 = cv2.dilate(
                                    (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8) * 255,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 19)),
                                    iterations=1,
                                )
                                final_hair_core_u8 = cv2.erode(
                                    final_hair_u8,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
                                    iterations=1,
                                )
                                upper_hair_guard_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                                upper_guard_top = max(0, int(cutoff_y_for_post - face_h * 0.04))
                                upper_guard_bottom = min(H, int(cutoff_y_for_post + face_h * 0.34))
                                if upper_guard_top < upper_guard_bottom:
                                    upper_hair_guard_u8[upper_guard_top:upper_guard_bottom, :] = 255
                                    upper_hair_guard_u8 = cv2.bitwise_and(upper_hair_guard_u8, final_hair_u8)
                                lower_hair_core_guard_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                                lower_guard_top = max(0, int(cutoff_y_for_post + face_h * 0.18))
                                lower_guard_bottom = min(H, int(cutoff_y_for_post + face_h * 1.42))
                                if lower_guard_top < lower_guard_bottom:
                                    lower_hair_core_guard_u8[lower_guard_top:lower_guard_bottom, :] = 255
                                    lower_hair_core_guard_u8 = cv2.bitwise_and(
                                        lower_hair_core_guard_u8,
                                        final_hair_core_u8,
                                    )
                                female_short_direct_cloth_restore_u8 = cv2.bitwise_and(
                                    female_short_direct_cloth_restore_u8,
                                    cv2.bitwise_not(cv2.bitwise_or(upper_hair_guard_u8, lower_hair_core_guard_u8)),
                                )
                                direct_female_short_restore_u8 = cv2.bitwise_and(
                                    direct_female_short_restore_u8,
                                    cv2.bitwise_not(cv2.bitwise_or(upper_hair_guard_u8, lower_hair_core_guard_u8)),
                                )
                            female_short_direct_cloth_restore_u8 = cv2.morphologyEx(
                                female_short_direct_cloth_restore_u8,
                                cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
                            )
                            female_short_direct_cloth_restore_u8 = cv2.dilate(
                                female_short_direct_cloth_restore_u8,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13)),
                                iterations=1,
                            )
                            if (
                                center_chest_strand_removal_mask is not None
                                and center_chest_strand_removal_mask.shape == final_bgr.shape[:2]
                            ):
                                center_direct_cleanup_u8 = cv2.dilate(
                                    (
                                        np.clip(center_chest_strand_removal_mask.astype(np.float32), 0.0, 1.0) > 0.05
                                    ).astype(np.uint8)
                                    * 255,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 73)),
                                    iterations=1,
                                )
                                center_direct_cleanup_u8 = cv2.bitwise_and(
                                    center_direct_cleanup_u8,
                                    torso_gate_u8,
                                )
                                female_short_direct_cloth_restore_u8 = cv2.bitwise_or(
                                    female_short_direct_cloth_restore_u8,
                                    center_direct_cleanup_u8,
                                )
                            female_short_direct_cloth_restore_px = int(
                                (female_short_direct_cloth_restore_u8 > 0).sum()
                            )
                            female_short_plain_cloth_cleanup_mask = None
                            female_short_broad_cloth_restore_mask = None
                            direct_side_restore_mask = None
                            if int((direct_female_short_restore_u8 > 0).sum()) >= 48:
                                direct_side_restore_mask = cv2.GaussianBlur(
                                    direct_female_short_restore_u8.astype(np.float32) / 255.0,
                                    (0, 0),
                                    sigmaX=3.6,
                                    sigmaY=5.4,
                                ).astype(np.float32)
                            if (
                                cloth_mask_dilated is not None
                                and cloth_mask_dilated.shape == final_bgr.shape[:2]
                                and int((female_short_direct_cloth_restore_u8 > 0).sum()) >= 80
                            ):
                                plain_cloth_u8 = cv2.dilate(
                                    (
                                        np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0) > 0.04
                                    ).astype(np.uint8)
                                    * 255,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
                                    iterations=1,
                                )
                                plain_cleanup_support_u8 = cv2.dilate(
                                    female_short_direct_cloth_restore_u8,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 55)),
                                    iterations=1,
                                )
                                source_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                                current_gray = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                                current_blur = cv2.GaussianBlur(current_gray, (0, 0), sigmaX=4.4, sigmaY=4.4)
                                gray_delta = np.abs(current_gray - source_gray)
                                current_sat = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
                                source_sat = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
                                diff_rgb = np.abs(
                                    final_rgb.astype(np.float32) - img_rgb.astype(np.float32)
                                ).mean(axis=2)
                                female_short_plain_cloth_cleanup_u8 = (
                                    (
                                        (
                                            (diff_rgb > 9.5)
                                            | (gray_delta > 8.0)
                                            | (current_gray + 8.0 < source_gray)
                                            | (current_gray > source_gray + 11.0)
                                            | (current_sat > source_sat + 10.0)
                                        )
                                        & (np.abs(current_gray - current_blur) < 13.0)
                                        & (current_sat < 144.0)
                                    ).astype(np.uint8)
                                    * 255
                                )
                                female_short_plain_cloth_cleanup_u8 = cv2.bitwise_and(
                                    female_short_plain_cloth_cleanup_u8,
                                    plain_cloth_u8,
                                )
                                female_short_plain_cloth_cleanup_u8 = cv2.bitwise_and(
                                    female_short_plain_cloth_cleanup_u8,
                                    plain_cleanup_support_u8,
                                )
                                female_short_plain_cloth_cleanup_u8 = cv2.bitwise_and(
                                    female_short_plain_cloth_cleanup_u8,
                                    torso_gate_u8,
                                )
                                if final_hair_mask is not None and final_hair_mask.shape == final_bgr.shape[:2]:
                                    final_hair_guard_u8 = cv2.dilate(
                                        (
                                            np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.16
                                        ).astype(np.uint8)
                                        * 255,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
                                        iterations=1,
                                    )
                                    female_short_plain_cloth_cleanup_u8 = cv2.bitwise_and(
                                        female_short_plain_cloth_cleanup_u8,
                                        cv2.bitwise_not(final_hair_guard_u8),
                                    )
                                female_short_plain_cloth_cleanup_u8 = cv2.morphologyEx(
                                    female_short_plain_cloth_cleanup_u8,
                                    cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 23)),
                                )
                                female_short_plain_cloth_cleanup_u8 = cv2.morphologyEx(
                                    female_short_plain_cloth_cleanup_u8,
                                    cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                )
                                if int((female_short_plain_cloth_cleanup_u8 > 0).sum()) >= 120:
                                    female_short_plain_cloth_cleanup_mask = cv2.GaussianBlur(
                                        female_short_plain_cloth_cleanup_u8.astype(np.float32) / 255.0,
                                        (0, 0),
                                        sigmaX=5.2,
                                        sigmaY=7.6,
                                    ).astype(np.float32)
                            if source_gray is None or current_gray is None or source_sat is None or current_sat is None or diff_rgb is None:
                                source_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                                current_gray = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                                source_sat = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
                                current_sat = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
                                diff_rgb = np.abs(
                                    final_rgb.astype(np.float32) - img_rgb.astype(np.float32)
                                ).mean(axis=2)
                            broad_restore_gate_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                            broad_top = max(0, int(y2 + face_h * 0.08))
                            broad_bottom = min(H, int(y2 + face_h * 2.05))
                            broad_left = max(0, int(cx - face_w * 1.95))
                            broad_right = min(W, int(cx + face_w * 1.95))
                            if broad_top < broad_bottom and broad_left < broad_right:
                                broad_restore_gate_u8[broad_top:broad_bottom, broad_left:broad_right] = 255
                            expanded_cloth_gate_u8 = None
                            if (
                                cloth_mask_dilated is not None
                                and cloth_mask_dilated.shape == final_bgr.shape[:2]
                                and float(cloth_mask_dilated.sum()) > 0.0
                            ):
                                expanded_cloth_gate_u8 = cv2.dilate(
                                    (
                                        np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0) > 0.04
                                    ).astype(np.uint8)
                                    * 255,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 131)),
                                    iterations=1,
                                )
                                broad_restore_gate_u8 = cv2.bitwise_or(
                                    broad_restore_gate_u8,
                                    cv2.bitwise_and(expanded_cloth_gate_u8, torso_gate_u8),
                                )
                            broad_source_cloth_support_u8 = broad_restore_gate_u8.copy()
                            if (
                                cloth_mask_dilated is not None
                                and cloth_mask_dilated.shape == final_bgr.shape[:2]
                                and float(cloth_mask_dilated.sum()) > 0.0
                            ):
                                cloth_seed_u8 = (
                                    (
                                        np.clip(cloth_mask_dilated.astype(np.float32), 0.0, 1.0) > 0.04
                                    ).astype(np.uint8)
                                    * 255
                                )
                                cloth_seed_bool = cloth_seed_u8 > 0
                                if int(cloth_seed_bool.sum()) >= 60:
                                    seed_gray_med = float(np.median(source_gray[cloth_seed_bool]))
                                    seed_sat_med = float(np.median(source_sat[cloth_seed_bool]))
                                    broad_source_cloth_support_u8 = (
                                        (
                                            (
                                                source_gray > max(78.0, seed_gray_med - 44.0)
                                            )
                                            & (
                                                source_sat < min(176.0, seed_sat_med + 48.0)
                                            )
                                        ).astype(np.uint8)
                                        * 255
                                    )
                                    broad_source_cloth_support_u8 = cv2.bitwise_and(
                                        broad_source_cloth_support_u8,
                                        broad_restore_gate_u8,
                                    )
                                    if expanded_cloth_gate_u8 is not None:
                                        broad_source_cloth_support_u8 = cv2.bitwise_and(
                                            broad_source_cloth_support_u8,
                                            expanded_cloth_gate_u8,
                                        )
                                    broad_source_cloth_support_u8 = cv2.morphologyEx(
                                        broad_source_cloth_support_u8,
                                        cv2.MORPH_CLOSE,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 27)),
                                    )
                                    broad_source_cloth_support_u8 = cv2.dilate(
                                        broad_source_cloth_support_u8,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13)),
                                        iterations=1,
                                    )
                            broad_changed_u8 = (
                                (
                                    (
                                        (diff_rgb > 7.5)
                                        | (np.abs(current_gray - source_gray) > 7.0)
                                        | (current_gray + 7.0 < source_gray)
                                        | (current_gray > source_gray + 9.0)
                                        | (current_sat > source_sat + 8.0)
                                    )
                                    & (
                                        (current_sat < 182.0)
                                        | (current_gray + 3.0 < source_gray)
                                        | (source_gray > 120.0)
                                    )
                                ).astype(np.uint8)
                                * 255
                            )
                            broad_changed_u8 = cv2.bitwise_and(
                                broad_changed_u8,
                                broad_restore_gate_u8,
                            )
                            broad_changed_u8 = cv2.bitwise_and(
                                broad_changed_u8,
                                broad_source_cloth_support_u8,
                            )
                            broad_changed_u8 = cv2.bitwise_or(
                                broad_changed_u8,
                                cv2.bitwise_and(
                                    female_short_direct_cloth_restore_u8,
                                    broad_source_cloth_support_u8,
                                ),
                            )
                            broad_changed_u8 = cv2.morphologyEx(
                                broad_changed_u8,
                                cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 35)),
                            )
                            broad_changed_u8 = cv2.dilate(
                                broad_changed_u8,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 25)),
                                iterations=1,
                            )
                            if int((broad_changed_u8 > 0).sum()) >= 240:
                                female_short_broad_cloth_restore_mask = cv2.GaussianBlur(
                                    broad_changed_u8.astype(np.float32) / 255.0,
                                    (0, 0),
                                    sigmaX=6.4,
                                    sigmaY=9.2,
                                ).astype(np.float32)
                                if (
                                    female_short_direct_cloth_restore_px >= 120
                                    or female_short_broad_cloth_restore_mask is not None
                                ):
                                    if female_short_direct_cloth_restore_px >= 120:
                                        x1, y1, x2, y2 = face_bbox
                                        face_w = max(int(x2 - x1), 1)
                                        face_h = max(int(y2 - y1), 1)
                                        cx = int(0.5 * (x1 + x2))
                                        if (
                                            source_gray is None
                                            or current_gray is None
                                            or source_sat is None
                                            or current_sat is None
                                            or diff_rgb is None
                                        ):
                                            source_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                                            current_gray = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                                            source_sat = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
                                            current_sat = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
                                            diff_rgb = np.abs(
                                                final_rgb.astype(np.float32) - img_rgb.astype(np.float32)
                                            ).mean(axis=2)

                                        source_visible_gate_u8 = cloth_gate_u8.copy()
                                        if (
                                            subject_cloth_anchor_for_post is not None
                                            and subject_cloth_anchor_for_post.shape == final_bgr.shape[:2]
                                        ):
                                            source_visible_gate_u8 = cv2.bitwise_or(
                                                source_visible_gate_u8,
                                                cv2.dilate(
                                                    (
                                                        np.clip(
                                                            subject_cloth_anchor_for_post.astype(np.float32),
                                                            0.0,
                                                            1.0,
                                                        )
                                                        > 0.04
                                                    ).astype(np.uint8)
                                                    * 255,
                                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
                                                    iterations=1,
                                                ),
                                            )
                                        source_visible_gate_u8 = cv2.bitwise_and(source_visible_gate_u8, torso_gate_u8)
                                        source_lap = np.abs(
                                            cv2.Laplacian(source_gray.astype(np.uint8), cv2.CV_32F, ksize=3)
                                        )
                                        source_visible_cloth_u8 = (
                                            (
                                                (source_gray > 170.0)
                                                & (source_sat < 92.0)
                                                & (source_lap < 30.0)
                                            ).astype(np.uint8)
                                            * 255
                                        )
                                        source_visible_cloth_u8 = cv2.bitwise_and(
                                            source_visible_cloth_u8,
                                            source_visible_gate_u8,
                                        )
                                        source_visible_cloth_u8 = cv2.morphologyEx(
                                            source_visible_cloth_u8,
                                            cv2.MORPH_CLOSE,
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 19)),
                                        )
                                        source_visible_cloth_u8 = cv2.dilate(
                                            source_visible_cloth_u8,
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
                                            iterations=1,
                                        )
                                        if int((source_visible_cloth_u8 > 0).sum()) >= 180:
                                            source_bridge_gate_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                                            source_bridge_half_w = max(26, int(face_w * 0.28))
                                            source_bridge_left = max(0, cx - source_bridge_half_w)
                                            source_bridge_right = min(W, cx + source_bridge_half_w)
                                            source_bridge_top = max(0, int(cutoff_y_for_post + face_h * 0.04))
                                            source_bridge_bottom = min(H, int(cutoff_y_for_post + face_h * 1.28))
                                            if (
                                                source_bridge_top < source_bridge_bottom
                                                and source_bridge_left < source_bridge_right
                                            ):
                                                source_bridge_gate_u8[
                                                    source_bridge_top:source_bridge_bottom,
                                                    source_bridge_left:source_bridge_right,
                                                ] = 255

                                            source_cloth_envelope_u8 = cv2.morphologyEx(
                                                source_visible_cloth_u8,
                                                cv2.MORPH_CLOSE,
                                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 43)),
                                            )
                                            source_cloth_envelope_u8 = cv2.dilate(
                                                source_cloth_envelope_u8,
                                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 23)),
                                                iterations=1,
                                            )
                                            source_cloth_envelope_u8 = cv2.bitwise_or(
                                                source_cloth_envelope_u8,
                                                cv2.bitwise_and(source_bridge_gate_u8, source_visible_gate_u8),
                                            )
                                            source_cloth_envelope_u8 = cv2.bitwise_and(
                                                source_cloth_envelope_u8,
                                                source_visible_gate_u8,
                                            )
                                            source_cloth_hole_u8 = cv2.bitwise_and(
                                                source_cloth_envelope_u8,
                                                cv2.bitwise_not(source_visible_cloth_u8),
                                            )
                                            source_visible_cloth_mask = cv2.GaussianBlur(
                                                source_visible_cloth_u8.astype(np.float32) / 255.0,
                                                (0, 0),
                                                sigmaX=4.2,
                                                sigmaY=6.0,
                                            ).astype(np.float32)
                                            source_cloth_reference_mask = cv2.GaussianBlur(
                                                source_cloth_envelope_u8.astype(np.float32) / 255.0,
                                                (0, 0),
                                                sigmaX=4.4,
                                                sigmaY=6.2,
                                            ).astype(np.float32)
                                            source_cloth_reference_rgb = source_cloth_reference_rgb.copy()
                                            visible_pixels = img_rgb[source_visible_cloth_u8 > 0]
                                            if visible_pixels.size > 0 and int((source_cloth_hole_u8 > 0).sum()) >= 80:
                                                source_fill_color = np.median(visible_pixels, axis=0).astype(np.uint8)
                                                source_cloth_reference_rgb[source_cloth_hole_u8 > 0] = source_fill_color
                                                source_cloth_hole_mask = cv2.GaussianBlur(
                                                    source_cloth_hole_u8.astype(np.float32) / 255.0,
                                                    (0, 0),
                                                    sigmaX=3.8,
                                                    sigmaY=5.6,
                                                ).astype(np.float32)
                                                source_cloth_reference_rgb = self._blend_neighbor_cloth_tone(
                                                    source_cloth_reference_rgb,
                                                    source_cloth_hole_mask,
                                                    cloth_mask=source_visible_cloth_mask,
                                                    reference_rgb=img_rgb,
                                                )
                                                source_cloth_reference_rgb = self._cv2_refine_cloth_region(
                                                    source_cloth_reference_rgb,
                                                    source_cloth_hole_mask,
                                                    reference_rgb=img_rgb,
                                                    reference_mask=source_visible_cloth_mask,
                                                )
                                            female_short_cloth_reference_mask = np.maximum(
                                                female_short_cloth_reference_mask,
                                                np.clip(source_cloth_reference_mask * 0.98, 0.0, 1.0),
                                            )
                                            source_visible_rescue_u8 = (
                                                (
                                                    (diff_rgb > 10.0)
                                                    | (np.abs(current_gray - source_gray) > 9.0)
                                                    | (current_gray + 8.0 < source_gray)
                                                    | (current_gray > source_gray + 10.0)
                                                    | (current_sat > source_sat + 9.0)
                                                ).astype(np.uint8)
                                                * 255
                                            )
                                            source_visible_rescue_u8 = cv2.bitwise_and(
                                                source_visible_rescue_u8,
                                                source_cloth_envelope_u8,
                                            )
                                            if (
                                                removal_mask_for_post is not None
                                                and removal_mask_for_post.shape == final_bgr.shape[:2]
                                            ):
                                                source_visible_rescue_u8 = cv2.bitwise_and(
                                                    source_visible_rescue_u8,
                                                    cv2.dilate(
                                                        (
                                                            np.clip(removal_mask_for_post.astype(np.float32), 0.0, 1.0)
                                                            > 0.04
                                                        ).astype(np.uint8)
                                                        * 255,
                                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
                                                        iterations=1,
                                                    ),
                                                )
                                            if int((source_visible_rescue_u8 > 0).sum()) >= 120:
                                                source_visible_rescue_mask = cv2.GaussianBlur(
                                                    source_visible_rescue_u8.astype(np.float32) / 255.0,
                                                    (0, 0),
                                                    sigmaX=4.4,
                                                    sigmaY=6.4,
                                                ).astype(np.float32)
                                                final_rgb = self._restore_reference_region(
                                                    final_rgb,
                                                    source_cloth_reference_rgb,
                                                    source_visible_rescue_mask,
                                                    strength=0.98,
                                                )
                                                final_rgb = self._overlay_reference_cloth_fill(
                                                    final_rgb,
                                                    source_cloth_reference_rgb,
                                                    source_visible_rescue_mask,
                                                    cloth_mask=female_short_cloth_reference_mask,
                                                )
                                                final_rgb = self._cv2_refine_cloth_region(
                                                    final_rgb,
                                                    source_visible_rescue_mask,
                                                    reference_rgb=source_cloth_reference_rgb,
                                                    reference_mask=source_cloth_reference_mask,
                                                )

                                        center_fill_gate_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                                        center_fill_half_w = max(24, int(face_w * 0.24))
                                        center_fill_left = max(0, cx - center_fill_half_w)
                                        center_fill_right = min(W, cx + center_fill_half_w)
                                        center_fill_top = max(0, int(cutoff_y_for_post + face_h * 0.04))
                                        center_fill_bottom = min(H, int(cutoff_y_for_post + face_h * 1.34))
                                        if (
                                            center_fill_top < center_fill_bottom
                                            and center_fill_left < center_fill_right
                                        ):
                                            center_fill_gate_u8[
                                                center_fill_top:center_fill_bottom,
                                                center_fill_left:center_fill_right,
                                            ] = 255

                                        center_fill_u8 = cv2.bitwise_and(
                                            female_short_direct_cloth_restore_u8,
                                            center_fill_gate_u8,
                                        )
                                        if (
                                            center_chest_strand_removal_mask is not None
                                            and center_chest_strand_removal_mask.shape == final_bgr.shape[:2]
                                        ):
                                            center_support_u8 = cv2.dilate(
                                                (
                                                    np.clip(center_chest_strand_removal_mask.astype(np.float32), 0.0, 1.0)
                                                    > 0.05
                                                ).astype(np.uint8)
                                                * 255,
                                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 37)),
                                                iterations=1,
                                            )
                                            center_support_u8 = cv2.bitwise_and(center_support_u8, center_fill_gate_u8)
                                            center_fill_u8 = cv2.bitwise_or(center_fill_u8, center_support_u8)
                                        center_fill_u8 = cv2.bitwise_and(center_fill_u8, torso_gate_u8)
                                        center_fill_u8 = cv2.bitwise_and(center_fill_u8, cloth_gate_u8)
                                        center_fill_u8 = cv2.morphologyEx(
                                            center_fill_u8,
                                            cv2.MORPH_CLOSE,
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 17)),
                                        )

                                        side_center_keepout_u8 = cv2.dilate(
                                            center_fill_gate_u8,
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 41)),
                                            iterations=1,
                                        )
                                        side_restore_u8 = cv2.bitwise_and(
                                            female_short_direct_cloth_restore_u8,
                                            cv2.bitwise_not(side_center_keepout_u8),
                                        )
                                        if direct_side_restore_mask is not None:
                                            side_restore_extra_u8 = (
                                                (
                                                    np.clip(direct_side_restore_mask.astype(np.float32), 0.0, 1.0) > 0.06
                                                ).astype(np.uint8)
                                                * 255
                                            )
                                            side_restore_u8 = cv2.bitwise_or(side_restore_u8, side_restore_extra_u8)
                                        side_restore_u8 = cv2.bitwise_and(side_restore_u8, torso_gate_u8)
                                        side_restore_u8 = cv2.bitwise_and(side_restore_u8, cloth_gate_u8)
                                        side_restore_u8 = cv2.morphologyEx(
                                            side_restore_u8,
                                            cv2.MORPH_CLOSE,
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 19)),
                                        )
                                        side_restore_u8 = cv2.dilate(
                                            side_restore_u8,
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
                                            iterations=1,
                                        )

                                        side_restore_px = int((side_restore_u8 > 0).sum())
                                        center_fill_px = int((center_fill_u8 > 0).sum())
                                        if side_restore_px >= 80:
                                            side_restore_mask = cv2.GaussianBlur(
                                                side_restore_u8.astype(np.float32) / 255.0,
                                                (0, 0),
                                                sigmaX=3.8,
                                                sigmaY=5.8,
                                            ).astype(np.float32)
                                            final_rgb = self._restore_reference_region(
                                                final_rgb,
                                                source_cloth_reference_rgb,
                                                side_restore_mask,
                                                strength=0.97,
                                            )
                                            final_rgb = self._overlay_reference_cloth_fill(
                                                final_rgb,
                                                source_cloth_reference_rgb,
                                                side_restore_mask,
                                                cloth_mask=female_short_cloth_reference_mask,
                                            )
                                            final_rgb = self._cv2_refine_cloth_region(
                                                final_rgb,
                                                side_restore_mask,
                                                reference_rgb=source_cloth_reference_rgb,
                                                reference_mask=source_cloth_reference_mask,
                                            )
                                        if center_fill_px >= 60:
                                            center_fill_mask = cv2.GaussianBlur(
                                                center_fill_u8.astype(np.float32) / 255.0,
                                                (0, 0),
                                                sigmaX=3.2,
                                                sigmaY=5.2,
                                            ).astype(np.float32)
                                            final_rgb = self._cv2_cleanup_dark_tail_blob(final_rgb, center_fill_u8)
                                            final_rgb = self._blend_neighbor_cloth_tone(
                                                final_rgb,
                                                center_fill_mask,
                                                cloth_mask=female_short_cloth_reference_mask,
                                                reference_rgb=source_cloth_reference_rgb,
                                            )
                                            final_rgb = self._cv2_refine_cloth_region(
                                                final_rgb,
                                                center_fill_mask,
                                                reference_rgb=source_cloth_reference_rgb,
                                                reference_mask=source_cloth_reference_mask,
                                            )
                                    elif female_short_broad_cloth_restore_mask is not None:
                                        final_rgb = self._blend_neighbor_cloth_tone(
                                            final_rgb,
                                            female_short_broad_cloth_restore_mask,
                                            cloth_mask=female_short_cloth_reference_mask,
                                            reference_rgb=img_rgb,
                                        )
                                        final_rgb = self._cv2_refine_cloth_region(
                                            final_rgb,
                                            female_short_broad_cloth_restore_mask,
                                            reference_rgb=img_rgb,
                                            reference_mask=female_short_cloth_reference_mask,
                                        )
                                final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                                if debug_images_common is not None and rank == 0:
                                    if female_short_direct_cloth_restore_px >= 120:
                                        debug_images_common["pipeline_female_short_direct_cloth_restore_mask"] = cv2.cvtColor(
                                            female_short_direct_cloth_restore_u8,
                                            cv2.COLOR_GRAY2BGR,
                                        )
                                    if female_short_broad_cloth_restore_mask is not None:
                                        debug_images_common["pipeline_female_short_broad_cloth_restore_mask"] = cv2.cvtColor(
                                            broad_changed_u8,
                                            cv2.COLOR_GRAY2BGR,
                                        )
                    except Exception as e:
                        logger.warning(f"[SDPipeline] female short direct cloth restore failed (ignored): {e}")
            if (
                self.config.enable_post_cloth_refine
                and rank == 0
                and hair_length in ("short", "medium")
                and cloth_mask_dilated is not None
                and removal_mask_for_post is not None
                and cutoff_y_for_post is not None
            ):
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                    cloth_reference_mask = (
                        cloth_restore_mask_for_post
                        if cloth_restore_mask_for_post is not None and cloth_restore_mask_for_post.shape == final_bgr.shape[:2]
                        else cloth_mask_dilated
                    )
                    short_cloth_neck_preserve_mask = np.zeros(final_bgr.shape[:2], dtype=np.float32)
                    if hair_length == "short":
                        if (
                            neckline_preserve_for_post is not None
                            and neckline_preserve_for_post.shape == final_bgr.shape[:2]
                        ):
                            short_cloth_neck_preserve_mask = np.maximum(
                                short_cloth_neck_preserve_mask,
                                np.clip(neckline_preserve_for_post.astype(np.float32), 0.0, 1.0),
                            ).astype(np.float32)
                        if (
                            lateral_neck_preserve_for_post is not None
                            and lateral_neck_preserve_for_post.shape == final_bgr.shape[:2]
                        ):
                            short_cloth_neck_preserve_mask = np.maximum(
                                short_cloth_neck_preserve_mask,
                                np.clip(lateral_neck_preserve_for_post.astype(np.float32), 0.0, 1.0) * 0.92,
                            ).astype(np.float32)
                    cloth_only_protect_mask = protect_mask_for_sd
                    if hair_length == "short":
                        cloth_only_protect_mask = np.maximum(
                            np.clip(protect_mask_for_sd.astype(np.float32), 0.0, 1.0),
                            short_cloth_neck_preserve_mask,
                        ).astype(np.float32)
                    under_jaw_candidate_mask = np.maximum(
                        short_lower_garment_cleanup_mask_for_post,
                        center_residual_cleanup_mask_for_post,
                    ).astype(np.float32)
                    under_jaw_candidate_mask = np.maximum(
                        under_jaw_candidate_mask,
                        final_source_cloth_rescue_mask_for_post,
                    ).astype(np.float32)
                    under_jaw_candidate_mask = np.maximum(
                        under_jaw_candidate_mask,
                        residual_strand_cleanup_mask_for_post,
                    ).astype(np.float32)
                    under_jaw_cloth_refine_mask = self._build_under_jaw_cloth_refine_mask(
                        current_rgb=final_rgb,
                        source_rgb=img_rgb,
                        removal_mask=removal_mask_for_post,
                        cloth_mask=cloth_reference_mask,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        protect_mask=protect_mask_for_sd,
                        final_hair_mask=final_hair_mask,
                        candidate_mask=under_jaw_candidate_mask,
                        center_support_mask=center_chest_strand_removal_mask,
                        neck_preserve_mask=short_cloth_neck_preserve_mask if hair_length == "short" else None,
                    )
                    under_jaw_cloth_refine_u8 = (
                        (np.clip(under_jaw_cloth_refine_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8)
                        * 255
                    )
                    short_under_jaw_second_pass_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                    short_under_jaw_generation_silhouette_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                    short_under_jaw_second_pass_silhouette_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                    short_under_jaw_control_map_rgb = np.zeros_like(final_rgb)
                    short_under_jaw_second_pass_control_map_rgb = np.zeros_like(final_rgb)
                    short_under_jaw_crop_refine_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                    short_under_jaw_insert_u8 = np.zeros(final_bgr.shape[:2], dtype=np.uint8)
                    short_under_jaw_insert_control_map_rgb = np.zeros_like(final_rgb)
                    under_jaw_cloth_refine_px = int((under_jaw_cloth_refine_u8 > 0).sum())
                    if under_jaw_cloth_refine_px >= 140:
                        under_jaw_generation_mask = under_jaw_cloth_refine_mask
                        short_under_jaw_control_rgb = None
                        short_under_jaw_crop_refine_mask = under_jaw_cloth_refine_mask
                        short_under_jaw_crop_control_rgb = None
                        if hair_length == "short":
                            short_under_jaw_generation_silhouette_mask = self._build_short_cloth_generation_silhouette_mask(
                                current_rgb=final_rgb,
                                source_rgb=img_rgb,
                                cloth_mask=cloth_reference_mask,
                                face_bbox=face_bbox,
                                cutoff_y=cutoff_y_for_post,
                                protect_mask=protect_mask_for_sd,
                                final_hair_mask=final_hair_mask,
                                seed_mask=under_jaw_cloth_refine_mask,
                                neck_preserve_mask=short_cloth_neck_preserve_mask,
                            )
                            short_under_jaw_generation_silhouette_u8 = (
                                (
                                    np.clip(
                                        short_under_jaw_generation_silhouette_mask.astype(np.float32),
                                        0.0,
                                        1.0,
                                    ) > 0.08
                                ).astype(np.uint8)
                                * 255
                            )
                            if int((short_under_jaw_generation_silhouette_u8 > 0).sum()) >= 56:
                                under_jaw_generation_mask = short_under_jaw_generation_silhouette_mask
                            short_under_jaw_control_rgb = self._build_short_cloth_control_map(
                                current_rgb=final_rgb,
                                source_rgb=img_rgb,
                                cloth_mask=cloth_reference_mask,
                                face_bbox=face_bbox,
                                cutoff_y=cutoff_y_for_post,
                                seed_mask=under_jaw_generation_mask,
                                neck_preserve_mask=short_cloth_neck_preserve_mask,
                                protect_mask=cloth_only_protect_mask,
                            )
                            if short_under_jaw_control_rgb is not None:
                                short_under_jaw_control_map_rgb = short_under_jaw_control_rgb.copy()
                                short_under_jaw_crop_control_rgb = short_under_jaw_control_rgb
                        pre_under_jaw_rgb = final_rgb.copy()
                        under_jaw_base_rgb = final_rgb
                        if hair_length == "short":
                            under_jaw_base_rgb = self._build_source_conditioned_cloth_base(
                                current_rgb=final_rgb,
                                source_rgb=img_rgb,
                                fill_mask=under_jaw_generation_mask,
                                cloth_mask=cloth_reference_mask,
                                hair_length=hair_length,
                                preserve_mask=short_cloth_neck_preserve_mask,
                            )
                        generated_under_jaw_rgb = self._sd_refine_removed_region(
                            base_rgb=under_jaw_base_rgb,
                            removal_mask=under_jaw_generation_mask,
                            face_bbox=face_bbox,
                            face_crop_pil=face_crop_pil,
                            protect_mask=cloth_only_protect_mask,
                            cloth_mask=cloth_reference_mask,
                            hair_length=hair_length,
                            seed=int(cand["seed"]) + 1871,
                            reference_rgb=img_rgb,
                            refine_mode="under_jaw_cloth",
                            control_rgb=short_under_jaw_control_rgb,
                        )
                        final_rgb = generated_under_jaw_rgb
                        if hair_length == "short":
                            final_rgb = self._restore_reference_region(
                                pre_under_jaw_rgb,
                                generated_under_jaw_rgb,
                                under_jaw_generation_mask,
                                strength=0.996,
                            )
                        if hair_length == "short" and float(short_cloth_neck_preserve_mask.sum()) > 0.0:
                            final_rgb = self._restore_reference_region(
                                final_rgb,
                                img_rgb,
                                short_cloth_neck_preserve_mask,
                                strength=0.992,
                            )
                        final_rgb = self._blend_neighbor_cloth_tone(
                            final_rgb,
                            under_jaw_generation_mask,
                            cloth_mask=cloth_reference_mask,
                            reference_rgb=img_rgb,
                        )
                        final_rgb = self._cv2_refine_cloth_region(
                            final_rgb,
                            under_jaw_generation_mask,
                            reference_rgb=img_rgb,
                            reference_mask=cloth_reference_mask,
                        )
                        final_rgb = self._stabilize_under_jaw_cloth_fill(
                            final_rgb,
                            img_rgb,
                            under_jaw_generation_mask,
                            cloth_reference_mask,
                            hair_length=hair_length,
                        )
                        if hair_length == "short":
                            final_rgb = self._apply_short_source_cloth_anchor_restore(
                                current_rgb=final_rgb,
                                source_rgb=img_rgb,
                                fill_mask=under_jaw_generation_mask,
                                cloth_mask=cloth_reference_mask,
                                face_bbox=face_bbox,
                                cutoff_y=cutoff_y_for_post,
                                neck_preserve_mask=short_cloth_neck_preserve_mask,
                            )
                        if hair_length == "short":
                            final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                            short_under_jaw_second_pass_mask = self._build_short_cloth_only_second_pass_mask(
                                current_rgb=final_rgb,
                                source_rgb=img_rgb,
                                cloth_mask=cloth_reference_mask,
                                face_bbox=face_bbox,
                                cutoff_y=cutoff_y_for_post,
                                hair_length=hair_length,
                                protect_mask=protect_mask_for_sd,
                                final_hair_mask=final_hair_mask,
                                seed_mask=under_jaw_cloth_refine_mask,
                                neck_preserve_mask=short_cloth_neck_preserve_mask,
                            )
                            short_under_jaw_second_pass_u8 = (
                                (
                                    np.clip(short_under_jaw_second_pass_mask.astype(np.float32), 0.0, 1.0) > 0.08
                                ).astype(np.uint8)
                                * 255
                            )
                            short_under_jaw_second_pass_px = int((short_under_jaw_second_pass_u8 > 0).sum())
                            if short_under_jaw_second_pass_px >= 48:
                                short_under_jaw_second_pass_generation_mask = short_under_jaw_second_pass_mask
                                short_under_jaw_second_pass_control_rgb = None
                                short_under_jaw_second_pass_silhouette_mask = self._build_short_cloth_generation_silhouette_mask(
                                    current_rgb=final_rgb,
                                    source_rgb=img_rgb,
                                    cloth_mask=cloth_reference_mask,
                                    face_bbox=face_bbox,
                                    cutoff_y=cutoff_y_for_post,
                                    protect_mask=protect_mask_for_sd,
                                    final_hair_mask=final_hair_mask,
                                    seed_mask=short_under_jaw_second_pass_mask,
                                    neck_preserve_mask=short_cloth_neck_preserve_mask,
                                )
                                short_under_jaw_second_pass_silhouette_u8 = (
                                    (
                                        np.clip(
                                            short_under_jaw_second_pass_silhouette_mask.astype(np.float32),
                                            0.0,
                                            1.0,
                                        ) > 0.08
                                    ).astype(np.uint8)
                                    * 255
                                )
                                if int((short_under_jaw_second_pass_silhouette_u8 > 0).sum()) >= 28:
                                    short_under_jaw_second_pass_generation_mask = (
                                        short_under_jaw_second_pass_silhouette_mask
                                    )
                                short_under_jaw_second_pass_control_rgb = self._build_short_cloth_control_map(
                                    current_rgb=final_rgb,
                                    source_rgb=img_rgb,
                                    cloth_mask=cloth_reference_mask,
                                    face_bbox=face_bbox,
                                    cutoff_y=cutoff_y_for_post,
                                    seed_mask=short_under_jaw_second_pass_generation_mask,
                                    neck_preserve_mask=short_cloth_neck_preserve_mask,
                                    protect_mask=cloth_only_protect_mask,
                                )
                                if short_under_jaw_second_pass_control_rgb is not None:
                                    short_under_jaw_second_pass_control_map_rgb = (
                                        short_under_jaw_second_pass_control_rgb.copy()
                                    )
                                    short_under_jaw_crop_control_rgb = short_under_jaw_second_pass_control_rgb
                                short_under_jaw_crop_refine_mask = short_under_jaw_second_pass_generation_mask
                                pre_second_pass_rgb = final_rgb.copy()
                                short_under_jaw_second_pass_base_rgb = self._build_source_conditioned_cloth_base(
                                    current_rgb=final_rgb,
                                    source_rgb=img_rgb,
                                    fill_mask=short_under_jaw_second_pass_generation_mask,
                                    cloth_mask=cloth_reference_mask,
                                    hair_length=hair_length,
                                    preserve_mask=short_cloth_neck_preserve_mask,
                                )
                                generated_second_pass_rgb = self._sd_refine_removed_region(
                                    base_rgb=short_under_jaw_second_pass_base_rgb,
                                    removal_mask=short_under_jaw_second_pass_generation_mask,
                                    face_bbox=face_bbox,
                                    face_crop_pil=face_crop_pil,
                                    protect_mask=cloth_only_protect_mask,
                                    cloth_mask=cloth_reference_mask,
                                    hair_length=hair_length,
                                    seed=int(cand["seed"]) + 1889,
                                    reference_rgb=img_rgb,
                                    refine_mode="cloth_only_second_pass",
                                    control_rgb=short_under_jaw_second_pass_control_rgb,
                                )
                                final_rgb = self._restore_reference_region(
                                    pre_second_pass_rgb,
                                    generated_second_pass_rgb,
                                    short_under_jaw_second_pass_generation_mask,
                                    strength=0.997,
                                )
                                if float(short_cloth_neck_preserve_mask.sum()) > 0.0:
                                    final_rgb = self._restore_reference_region(
                                        final_rgb,
                                        img_rgb,
                                        short_cloth_neck_preserve_mask,
                                        strength=0.995,
                                    )
                                final_rgb = self._blend_neighbor_cloth_tone(
                                    final_rgb,
                                    short_under_jaw_second_pass_generation_mask,
                                    cloth_mask=cloth_reference_mask,
                                    reference_rgb=img_rgb,
                                )
                                final_rgb = self._cv2_refine_cloth_region(
                                    final_rgb,
                                    short_under_jaw_second_pass_generation_mask,
                                    reference_rgb=img_rgb,
                                    reference_mask=cloth_reference_mask,
                                )
                                final_rgb = self._stabilize_under_jaw_cloth_fill(
                                    final_rgb,
                                    img_rgb,
                                    short_under_jaw_second_pass_generation_mask,
                                    cloth_reference_mask,
                                    hair_length=hair_length,
                                )
                                final_rgb = self._apply_short_source_cloth_anchor_restore(
                                    current_rgb=final_rgb,
                                    source_rgb=img_rgb,
                                    fill_mask=short_under_jaw_second_pass_generation_mask,
                                    cloth_mask=cloth_reference_mask,
                                    face_bbox=face_bbox,
                                    cutoff_y=cutoff_y_for_post,
                                    neck_preserve_mask=short_cloth_neck_preserve_mask,
                                )
                        if hair_length == "short":
                            short_under_jaw_crop_refine_u8 = (
                                (
                                    np.clip(short_under_jaw_crop_refine_mask.astype(np.float32), 0.0, 1.0) > 0.08
                                ).astype(np.uint8)
                                * 255
                            )
                            if int((short_under_jaw_crop_refine_u8 > 0).sum()) >= 36:
                                final_rgb = self._refine_short_under_jaw_crop_region(
                                    current_rgb=final_rgb,
                                    source_rgb=img_rgb,
                                    fill_mask=short_under_jaw_crop_refine_mask,
                                    cloth_mask=cloth_reference_mask,
                                    protect_mask=cloth_only_protect_mask,
                                    face_bbox=face_bbox,
                                    cutoff_y=cutoff_y_for_post,
                                    face_crop_pil=face_crop_pil,
                                    seed=int(cand["seed"]) + 1907,
                                    control_rgb=short_under_jaw_crop_control_rgb,
                                    neck_preserve_mask=short_cloth_neck_preserve_mask,
                                )
                                short_under_jaw_insert_u8 = short_under_jaw_crop_refine_u8.copy()
                                if short_under_jaw_crop_control_rgb is not None:
                                    short_under_jaw_insert_control_map_rgb = short_under_jaw_crop_control_rgb.copy()
                                final_rgb = self._generate_short_under_jaw_cloth_insert_region(
                                    current_rgb=final_rgb,
                                    source_rgb=img_rgb,
                                    fill_mask=short_under_jaw_crop_refine_mask,
                                    cloth_mask=cloth_reference_mask,
                                    protect_mask=cloth_only_protect_mask,
                                    face_bbox=face_bbox,
                                    cutoff_y=cutoff_y_for_post,
                                    face_crop_pil=face_crop_pil,
                                    seed=int(cand["seed"]) + 1923,
                                    control_rgb=short_under_jaw_crop_control_rgb,
                                    neck_preserve_mask=short_cloth_neck_preserve_mask,
                                )
                        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_under_jaw_cloth_refine_mask"] = cv2.cvtColor(
                            under_jaw_cloth_refine_u8,
                            cv2.COLOR_GRAY2BGR,
                        )
                        if hair_length == "short" and float(short_cloth_neck_preserve_mask.sum()) > 0.0:
                            debug_images_common["pipeline_short_cloth_neck_preserve_mask"] = cv2.cvtColor(
                                (
                                    np.clip(short_cloth_neck_preserve_mask.astype(np.float32), 0.0, 1.0) > 0.05
                                ).astype(np.uint8)
                                * 255,
                                cv2.COLOR_GRAY2BGR,
                            )
                        if int((short_under_jaw_generation_silhouette_u8 > 0).sum()) >= 24:
                            debug_images_common["pipeline_short_under_jaw_generation_silhouette_mask"] = cv2.cvtColor(
                                short_under_jaw_generation_silhouette_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                        if int((short_under_jaw_control_map_rgb > 0).sum()) >= 48:
                            debug_images_common["pipeline_short_under_jaw_control_map"] = cv2.cvtColor(
                                short_under_jaw_control_map_rgb,
                                cv2.COLOR_RGB2BGR,
                            )
                        if int((short_under_jaw_second_pass_u8 > 0).sum()) >= 24:
                            debug_images_common["pipeline_short_under_jaw_second_pass_mask"] = cv2.cvtColor(
                                short_under_jaw_second_pass_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                        if int((short_under_jaw_second_pass_silhouette_u8 > 0).sum()) >= 24:
                            debug_images_common["pipeline_short_under_jaw_second_pass_silhouette_mask"] = cv2.cvtColor(
                                short_under_jaw_second_pass_silhouette_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                        if int((short_under_jaw_second_pass_control_map_rgb > 0).sum()) >= 48:
                            debug_images_common["pipeline_short_under_jaw_second_pass_control_map"] = cv2.cvtColor(
                                short_under_jaw_second_pass_control_map_rgb,
                                cv2.COLOR_RGB2BGR,
                            )
                        if int((short_under_jaw_crop_refine_u8 > 0).sum()) >= 24:
                            debug_images_common["pipeline_short_under_jaw_crop_refine_mask"] = cv2.cvtColor(
                                short_under_jaw_crop_refine_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                        if int((short_under_jaw_insert_u8 > 0).sum()) >= 24:
                            debug_images_common["pipeline_short_under_jaw_insert_mask"] = cv2.cvtColor(
                                short_under_jaw_insert_u8,
                                cv2.COLOR_GRAY2BGR,
                            )
                        if int((short_under_jaw_insert_control_map_rgb > 0).sum()) >= 48:
                            debug_images_common["pipeline_short_under_jaw_insert_control_map"] = cv2.cvtColor(
                                short_under_jaw_insert_control_map_rgb,
                                cv2.COLOR_RGB2BGR,
                            )
                except Exception as e:
                    logger.warning(f"[SDPipeline] under-jaw cloth refine failed (ignored): {e}")
            try:
                final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
                eye_restore_mask = self._build_eye_region_restore_mask(
                    landmark_debug_data=landmark_debug_data,
                    image_shape=(H, W),
                    face_bbox=face_bbox,
                    hair_length=hair_length,
                    final_hair_mask=final_hair_mask,
                )
                if (
                    float(eye_restore_mask.sum()) <= 0.0
                    and hair_length == "long"
                ):
                    eye_restore_mask = self._build_face_eye_band_restore_mask(
                        face_mask=landmark_face_mask,
                        image_shape=(H, W),
                        face_bbox=face_bbox,
                    )
                if float(eye_restore_mask.sum()) > 0.0:
                    final_rgb = self._restore_reference_region(
                        final_rgb,
                        img_rgb,
                        eye_restore_mask,
                        strength=0.96 if hair_length == "long" else 0.92,
                    )
                    final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    if debug_images_common is not None and rank == 0:
                        debug_images_common["pipeline_eye_region_restore_mask"] = cv2.cvtColor(
                            ((np.clip(eye_restore_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255),
                            cv2.COLOR_GRAY2BGR,
                        )
            except Exception as e:
                logger.warning(f"[SDPipeline] eye region restore failed (ignored): {e}")
            results.append(SDInpaintResult(
                image=final_bgr,
                image_pil=Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)),
                seed=cand["seed"],
                rank=rank,
                mask_used=mask_source,
                mask_refine_mode=mask_refine_mode_used,
                clip_score=float(cand["color_score"]),
                mask=hair_mask_for_sd,
                face_bbox=face_bbox,
                debug_images=debug_images_common if (debug_images_common is not None and rank == 0) else None,
                debug_data=debug_data_common if (debug_data_common is not None and rank == 0) else None,
            ))

        return results
    def _maybe_standardize_input_portrait(
        self,
        img_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> Dict[str, Any]:
        """
        얼굴이 너무 작게 잡히는 입력을 미용실 상담용 포트레이트 프레임으로 정규화한다.
        작은 얼굴/넓은 배경/과도한 상반신 정보가 포함된 이미지만 보수적으로 크롭한다.
        """
        meta: Dict[str, Any] = {"applied": False}
        if not getattr(self.config, "enable_input_standardization", True):
            return meta

        H, W = img_rgb.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        face_w = max(x2 - x1, 1)
        face_h = max(y2 - y1, 1)
        face_w_ratio = face_w / max(float(W), 1.0)
        face_h_ratio = face_h / max(float(H), 1.0)
        top_gap_ratio = y1 / max(float(H), 1.0)
        is_landscape = W > H
        enable_reframe = bool(getattr(self.config, "enable_portrait_reframe", True))
        needs_reframe = (
            enable_reframe
            and not is_landscape
            and (
                face_h_ratio > float(getattr(self.config, "portrait_reframe_face_height_ratio_max", 0.40))
                or top_gap_ratio < float(getattr(self.config, "portrait_reframe_top_gap_ratio_min", 0.06))
            )
        )

        if (
            face_h_ratio >= float(self.config.standardize_face_height_ratio_min)
            and face_w_ratio >= float(self.config.standardize_face_width_ratio_min)
            and not is_landscape
            and not needs_reframe
        ):
            return meta

        if needs_reframe:
            target_w = int(W)
            target_h = int(H)
        else:
            target_w = int(getattr(self.config, "standardized_width", 768))
            target_h = int(getattr(self.config, "standardized_height", 1024))
            crop_top = int(round(y1 - face_h * 0.95))
            crop_bottom = int(round(y2 + face_h * 2.15))
        target_aspect = target_w / max(float(target_h), 1.0)

        cx = 0.5 * (x1 + x2)
        if needs_reframe:
            scale_candidates = [1.0]
            max_face_ratio = float(getattr(self.config, "portrait_reframe_face_height_ratio_max", 0.40))
            min_top_gap = float(getattr(self.config, "portrait_reframe_top_gap_ratio_min", 0.06))
            if max_face_ratio > 0.0:
                scale_candidates.append(face_h_ratio / max_face_ratio)
            if min_top_gap > 0.0 and top_gap_ratio > 1e-6:
                scale_candidates.append(min_top_gap / top_gap_ratio)
            reframe_scale = float(np.clip(max(scale_candidates) * 1.04, 1.02, 1.18))
            crop_h = max(int(round(H * reframe_scale)), face_h + 1)
            crop_w = max(int(round(crop_h * target_aspect)), W + 1)
            extra_h = max(0, crop_h - H)
            crop_top = int(round(-extra_h * 0.58))
            crop_bottom = crop_top + crop_h
        else:
            crop_h = max(crop_bottom - crop_top, face_h + 1)
            crop_w = max(int(round(crop_h * target_aspect)), face_w + 1)
        crop_left = int(round(cx - crop_w * 0.5))
        crop_right = crop_left + crop_w
        crop_rgb = self._crop_with_soft_padding(
            img_rgb,
            crop_left=crop_left,
            crop_top=crop_top,
            crop_right=crop_right,
            crop_bottom=crop_bottom,
        )
        if crop_rgb.size == 0:
            return meta

        interp = cv2.INTER_AREA
        if crop_rgb.shape[0] < target_h or crop_rgb.shape[1] < target_w:
            interp = cv2.INTER_CUBIC
        standardized_rgb = cv2.resize(crop_rgb, (target_w, target_h), interpolation=interp)

        meta.update({
            "applied": True,
            "reason": {
                "face_h_ratio": round(face_h_ratio, 4),
                "face_w_ratio": round(face_w_ratio, 4),
                "top_gap_ratio": round(top_gap_ratio, 4),
                "is_landscape": bool(is_landscape),
                "reframe_applied": bool(needs_reframe),
            },
            "crop_box": [int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)],
            "original_shape": [int(H), int(W)],
            "standardized_shape": [int(target_h), int(target_w)],
            "image_rgb": standardized_rgb,
        })
        return meta

    @staticmethod
    def _crop_with_soft_padding(
        img_rgb: np.ndarray,
        crop_left: int,
        crop_top: int,
        crop_right: int,
        crop_bottom: int,
    ) -> np.ndarray:
        H, W = img_rgb.shape[:2]
        pad_left = max(0, -int(crop_left))
        pad_top = max(0, -int(crop_top))
        pad_right = max(0, int(crop_right) - W)
        pad_bottom = max(0, int(crop_bottom) - H)

        padded = img_rgb
        if pad_left or pad_top or pad_right or pad_bottom:
            padded = cv2.copyMakeBorder(
                img_rgb,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REFLECT_101,
            )
            pad_mask = np.zeros(padded.shape[:2], dtype=np.float32)
            if pad_top:
                pad_mask[:pad_top, :] = 1.0
            if pad_bottom:
                pad_mask[-pad_bottom:, :] = 1.0
            if pad_left:
                pad_mask[:, :pad_left] = 1.0
            if pad_right:
                pad_mask[:, -pad_right:] = 1.0
            pad_mask = cv2.GaussianBlur(pad_mask, (0, 0), sigmaX=7.0, sigmaY=7.0)
            blurred = cv2.GaussianBlur(padded, (0, 0), sigmaX=18.0, sigmaY=18.0)
            padded = (
                padded.astype(np.float32) * (1.0 - pad_mask[..., np.newaxis])
                + blurred.astype(np.float32) * pad_mask[..., np.newaxis]
            )
            padded = np.clip(padded, 0, 255).astype(np.uint8)

        x1 = int(crop_left) + pad_left
        y1 = int(crop_top) + pad_top
        x2 = int(crop_right) + pad_left
        y2 = int(crop_bottom) + pad_top
        return padded[y1:y2, x1:x2]

    def _detect_face(
        self, img_rgb: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """MediaPipe로 얼굴 bbox (x1, y1, x2, y2) 반환"""
        H, W = img_rgb.shape[:2]
        result = self._mp_face.process(img_rgb)
        if not result.detections:
            return None
        bb = result.detections[0].location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * W))
        y1 = max(0, int(bb.ymin * H))
        x2 = min(W, int((bb.xmin + bb.width) * W))
        y2 = min(H, int((bb.ymin + bb.height) * H))
        return (x1, y1, x2, y2)

    def _detect_face_mesh(
        self, img_rgb: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        MediaPipe FaceMesh 랜드마크 검출.

        Returns:
            landmarks_norm: (N, 3) float32, normalized [0,1] 좌표
            landmarks_px:   (N, 2) int32, 원본 픽셀 좌표
        """
        H, W = img_rgb.shape[:2]
        if self._mp_face_mesh is None:
            return None, None

        result = self._mp_face_mesh.process(img_rgb)
        if not result.multi_face_landmarks:
            return None, None

        lms = result.multi_face_landmarks[0].landmark
        if not lms:
            return None, None

        landmarks_norm = np.asarray([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
        xs = np.clip(np.round(landmarks_norm[:, 0] * W), 0, W - 1).astype(np.int32)
        ys = np.clip(np.round(landmarks_norm[:, 1] * H), 0, H - 1).astype(np.int32)
        landmarks_px = np.stack([xs, ys], axis=1)
        return landmarks_norm, landmarks_px

    @staticmethod
    def _build_landmark_hull_mask(
        points_px: np.ndarray,
        shape: Tuple[int, int],
        *,
        dilate_px: int = 0,
    ) -> np.ndarray:
        H, W = shape
        mask = np.zeros((H, W), dtype=np.uint8)
        if points_px is None:
            return mask.astype(np.float32)

        pts = np.asarray(points_px, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return mask.astype(np.float32)

        valid = (
            (pts[:, 0] >= 0) & (pts[:, 0] < W) &
            (pts[:, 1] >= 0) & (pts[:, 1] < H)
        )
        pts = pts[valid]
        if len(pts) < 3:
            return mask.astype(np.float32)

        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, hull, 255)
        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
            mask = cv2.dilate(mask, k, iterations=1)
        return (mask > 0).astype(np.float32)

    def _build_mediapipe_face_oval_mask(
        self,
        landmarks_px: np.ndarray,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        import mediapipe as mp

        oval_idxs = sorted(
            {i for edge in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in edge}
        )
        if not oval_idxs:
            return np.zeros(shape, dtype=np.float32)
        oval_pts = np.asarray(
            [landmarks_px[i] for i in oval_idxs if i < len(landmarks_px)],
            dtype=np.int32,
        )
        return self._build_landmark_hull_mask(
            oval_pts,
            shape,
            dilate_px=5,
        )

    def _detect_landmark_data(
        self,
        img_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> Dict[str, Any]:
        mesh_norm, mesh_px = self._detect_face_mesh(img_rgb)
        if mesh_norm is None or mesh_px is None:
            return {"detected": False}

        face_mask = self._build_mediapipe_face_oval_mask(
            mesh_px,
            img_rgb.shape[:2],
        )
        debug_data = self._build_face_mesh_analysis(mesh_norm, mesh_px)
        debug_data["detected"] = True
        debug_data["face_mask_ratio"] = float(self._mask_ratio(face_mask))
        return {
            "detected": True,
            "landmarks_count": int(len(mesh_norm)),
            "face_mask": face_mask,
            "debug_images": self._render_face_mesh_debug_images(img_rgb, mesh_px),
            "debug_data": debug_data,
        }

    @staticmethod
    def _mask_bbox(
        mask: Optional[np.ndarray],
        *,
        threshold: float = 0.5,
    ) -> Optional[Tuple[int, int, int, int]]:
        if mask is None or mask.ndim != 2:
            return None
        ys, xs = np.where(mask > threshold)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )

    def _estimate_head_generation_box(
        self,
        *,
        image_shape: Tuple[int, int],
        face_bbox: Tuple[int, int, int, int],
        landmark_face_mask: Optional[np.ndarray],
        landmark_debug_data: Optional[Dict[str, Any]],
        hair_length: str,
    ) -> Tuple[int, int, int, int, int]:
        H, W = image_shape
        x1f, y1f, x2f, y2f = [int(v) for v in face_bbox]
        ref_x1, ref_y1, ref_x2, ref_y2 = x1f, y1f, x2f, y2f

        mask_bbox = self._mask_bbox(landmark_face_mask, threshold=0.20)
        if mask_bbox is not None:
            ref_x1, ref_y1, ref_x2, ref_y2 = mask_bbox

        keypoints = landmark_debug_data.get("keypoints", {}) if isinstance(landmark_debug_data, dict) else {}
        forehead_top = keypoints.get("forehead_top", {}).get("px")
        chin = keypoints.get("chin", {}).get("px")
        forehead_y = int(forehead_top[1]) if isinstance(forehead_top, list) and len(forehead_top) >= 2 else ref_y1
        chin_y = int(chin[1]) if isinstance(chin, list) and len(chin) >= 2 else ref_y2

        ref_w = max(int(ref_x2 - ref_x1), 1)
        ref_h = max(int(max(ref_y2, chin_y) - min(ref_y1, forehead_y)), 1)

        if hair_length == "short":
            margin_x = int(ref_w * 0.42)
            top_pad = int(ref_h * 0.72)
            cutoff_y = int(chin_y + ref_h * 0.10)
        else:
            margin_x = int(ref_w * 0.50)
            top_pad = int(ref_h * 0.64)
            cutoff_y = int(chin_y + ref_h * 0.46)

        head_x1 = max(0, ref_x1 - margin_x)
        head_x2 = min(W, ref_x2 + margin_x)
        head_y1 = max(0, min(ref_y1, forehead_y) - top_pad)
        cutoff_y = int(np.clip(cutoff_y, 0, H - 1))
        head_y2 = max(head_y1 + 1, cutoff_y)
        return head_x1, head_y1, head_x2, head_y2, cutoff_y

    @staticmethod
    def _build_face_mesh_analysis(
        landmarks_norm: np.ndarray,
        landmarks_px: np.ndarray,
    ) -> Dict[str, Any]:
        """
        얼굴형 분석용 FaceMesh 메타데이터 생성.
        """
        n = int(landmarks_norm.shape[0])

        def _safe_dist(i: int, j: int) -> Optional[float]:
            if i >= n or j >= n:
                return None
            p = landmarks_px[i].astype(np.float32)
            q = landmarks_px[j].astype(np.float32)
            return float(np.linalg.norm(p - q))

        face_height = _safe_dist(10, 152)    # forehead(top) ~ chin
        cheekbone_width = _safe_dist(234, 454)
        jaw_width = _safe_dist(172, 397)
        temple_width = _safe_dist(127, 356)

        ratios: Dict[str, Optional[float]] = {
            "cheekbone_to_height": None,
            "jaw_to_height": None,
            "temple_to_height": None,
            "jaw_to_cheekbone": None,
        }
        if face_height and face_height > 1e-6:
            if cheekbone_width is not None:
                ratios["cheekbone_to_height"] = cheekbone_width / face_height
            if jaw_width is not None:
                ratios["jaw_to_height"] = jaw_width / face_height
            if temple_width is not None:
                ratios["temple_to_height"] = temple_width / face_height
        if cheekbone_width and cheekbone_width > 1e-6 and jaw_width is not None:
            ratios["jaw_to_cheekbone"] = jaw_width / cheekbone_width

        keypoints: Dict[str, Any] = {}
        keypoint_map = {
            "forehead_top": 10,
            "chin": 152,
            "left_cheekbone": 234,
            "right_cheekbone": 454,
            "left_jaw": 172,
            "right_jaw": 397,
            "left_temple": 127,
            "right_temple": 356,
        }
        for name, idx in keypoint_map.items():
            if idx < n:
                keypoints[name] = {
                    "index": idx,
                    "norm": [
                        float(landmarks_norm[idx, 0]),
                        float(landmarks_norm[idx, 1]),
                        float(landmarks_norm[idx, 2]),
                    ],
                    "px": [int(landmarks_px[idx, 0]), int(landmarks_px[idx, 1])],
                }

        return {
            "landmarks_count": n,
            "landmarks_norm": landmarks_norm.astype(float).round(6).tolist(),
            "landmarks_px": landmarks_px.astype(int).tolist(),
            "metrics_px": {
                "face_height": face_height,
                "cheekbone_width": cheekbone_width,
                "jaw_width": jaw_width,
                "temple_width": temple_width,
            },
            "ratios": ratios,
            "keypoints": keypoints,
        }

    def _render_face_mesh_debug_images(
        self,
        img_rgb: np.ndarray,
        landmarks_px: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        FaceMesh 디버그 이미지 생성 (BGR).
        """
        import mediapipe as mp

        H, W = img_rgb.shape[:2]
        base_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        points_bgr = base_bgr.copy()
        tess_bgr = base_bgr.copy()
        contour_bgr = base_bgr.copy()

        # points
        for x, y in landmarks_px:
            cv2.circle(points_bgr, (int(x), int(y)), 1, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

        # tessellation
        for a, b in mp.solutions.face_mesh.FACEMESH_TESSELATION:
            if a >= len(landmarks_px) or b >= len(landmarks_px):
                continue
            p1 = tuple(int(v) for v in landmarks_px[a])
            p2 = tuple(int(v) for v in landmarks_px[b])
            cv2.line(tess_bgr, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)

        # contours
        for a, b in mp.solutions.face_mesh.FACEMESH_CONTOURS:
            if a >= len(landmarks_px) or b >= len(landmarks_px):
                continue
            p1 = tuple(int(v) for v in landmarks_px[a])
            p2 = tuple(int(v) for v in landmarks_px[b])
            cv2.line(contour_bgr, p1, p2, (255, 255, 0), 1, cv2.LINE_AA)

        # face oval mask
        oval_idxs = sorted(
            {i for edge in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in edge}
        )
        oval_mask = np.zeros((H, W), dtype=np.uint8)
        if oval_idxs:
            pts = np.asarray([landmarks_px[i] for i in oval_idxs if i < len(landmarks_px)], dtype=np.int32)
            if len(pts) >= 3:
                hull = cv2.convexHull(pts.reshape(-1, 1, 2))
                cv2.fillConvexPoly(oval_mask, hull, 255)
        oval_mask_bgr = cv2.cvtColor(oval_mask, cv2.COLOR_GRAY2BGR)

        return {
            "mediapipe_face_mesh_points": points_bgr,
            "mediapipe_face_mesh_tessellation": tess_bgr,
            "mediapipe_face_mesh_contours": contour_bgr,
            "mediapipe_face_mesh_oval_mask": oval_mask_bgr,
        }

    def _segface_hair_mask(self, img_rgb: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SegFace로 머리카락 + 얼굴 + 옷 영역 마스크 생성.
        얼굴 bbox를 기준으로 여유 있게 크롭한 뒤 512x512로 리사이즈하여 SegFace에 입력.
        이후 원본 해상도의 전체 영역으로 다시 복원하여 출력.

        Returns:
            hair_mask  (H×W float32): 머리카락 영역
            face_mask  (H×W float32): 얼굴/목/귀 영역 (inpaint에서 보호)
            cloth_mask (H×W float32): 옷 영역 (inpaint에서 보호)
        """
        H, W = img_rgb.shape[:2]
        x1, y1, x2, y2 = face_bbox
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw // 2, y1 + bh // 2
        
        crop_scale = float(os.environ.get("SEGFACE_CROP_SCALE", "3.2"))
        crop_center_y_offset = float(os.environ.get("SEGFACE_CROP_CENTER_Y_OFFSET", "0.12"))
        box_size = int(max(bw, bh) * crop_scale)
        cy = max(0, cy - int(box_size * crop_center_y_offset))
        
        crop_x1 = max(0, cx - box_size // 2)
        crop_y1 = max(0, cy - box_size // 2)
        crop_x2 = min(W, crop_x1 + box_size)
        crop_y2 = min(H, crop_y1 + box_size)
        
        cw = crop_x2 - crop_x1
        ch = crop_y2 - crop_y1
        
        # 정사각형 형태로 패딩해서 512x512 로 만들기 위한 준비
        crop_max = max(cw, ch)
        pad_bottom = crop_max - ch
        pad_right = crop_max - cw
        
        # 크롭
        crop_img = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        # 패딩 (검은 배경)
        if pad_bottom > 0 or pad_right > 0:
            crop_img = cv2.copyMakeBorder(crop_img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
            
        crop_h, crop_w = crop_img.shape[:2]
        
        # 512x512 변환
        inp_np = cv2.resize(crop_img, (512, 512), interpolation=cv2.INTER_AREA)
        
        # SegFace 입력 형식: [0, 1]로 Normalize (ImageNet mean/std 사용)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp_t = (inp_np / 255.0 - mean) / std
        inp_t = torch.from_numpy(inp_t).float().permute(2, 0, 1).unsqueeze(0)
        
        # SegFace는 float32로 고정 실행 (모델 내부 float32 하드코딩 때문에 half 금지)
        inp_t = inp_t.float().to(self.device)

        with torch.no_grad():
            DUMMY_LABELS = None
            DUMMY_DATASET = None
            custom_logits = self._segface(inp_t, DUMMY_LABELS, DUMMY_DATASET)
            if self._segface_custom_binary_hair:
                custom_hair_512 = (
                    torch.sigmoid(custom_logits[:, HAIR_CLASS_IDX:HAIR_CLASS_IDX + 1])
                    >= self._segface_custom_hair_threshold
                ).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            else:
                custom_parsing = custom_logits.argmax(dim=1).squeeze(0).cpu().numpy()
                custom_hair_512 = (custom_parsing == HAIR_CLASS_IDX).astype(np.float32)

            protect_model = self._segface_base or self._segface
            protect_logits = protect_model(inp_t, DUMMY_LABELS, DUMMY_DATASET)
            protect_parsing = protect_logits.argmax(dim=1).squeeze(0).cpu().numpy()

        base_hair_512 = (protect_parsing == HAIR_CLASS_IDX).astype(np.float32)
        glasses_512 = (protect_parsing == GLASS_CLASS_IDX).astype(np.float32)
        earrings_512 = (protect_parsing == EARRING_CLASS_IDX).astype(np.float32)
        necklace_512 = (protect_parsing == NECKLACE_CLASS_IDX).astype(np.float32)
        face_512 = np.isin(protect_parsing, list(FACE_CLASS_IDXS)).astype(np.float32)
        face_512 = np.maximum(face_512, glasses_512).astype(np.float32)
        cloth_512 = (protect_parsing == CLOTH_CLASS_IDX).astype(np.float32)

        # 1. Binary masks keep hard edges better with nearest-neighbor resizing.
        custom_hair_crop = cv2.resize(
            custom_hair_512,
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST,
        )
        base_hair_crop = cv2.resize(
            base_hair_512,
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST,
        )
        face_crop = cv2.resize(
            face_512,
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST,
        )
        glasses_crop = cv2.resize(
            glasses_512,
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST,
        )
        earrings_crop = cv2.resize(
            earrings_512,
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST,
        )
        necklace_crop = cv2.resize(
            necklace_512,
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST,
        )
        cloth_crop = cv2.resize(
            cloth_512,
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # 2. 패딩 부분 잘라내기
        custom_hair_crop = custom_hair_crop[:ch, :cw]
        base_hair_crop = base_hair_crop[:ch, :cw]
        face_crop = face_crop[:ch, :cw]
        glasses_crop = glasses_crop[:ch, :cw]
        earrings_crop = earrings_crop[:ch, :cw]
        necklace_crop = necklace_crop[:ch, :cw]
        cloth_crop = cloth_crop[:ch, :cw]

        custom_hair_ratio = self._mask_ratio(custom_hair_crop)
        base_hair_ratio = self._mask_ratio(base_hair_crop)
        custom_hair_strength_min = float(
            os.environ.get("SEGFACE_CUSTOM_HAIR_RATIO_MIN", "0.010")
        )

        hair_crop = custom_hair_crop
        base_hair_support = np.zeros_like(base_hair_crop, dtype=np.float32)
        if base_hair_ratio > 0.0 and custom_hair_ratio < custom_hair_strength_min:
            local_x1 = max(0, x1 - crop_x1)
            local_y1 = max(0, y1 - crop_y1)
            local_x2 = min(cw, x2 - crop_x1)
            local_y2 = min(ch, y2 - crop_y1)

            head_prior = np.zeros_like(base_hair_crop, dtype=np.float32)
            prior_x1 = max(0, int(local_x1 - bw * 0.65))
            prior_x2 = min(cw, int(local_x2 + bw * 0.65))
            prior_y1 = max(0, int(local_y1 - bh * 1.15))
            prior_y2 = min(ch, int(local_y2 + bh * 0.90))
            if prior_x1 < prior_x2 and prior_y1 < prior_y2:
                head_prior[prior_y1:prior_y2, prior_x1:prior_x2] = 1.0

            base_hair_fallback = base_hair_crop * head_prior
            fallback_ratio = self._mask_ratio(base_hair_fallback)
            if fallback_ratio > 0.0:
                logger.info(
                    "[SDPipeline] custom hair mask fallback to constrained base: custom_ratio=%.4f base_ratio=%.4f fallback_ratio=%.4f",
                    custom_hair_ratio,
                    base_hair_ratio,
                    fallback_ratio,
                )
                base_hair_support = base_hair_fallback
                hair_crop = np.maximum(custom_hair_crop, base_hair_fallback)

        # 3. 원본 HxW 해상도에 덮어쓰기
        hair_orig  = np.zeros((H, W), dtype=np.float32)
        face_orig  = np.zeros((H, W), dtype=np.float32)
        cloth_orig = np.zeros((H, W), dtype=np.float32)
        custom_hair_orig = np.zeros((H, W), dtype=np.float32)
        base_hair_orig = np.zeros((H, W), dtype=np.float32)
        base_hair_support_orig = np.zeros((H, W), dtype=np.float32)
        glasses_orig = np.zeros((H, W), dtype=np.float32)
        earrings_orig = np.zeros((H, W), dtype=np.float32)
        necklace_orig = np.zeros((H, W), dtype=np.float32)

        hair_orig[crop_y1:crop_y2, crop_x1:crop_x2]  = hair_crop
        face_orig[crop_y1:crop_y2, crop_x1:crop_x2]  = face_crop
        glasses_orig[crop_y1:crop_y2, crop_x1:crop_x2] = glasses_crop
        earrings_orig[crop_y1:crop_y2, crop_x1:crop_x2] = earrings_crop
        necklace_orig[crop_y1:crop_y2, crop_x1:crop_x2] = necklace_crop
        cloth_orig[crop_y1:crop_y2, crop_x1:crop_x2] = cloth_crop
        custom_hair_orig[crop_y1:crop_y2, crop_x1:crop_x2] = custom_hair_crop
        base_hair_orig[crop_y1:crop_y2, crop_x1:crop_x2] = base_hair_crop
        base_hair_support_orig[crop_y1:crop_y2, crop_x1:crop_x2] = base_hair_support

        self._last_segface_mask_debug = {
            "custom_hair_mask": (custom_hair_orig > 0.5).astype(np.float32),
            "base_hair_mask": (base_hair_orig > 0.5).astype(np.float32),
            "base_hair_support_mask": (base_hair_support_orig > 0.5).astype(np.float32),
            "glasses_mask": (glasses_orig > 0.5).astype(np.float32),
            "earring_mask": (earrings_orig > 0.5).astype(np.float32),
            "necklace_mask": (necklace_orig > 0.5).astype(np.float32),
            "meta": {
                "crop_box": [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)],
                "crop_scale": crop_scale,
                "crop_center_y_offset": crop_center_y_offset,
                "custom_hair_ratio_crop": float(custom_hair_ratio),
                "base_hair_ratio_crop": float(base_hair_ratio),
                "glasses_ratio_crop": float(self._mask_ratio(glasses_crop)),
                "earring_ratio_crop": float(self._mask_ratio(earrings_crop)),
                "necklace_ratio_crop": float(self._mask_ratio(necklace_crop)),
                "custom_hair_ratio_min": float(custom_hair_strength_min),
                "base_hair_support_ratio_crop": float(self._mask_ratio(base_hair_support)),
                "protect_model": "segface_base" if self._segface_base is not None else "segface_custom",
                "custom_checkpoint": dict(self._segface_load_info),
                "base_checkpoint": dict(self._segface_base_load_info),
            },
        }

        return (
            (hair_orig  > 0.5).astype(np.float32),
            (face_orig  > 0.5).astype(np.float32),
            (cloth_orig > 0.5).astype(np.float32),
        )

    @staticmethod
    def _mask_ratio(mask: np.ndarray) -> float:
        if mask.size == 0:
            return 0.0
        return float((mask > 0.5).sum()) / float(mask.size)

    @staticmethod
    def _resize_mask_to_shape(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        H, W = shape
        if mask.shape[:2] == (H, W):
            return np.clip(mask.astype(np.float32), 0.0, 1.0)
        return cv2.resize(
            np.clip(mask.astype(np.float32), 0.0, 1.0),
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

    def _build_face_protect_mask(
        self,
        face_bbox: Tuple[int, int, int, int],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """Build a conservative face/neck protection mask from the detected face box."""
        H, W = shape
        x1, y1, x2, y2 = face_bbox
        bw = max(int(x2 - x1), 1)
        bh = max(int(y2 - y1), 1)

        mask = np.zeros((H, W), dtype=np.uint8)

        center = (
            int(0.5 * (x1 + x2)),
            int(y1 + bh * 0.50),
        )
        axes = (
            max(1, int(bw * 0.68)),
            max(1, int(bh * 0.82)),
        )
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1)

        neck_x1 = max(0, int(x1 + bw * 0.18))
        neck_x2 = min(W, int(x2 - bw * 0.18))
        neck_y1 = max(0, int(y2 - bh * 0.02))
        neck_y2 = min(H, int(y2 + bh * 0.30))
        if neck_x1 < neck_x2 and neck_y1 < neck_y2:
            mask[neck_y1:neck_y2, neck_x1:neck_x2] = 255

        return (mask > 0).astype(np.float32)

    def _sanitize_face_region_mask(
        self,
        face_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        landmark_face_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Clamp noisy custom-model face masks to a conservative face region."""
        H, W = face_mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        bw = max(int(x2 - x1), 1)
        bh = max(int(y2 - y1), 1)

        window = np.zeros((H, W), dtype=np.uint8)
        x_min = max(0, int(x1 - bw * 0.25))
        x_max = min(W, int(x2 + bw * 0.25))
        y_min = max(0, int(y1 - bh * 0.18))
        y_max = min(H, int(y2 + bh * 0.35))
        if x_min < x_max and y_min < y_max:
            window[y_min:y_max, x_min:x_max] = 255

        landmark_mask_f = None
        landmark_ratio = 0.0
        if landmark_face_mask is not None and landmark_face_mask.shape == (H, W):
            landmark_mask_f = (np.clip(landmark_face_mask, 0.0, 1.0) > 0.5).astype(np.float32)
            landmark_ratio = self._mask_ratio(landmark_mask_f)
            if landmark_ratio > 0.0:
                landmark_u8 = (landmark_mask_f > 0.5).astype(np.uint8) * 255
                landmark_u8 = cv2.dilate(
                    landmark_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                    iterations=1,
                )
                window = np.maximum(window, landmark_u8)

        clipped = cv2.bitwise_and(
            (np.clip(face_mask, 0.0, 1.0) > 0.5).astype(np.uint8) * 255,
            window,
        )
        clipped_f = (clipped > 0).astype(np.float32)
        clipped_ratio = self._mask_ratio(clipped_f)

        if clipped_ratio < 0.01 or clipped_ratio > 0.18:
            if landmark_mask_f is not None and 0.01 <= landmark_ratio <= 0.24:
                logger.info(
                    "[SDPipeline] face mask fallback used: landmark_ratio=%.4f",
                    landmark_ratio,
                )
                return landmark_mask_f
            fallback = self._build_face_protect_mask(face_bbox, (H, W))
            if landmark_mask_f is not None and 0.005 <= landmark_ratio <= 0.24:
                fallback = np.maximum(fallback, landmark_mask_f)
            logger.info(
                "[SDPipeline] face mask fallback used: clipped_ratio=%.4f",
                clipped_ratio,
            )
            return fallback

        if landmark_mask_f is not None and 0.005 <= landmark_ratio <= 0.24:
            merged = np.maximum(clipped_f, landmark_mask_f)
            if self._mask_ratio(merged) <= 0.22:
                return merged.astype(np.float32)

        return clipped_f

    def _build_accessory_protect_mask(
        self,
        face_bbox: Tuple[int, int, int, int],
        earring_mask: Optional[np.ndarray],
        necklace_mask: Optional[np.ndarray],
        hair_length: str,
    ) -> np.ndarray:
        base_mask = earring_mask if isinstance(earring_mask, np.ndarray) else necklace_mask
        if not isinstance(base_mask, np.ndarray):
            return np.zeros((1, 1), dtype=np.float32)

        H, W = base_mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(y1 - face_h * 0.18))
        bottom = min(H, int(y2 + face_h * (0.72 if hair_length == "short" else 0.92)))
        left = max(0, int(x1 - face_w * 0.90))
        right = min(W, int(x2 + face_w * 0.90))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255

        protect_u8 = np.zeros((H, W), dtype=np.uint8)
        if isinstance(earring_mask, np.ndarray) and earring_mask.shape == (H, W):
            earring_u8 = cv2.dilate(
                (np.clip(earring_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9) if hair_length == "short" else (7, 7)),
                iterations=1,
            )
            protect_u8 = cv2.bitwise_or(protect_u8, earring_u8)
        if isinstance(necklace_mask, np.ndarray) and necklace_mask.shape == (H, W):
            necklace_u8 = cv2.dilate(
                (np.clip(necklace_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            )
            protect_u8 = cv2.bitwise_or(protect_u8, necklace_u8)

        protect_u8 = cv2.bitwise_and(protect_u8, corridor_u8)
        if int((protect_u8 > 0).sum()) == 0:
            return np.zeros((H, W), dtype=np.float32)

        protect_f = cv2.GaussianBlur(
            protect_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=4.2,
            sigmaY=4.2,
        )
        return np.clip(protect_f, 0.0, 1.0).astype(np.float32)

    def _sanitize_cloth_mask(
        self,
        img_rgb: np.ndarray,
        cloth_mask: np.ndarray,
        hair_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Keep cloth protection local to the shoulder band and drop noisy masks."""
        H, W = cloth_mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        bw = max(int(x2 - x1), 1)
        bh = max(int(y2 - y1), 1)

        corridor = np.zeros((H, W), dtype=np.uint8)
        x_min = max(0, int(x1 - bw * 1.20))
        x_max = min(W, int(x2 + bw * 1.20))
        y_min = max(0, int(y2 - bh * 0.05))
        y_max = min(H, int(y2 + bh * 0.95))
        if x_min < x_max and y_min < y_max:
            corridor[y_min:y_max, x_min:x_max] = 255

        cloth_u8 = cv2.bitwise_and(
            (np.clip(cloth_mask, 0.0, 1.0) > 0.5).astype(np.uint8) * 255,
            corridor,
        )
        cloth_u8 = cv2.morphologyEx(
            cloth_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        cloth_u8 = cv2.erode(
            cloth_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        original_cloth_u8 = cloth_u8.copy()
        subject_anchor_u8 = self._build_subject_cloth_anchor_mask(
            image_shape=(H, W),
            face_bbox=face_bbox,
        )
        filtered_subject_cloth_u8 = self._filter_cloth_mask_to_subject_anchor(
            cloth_mask_u8=cloth_u8,
            subject_anchor_u8=subject_anchor_u8,
            face_bbox=face_bbox,
        )
        if int((filtered_subject_cloth_u8 > 0).sum()) >= 60:
            cloth_u8 = filtered_subject_cloth_u8
        else:
            cloth_u8 = np.zeros((H, W), dtype=np.uint8)
        if isinstance(self._last_segface_mask_debug, dict):
            self._last_segface_mask_debug["subject_cloth_anchor_mask"] = (
                subject_anchor_u8 > 0
            ).astype(np.float32)
            self._last_segface_mask_debug["subject_cloth_filtered_mask"] = (
                cloth_u8 > 0
            ).astype(np.float32)

        sparse_thresh_px = max(900, int(bw * bh * 0.020))
        filtered_px = int((cloth_u8 > 0).sum())
        original_px = int((original_cloth_u8 > 0).sum())
        should_add_sparse_support = (
            img_rgb.shape[:2] == (H, W)
            and (
                filtered_px < sparse_thresh_px
                or (original_px >= 240 and filtered_px < max(90, int(original_px * 0.22)))
            )
        )
        if should_add_sparse_support:
            sparse_dark_support = self._build_sparse_dark_cloth_support_mask(
                img_rgb=img_rgb,
                base_cloth_mask_u8=cloth_u8,
                hair_mask=hair_mask,
                face_bbox=face_bbox,
                subject_anchor_u8=subject_anchor_u8,
            )
            if int((sparse_dark_support > 0).sum()) >= 120:
                cloth_u8 = cv2.bitwise_or(cloth_u8, sparse_dark_support)
                if isinstance(self._last_segface_mask_debug, dict):
                    self._last_segface_mask_debug["sparse_dark_cloth_support_mask"] = (
                        sparse_dark_support > 0
                    ).astype(np.float32)

        cloth_f = (cloth_u8 > 0).astype(np.float32)
        cloth_ratio = self._mask_ratio(cloth_f)
        hair_area = float((hair_mask > 0.5).sum())
        overlap = float(((cloth_f > 0.5) & (hair_mask > 0.5)).sum())
        overlap_ratio = overlap / max(hair_area, 1.0)

        if cloth_ratio > 0.085 or overlap_ratio > 0.28:
            logger.info(
                "[SDPipeline] cloth mask disabled: ratio=%.4f overlap_ratio=%.4f",
                cloth_ratio,
                overlap_ratio,
            )
            return np.zeros((H, W), dtype=np.float32)

        return cloth_f

    def _build_subject_cloth_anchor_mask(
        self,
        *,
        image_shape: Tuple[int, int],
        face_bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        H, W = image_shape
        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        anchor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(y2 - face_h * 0.02))
        bottom = min(H, int(y2 + face_h * 1.18))
        left = max(0, int(cx - face_w * 0.82))
        right = min(W, int(cx + face_w * 0.82))
        if top < bottom and left < right:
            anchor_gate_u8 = np.zeros((H, W), dtype=np.uint8)
            anchor_gate_u8[top:bottom, left:right] = 255
        else:
            anchor_gate_u8 = np.zeros((H, W), dtype=np.uint8)

        torso_center = (cx, int(y2 + face_h * 0.58))
        torso_axes = (
            max(20, int(face_w * 0.42)),
            max(28, int(face_h * 0.56)),
        )
        cv2.ellipse(anchor_u8, torso_center, torso_axes, 0, 0, 360, 255, -1)

        upper_center = (cx, int(y2 + face_h * 0.22))
        upper_axes = (
            max(16, int(face_w * 0.28)),
            max(12, int(face_h * 0.16)),
        )
        cv2.ellipse(anchor_u8, upper_center, upper_axes, 0, 0, 360, 255, -1)

        shoulder_centers = (
            (int(cx - face_w * 0.40), int(y2 + face_h * 0.18)),
            (int(cx + face_w * 0.40), int(y2 + face_h * 0.18)),
        )
        shoulder_axes = (
            max(18, int(face_w * 0.24)),
            max(16, int(face_h * 0.16)),
        )
        for center in shoulder_centers:
            cv2.ellipse(anchor_u8, center, shoulder_axes, 0, 0, 360, 255, -1)

        anchor_u8 = cv2.morphologyEx(
            anchor_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 19)),
        )
        anchor_u8 = cv2.bitwise_and(anchor_u8, anchor_gate_u8)
        return anchor_u8

    def _filter_cloth_mask_to_subject_anchor(
        self,
        *,
        cloth_mask_u8: np.ndarray,
        subject_anchor_u8: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        H, W = cloth_mask_u8.shape[:2]
        if subject_anchor_u8.shape[:2] != (H, W):
            return np.zeros((H, W), dtype=np.uint8)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (cloth_mask_u8 > 0).astype(np.uint8),
            8,
        )
        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        min_area = max(40, int(face_w * face_h * 0.002))
        max_dx = max(72, int(face_w * 0.95))
        min_bottom = int(y2 + face_h * 0.02)

        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom_y = y + h
            if area < min_area or bottom_y < min_bottom:
                continue
            comp_u8 = (labels == idx).astype(np.uint8) * 255
            overlap_px = int((cv2.bitwise_and(comp_u8, subject_anchor_u8) > 0).sum())
            comp_cx = float(centroids[idx][0])
            if overlap_px <= 0 and abs(comp_cx - cx) > max_dx:
                continue
            keep_u8[labels == idx] = 255

        return keep_u8

    def _trim_blocky_short_restore_mask_u8(
        self,
        *,
        mask_u8: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        min_keep_px: int = 60,
    ) -> np.ndarray:
        H, W = mask_u8.shape[:2]
        original_px = int((mask_u8 > 0).sum())
        if original_px < min_keep_px:
            return mask_u8

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (mask_u8 > 0).astype(np.uint8),
            8,
        )
        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        min_area = max(16, int(face_w * face_h * 0.0012))
        reject_far_offset = max(180, int(face_w * 1.05))
        reject_center_offset = max(24, int(face_w * 0.22))
        reject_center_width = max(72, int(face_w * 0.38))
        reject_wide_width = max(110, int(face_w * 0.62))
        reject_center_area = max(280, int(face_w * face_h * 0.030))
        reject_wide_area = max(520, int(face_w * face_h * 0.045))
        reject_far_area = max(180, int(face_w * face_h * 0.015))
        min_bottom = int(cutoff_y + face_h * 0.06)

        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            bottom_y = y + h
            comp_cx = float(centroids[idx][0])
            offset = abs(comp_cx - cx)
            fill_ratio = float(area) / float(max(w * h, 1))

            reject_far_blob = (
                offset > reject_far_offset
                and area >= reject_far_area
                and (fill_ratio >= 0.30 or w >= max(56, int(face_w * 0.26)))
            )
            reject_center_block = (
                offset <= reject_center_offset
                and bottom_y >= min_bottom
                and w >= reject_center_width
                and area >= reject_center_area
                and fill_ratio >= 0.34
            )
            reject_dense_wide = (
                bottom_y >= min_bottom
                and w >= reject_wide_width
                and area >= reject_wide_area
                and fill_ratio >= 0.46
            )
            reject_rect_patch = (
                bottom_y >= min_bottom
                and h >= max(72, int(face_h * 0.26))
                and w >= max(84, int(face_w * 0.34))
                and fill_ratio >= 0.58
            )
            if reject_far_blob or reject_center_block or reject_dense_wide or reject_rect_patch:
                continue
            keep_u8[labels == idx] = 255

        kept_px = int((keep_u8 > 0).sum())
        if kept_px == original_px:
            return mask_u8
        if kept_px < max(12, min_keep_px // 4):
            if original_px >= max(min_keep_px * 2, int(face_w * face_h * 0.05)):
                return np.zeros((H, W), dtype=np.uint8)
            return mask_u8
        return keep_u8

    def _build_sparse_dark_cloth_support_mask(
        self,
        *,
        img_rgb: np.ndarray,
        base_cloth_mask_u8: np.ndarray,
        hair_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        subject_anchor_u8: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        H, W = img_rgb.shape[:2]
        if base_cloth_mask_u8.shape[:2] != (H, W) or hair_mask.shape[:2] != (H, W):
            return np.zeros((H, W), dtype=np.uint8)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        sat = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.2, sigmaY=5.2)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(y2 - face_h * 0.02))
        bottom = min(H, int(y2 + face_h * 1.18))
        left = max(0, int(x1 - face_w * 1.20))
        right = min(W, int(x2 + face_w * 1.20))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.uint8)
        corridor_u8[top:bottom, left:right] = 255

        candidate_u8 = (
            (
                (gray < 132.0)
                & (sat < 154.0)
                & (
                    (np.abs(gray - blur) < 11.5)
                    | (lap < 11.0)
                )
            ).astype(np.uint8)
            * 255
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, corridor_u8)

        hair_u8 = cv2.dilate(
            (np.clip(hair_mask.astype(np.float32), 0.0, 1.0) > 0.45).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
            iterations=1,
        )
        upper_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
        upper_bottom = min(H, int(y2 + face_h * 0.30))
        upper_left = max(0, int(cx - face_w * 0.46))
        upper_right = min(W, int(cx + face_w * 0.46))
        if top < upper_bottom and upper_left < upper_right:
            upper_keepout_u8[top:upper_bottom, upper_left:upper_right] = 255
            candidate_u8 = cv2.bitwise_and(
                candidate_u8,
                cv2.bitwise_not(cv2.bitwise_and(hair_u8, upper_keepout_u8)),
            )

        anchor_u8 = cv2.dilate(
            base_cloth_mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )
        if subject_anchor_u8 is not None and subject_anchor_u8.shape[:2] == (H, W):
            anchor_u8 = cv2.bitwise_or(
                anchor_u8,
                cv2.bitwise_and(subject_anchor_u8, corridor_u8),
            )
        side_seed_u8 = np.zeros((H, W), dtype=np.uint8)
        side_centers = (
            (int(cx - face_w * 0.54), int(y2 + face_h * 0.14)),
            (int(cx + face_w * 0.54), int(y2 + face_h * 0.14)),
        )
        side_axes = (
            max(18, int(face_w * 0.28)),
            max(16, int(face_h * 0.18)),
        )
        for center in side_centers:
            cv2.ellipse(side_seed_u8, center, side_axes, 0, 0, 360, 255, -1)
        anchor_u8 = cv2.bitwise_or(anchor_u8, cv2.bitwise_and(side_seed_u8, corridor_u8))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8, 8)
        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        min_area = max(90, int(face_w * face_h * 0.003))
        max_area = max(42000, int(face_w * face_h * 0.80))
        min_height = max(18, int(face_h * 0.10))
        max_width = max(280, int(face_w * 1.72))
        min_bottom = int(y2 + face_h * 0.10)
        deep_bottom = int(y2 + face_h * 0.42)
        thin_width = max(18, int(face_w * 0.14))
        tall_height = max(60, int(face_h * 0.52))

        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom_y = y + h
            if area < min_area or area > max_area:
                continue
            if h < min_height or w > max_width:
                continue
            if w <= thin_width and h >= tall_height:
                continue
            if bottom_y < min_bottom:
                continue
            comp_u8 = (labels == idx).astype(np.uint8) * 255
            anchor_overlap = int(cv2.bitwise_and(comp_u8, anchor_u8).sum() > 0)
            if anchor_overlap <= 0 and not (bottom_y >= deep_bottom and w >= max(26, int(face_w * 0.18))):
                continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 120:
            return np.zeros((H, W), dtype=np.uint8)

        keep_u8 = cv2.morphologyEx(
            keep_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
        )
        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
            iterations=1,
        )
        return cv2.bitwise_and(keep_u8, corridor_u8)

    def _build_bangs_recovery_mask(
        self,
        hair_mask: np.ndarray,
        protect_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        *,
        landmark_debug_data: Optional[Dict[str, Any]] = None,
        hair_length: str = "short",
    ) -> np.ndarray:
        """
        얼굴 보호 마스크에 의해 같이 깎인 앞머리만 제한적으로 복원한다.
        중앙 이마 밴드에서 원래 hair mask가 잡고 있던 성분만 되살린다.
        """
        H, W = hair_mask.shape[:2]
        if protect_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        hair_f = np.clip(hair_mask.astype(np.float32), 0.0, 1.0)
        protect_f = np.clip(protect_mask.astype(np.float32), 0.0, 1.0)
        overlap = np.clip(hair_f * (protect_f > 0.10).astype(np.float32), 0.0, 1.0)
        if float(overlap.sum()) < 12.0:
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))
        keypoints = landmark_debug_data.get("keypoints", {}) if isinstance(landmark_debug_data, dict) else {}
        forehead_top = keypoints.get("forehead_top", {}).get("px")
        forehead_y = int(forehead_top[1]) if isinstance(forehead_top, list) and len(forehead_top) >= 2 else int(y1)

        band_top = max(0, int(min(y1, forehead_y) - face_h * 0.16))
        band_bottom = min(
            H,
            int(forehead_y + face_h * (0.48 if hair_length == "short" else 0.42 if hair_length == "medium" else 0.40)),
        )
        center_half = max(
            18,
            int(face_w * (0.50 if hair_length == "short" else 0.50 if hair_length == "medium" else 0.46)),
        )
        band_x1 = max(0, cx - center_half)
        band_x2 = min(W, cx + center_half)
        if band_top >= band_bottom or band_x1 >= band_x2:
            return np.zeros((H, W), dtype=np.float32)

        corridor = np.zeros((H, W), dtype=np.uint8)
        corridor[band_top:band_bottom, band_x1:band_x2] = 255
        recover_u8 = cv2.bitwise_and((overlap > 0.12).astype(np.uint8) * 255, corridor)
        if int((recover_u8 > 0).sum()) < 18:
            return np.zeros((H, W), dtype=np.float32)

        support_top = max(0, int(band_top - face_h * 0.22))
        support_bottom = min(
            H,
            int(forehead_y + face_h * (0.22 if hair_length == "short" else 0.16 if hair_length == "medium" else 0.14)),
        )
        support_half = max(24, int(face_w * (0.52 if hair_length == "short" else 0.48 if hair_length == "medium" else 0.42)))
        support_x1 = max(0, cx - support_half)
        support_x2 = min(W, cx + support_half)
        support_u8 = np.zeros((H, W), dtype=np.uint8)
        if support_top < support_bottom and support_x1 < support_x2:
            support_u8[support_top:support_bottom, support_x1:support_x2] = 255
            support_u8 = cv2.bitwise_and(
                support_u8,
                (hair_f > 0.20).astype(np.uint8) * 255,
            )

        recover_u8 = cv2.morphologyEx(
            recover_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)),
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(recover_u8, 8)
        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        max_component_area = max(
            160,
            int(face_w * face_h * (0.36 if hair_length == "short" else 0.30 if hair_length == "medium" else 0.24)),
        )
        max_component_width = max(
            34,
            int(face_w * (1.02 if hair_length == "short" else 0.92 if hair_length == "medium" else 0.80)),
        )
        min_component_height = max(8, int(face_h * 0.08))
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 14 or area > max_component_area:
                continue
            if w > max_component_width or h < min_component_height:
                continue
            comp_u8 = (labels == idx).astype(np.uint8) * 255
            support_overlap = int((cv2.bitwise_and(comp_u8, support_u8) > 0).sum())
            if support_overlap < 6 and y > int(forehead_y + face_h * 0.08):
                continue
            keep_u8 = cv2.bitwise_or(keep_u8, comp_u8)

        if int((keep_u8 > 0).sum()) < 14:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 11) if hair_length == "short" else (7, 11) if hair_length == "medium" else (5, 9),
            ),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, corridor)
        return (keep_u8 > 0).astype(np.float32)

    def _build_soft_bangs_generation_mask(
        self,
        bangs_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        hair_length: str = "short",
    ) -> np.ndarray:
        H, W = bangs_mask.shape[:2]
        base = (np.clip(bangs_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255
        if int((base > 0).sum()) < 8:
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        is_long = hair_length == "long"
        base = cv2.morphologyEx(
            base,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)),
        )
        base = cv2.dilate(
            base,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5) if is_long else (5, 7)),
            iterations=1,
        )
        ys = np.where(base > 0)[0]
        if ys.size == 0:
            return np.zeros((H, W), dtype=np.float32)

        top = int(ys.min())
        bottom = int(ys.max()) + 1
        cx = int(0.5 * (x1 + x2))
        side_keep = max(18, int(face_w * (0.38 if is_long else 0.46)))
        x_min = max(0, cx - side_keep)
        x_max = min(W, cx + side_keep)
        band_top = max(0, int(top - face_h * (0.05 if is_long else 0.08)))
        band_bottom = min(H, int(bottom + face_h * (0.05 if is_long else 0.08)))
        if band_top >= band_bottom or x_min >= x_max:
            return np.zeros((H, W), dtype=np.float32)

        band_u8 = np.zeros((H, W), dtype=np.uint8)
        band_u8[band_top:band_bottom, x_min:x_max] = 255
        soft_u8 = cv2.bitwise_and(base, band_u8)
        if int((soft_u8 > 0).sum()) < 8:
            return np.zeros((H, W), dtype=np.float32)

        soft_u8 = cv2.morphologyEx(
            soft_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5) if is_long else (5, 7)),
        )
        soft_u8 = cv2.dilate(
            soft_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7) if is_long else (5, 9)),
            iterations=1,
        )
        if int((soft_u8 > 0).sum()) < 8:
            return np.zeros((H, W), dtype=np.float32)

        alpha = cv2.GaussianBlur(
            soft_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=1.8 if is_long else 2.2,
            sigmaY=2.1 if is_long else 2.6,
        )
        alpha = np.clip((alpha - 0.02) / 0.94, 0.0, 1.0)

        fade = np.zeros((H,), dtype=np.float32)
        if band_bottom > band_top:
            inner_top = band_top
            inner_mid = min(band_bottom, band_top + max(10, int((band_bottom - band_top) * 0.34)))
            inner_low = min(band_bottom, band_top + max(16, int((band_bottom - band_top) * 0.72)))
            if inner_mid > inner_top:
                fade[inner_top:inner_mid] = np.linspace(
                    0.16,
                    0.42,
                    inner_mid - inner_top,
                    dtype=np.float32,
                )
            if inner_low > inner_mid:
                fade[inner_mid:inner_low] = np.linspace(
                    0.42,
                    0.72,
                    inner_low - inner_mid,
                    dtype=np.float32,
                )
            if band_bottom > inner_low:
                fade[inner_low:band_bottom] = np.linspace(
                    0.72,
                    0.52,
                    band_bottom - inner_low,
                    dtype=np.float32,
                )
        tail_end = min(H, band_bottom + max(4 if is_long else 6, int(face_h * (0.04 if is_long else 0.06))))
        if tail_end > band_bottom:
            fade[band_bottom:tail_end] = np.linspace(
                max(0.0, float(fade[band_bottom - 1])) if band_bottom > 0 else 0.44,
                0.0,
                tail_end - band_bottom,
                dtype=np.float32,
            )

        alpha = alpha * fade[:, np.newaxis]
        x_coords = np.arange(W, dtype=np.float32)
        side_scale = max(float(side_keep), 1.0)
        x_dist = np.abs(x_coords - float(cx)) / side_scale
        x_fade = np.clip(
            1.0 - (x_dist ** (1.72 if is_long else 1.55)) * (0.62 if is_long else 0.52),
            0.36 if is_long else 0.44,
            1.0,
        ).astype(np.float32)
        alpha = alpha * x_fade[np.newaxis, :]
        return np.clip(alpha, 0.0, 0.54 if is_long else 0.68).astype(np.float32)

    def _build_eye_region_restore_mask(
        self,
        landmark_debug_data: Optional[Dict[str, Any]],
        image_shape: Tuple[int, int],
        face_bbox: Tuple[int, int, int, int],
        hair_length: str = "long",
        final_hair_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        H, W = image_shape
        if not isinstance(landmark_debug_data, dict):
            return np.zeros((H, W), dtype=np.float32)

        landmarks_px = landmark_debug_data.get("landmarks_px")
        if not isinstance(landmarks_px, list) or len(landmarks_px) < 20:
            return np.zeros((H, W), dtype=np.float32)

        try:
            import mediapipe as mp

            left_eye_indices = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_LEFT_EYE for i in edge})
            right_eye_indices = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE for i in edge})
            left_brow_indices = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW for i in edge})
            right_brow_indices = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW for i in edge})
        except Exception:
            return np.zeros((H, W), dtype=np.float32)

        pts = np.asarray(landmarks_px, dtype=np.int32)
        n = int(len(pts))
        left_eye_pts = np.asarray([pts[i] for i in left_eye_indices if i < n], dtype=np.int32)
        right_eye_pts = np.asarray([pts[i] for i in right_eye_indices if i < n], dtype=np.int32)
        left_brow_pts = np.asarray([pts[i] for i in left_brow_indices if i < n], dtype=np.int32)
        right_brow_pts = np.asarray([pts[i] for i in right_brow_indices if i < n], dtype=np.int32)
        if len(left_eye_pts) < 3 or len(right_eye_pts) < 3:
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_h = max(int(y2 - y1), 1)
        face_w = max(int(x2 - x1), 1)
        dilate_px = max(9, int(face_h * 0.10))
        brow_dilate_px = max(7, int(face_h * 0.07))

        left_mask = self._build_landmark_hull_mask(left_eye_pts, (H, W), dilate_px=dilate_px)
        right_mask = self._build_landmark_hull_mask(right_eye_pts, (H, W), dilate_px=dilate_px)
        brow_mask = np.zeros((H, W), dtype=np.float32)
        if len(left_brow_pts) >= 3:
            brow_mask = np.maximum(
                brow_mask,
                self._build_landmark_hull_mask(left_brow_pts, (H, W), dilate_px=brow_dilate_px),
            ).astype(np.float32)
        if len(right_brow_pts) >= 3:
            brow_mask = np.maximum(
                brow_mask,
                self._build_landmark_hull_mask(right_brow_pts, (H, W), dilate_px=brow_dilate_px),
            ).astype(np.float32)
        eye_mask = np.maximum(left_mask, right_mask).astype(np.float32)
        if float(brow_mask.sum()) > 0.0:
            eye_mask = np.maximum(
                eye_mask,
                np.clip(brow_mask * 0.88, 0.0, 1.0),
            ).astype(np.float32)
        if float(eye_mask.sum()) < 10.0:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(y1 + face_h * 0.18))
        bottom = min(H, int(y1 + face_h * 0.64))
        left = max(0, int(x1 - face_w * 0.04))
        right = min(W, int(x2 + face_w * 0.04))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255

        eye_u8 = cv2.bitwise_and(
            (np.clip(eye_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255,
            corridor_u8,
        )
        eye_u8 = cv2.dilate(
            eye_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 9)),
            iterations=1,
        )
        band_pts = [left_eye_pts, right_eye_pts]
        if len(left_brow_pts) >= 3:
            band_pts.append(left_brow_pts)
        if len(right_brow_pts) >= 3:
            band_pts.append(right_brow_pts)
        try:
            band_pts_arr = np.concatenate(band_pts, axis=0)
        except Exception:
            band_pts_arr = np.zeros((0, 2), dtype=np.int32)
        if len(band_pts_arr) >= 6:
            band_x1 = max(0, int(np.min(band_pts_arr[:, 0]) - face_w * 0.04))
            band_x2 = min(W, int(np.max(band_pts_arr[:, 0]) + face_w * 0.04))
            band_top = max(0, int(np.min(band_pts_arr[:, 1]) - face_h * 0.04))
            band_bottom = min(H, int(max(left_eye_pts[:, 1].max(), right_eye_pts[:, 1].max()) + face_h * 0.06))
            if band_top < band_bottom and band_x1 < band_x2:
                band_u8 = np.zeros((H, W), dtype=np.uint8)
                band_u8[band_top:band_bottom, band_x1:band_x2] = 255
                band_u8 = cv2.bitwise_and(band_u8, corridor_u8)
                band_u8 = cv2.GaussianBlur(band_u8, (0, 0), sigmaX=2.2, sigmaY=1.6)
                eye_u8 = cv2.bitwise_or(eye_u8, (band_u8 > 24).astype(np.uint8) * 255)
        if (
            hair_length != "long"
            and final_hair_mask is not None
            and final_hair_mask.shape == (H, W)
        ):
            hair_u8 = cv2.dilate(
                (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.28).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)),
                iterations=1,
            )
            eye_u8 = cv2.bitwise_and(eye_u8, cv2.bitwise_not(hair_u8))
        if int((eye_u8 > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)

        return cv2.GaussianBlur(
            eye_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=2.0,
            sigmaY=2.4,
        ).astype(np.float32)

    def _build_face_eye_band_restore_mask(
        self,
        face_mask: Optional[np.ndarray],
        image_shape: Tuple[int, int],
        face_bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        H, W = image_shape
        if face_mask is None or face_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_h = max(int(y2 - y1), 1)
        face_w = max(int(x2 - x1), 1)

        face_u8 = cv2.dilate(
            (np.clip(face_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        band_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(y1 + face_h * 0.20))
        bottom = min(H, int(y1 + face_h * 0.60))
        left = max(0, int(x1 - face_w * 0.04))
        right = min(W, int(x2 + face_w * 0.04))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        band_u8[top:bottom, left:right] = 255
        band_u8 = cv2.bitwise_and(band_u8, face_u8)
        if int((band_u8 > 0).sum()) < 36:
            return np.zeros((H, W), dtype=np.float32)
        return cv2.GaussianBlur(
            band_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=2.8,
            sigmaY=2.4,
        ).astype(np.float32)

    @staticmethod
    def _restore_reference_region(
        base_rgb: np.ndarray,
        reference_rgb: np.ndarray,
        restore_mask: np.ndarray,
        *,
        strength: float = 0.94,
    ) -> np.ndarray:
        H, W = base_rgb.shape[:2]
        if reference_rgb.shape[:2] != (H, W) or restore_mask.shape != (H, W):
            return base_rgb
        alpha = cv2.GaussianBlur(
            np.clip(restore_mask.astype(np.float32), 0.0, 1.0),
            (0, 0),
            sigmaX=1.8,
            sigmaY=1.8,
        )[..., np.newaxis]
        alpha = np.clip(alpha * strength, 0.0, 1.0)
        out = reference_rgb.astype(np.float32) * alpha + base_rgb.astype(np.float32) * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _dilate_mask_with_px(self, mask: np.ndarray, px: int) -> np.ndarray:
        """Dilate with an explicit pixel size without changing the global config."""
        if px <= 0:
            return np.clip(mask, 0.0, 1.0).astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return np.clip(dilated, 0.0, 1.0).astype(np.float32)

    def _dilate_hair_mask_for_length(
        self,
        mask: np.ndarray,
        hair_length: str,
    ) -> np.ndarray:
        """Use a smaller dilation for short/medium hair to avoid lower-mask overgrowth."""
        px = int(self.config.mask_dilate_px)
        if hair_length == "short":
            px = max(5, min(11, int(round(px * 0.35))))
        elif hair_length == "medium":
            px = max(7, min(15, int(round(px * 0.50))))
        return self._dilate_mask_with_px(mask, px)

    def _resolve_mask_refine_mode(
        self,
        mask_refine_mode: Optional[str],
    ) -> str:
        value = str(
            mask_refine_mode
            or getattr(self.config, "mask_refine_mode", "")
            or os.environ.get("MASK_REFINE_MODE", "sam2")
        ).strip().lower()
        if value not in {"sam2", "segface_priority", "segface_only"}:
            logger.warning(
                "[SDPipeline] unsupported mask_refine_mode '%s', falling back to sam2",
                value,
            )
            return "sam2"
        return value

    def _merge_segface_priority_mask(
        self,
        *,
        base_mask: np.ndarray,
        sam2_mask: np.ndarray,
        hair_length: str,
    ) -> np.ndarray:
        base_u8 = ((np.clip(base_mask, 0.0, 1.0) > 0.5).astype(np.uint8) * 255)
        sam2_u8 = ((np.clip(sam2_mask, 0.0, 1.0) > 0.5).astype(np.uint8) * 255)

        if hair_length == "short":
            core_kernel = (5, 5)
            growth_px = 7
            close_kernel = (7, 7)
        elif hair_length == "medium":
            core_kernel = (7, 7)
            growth_px = 9
            close_kernel = (9, 9)
        else:
            core_kernel = (9, 9)
            growth_px = 13
            close_kernel = (11, 11)

        base_core = cv2.erode(
            base_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, core_kernel),
            iterations=1,
        )
        allow_growth = (self._dilate_mask_with_px(base_u8.astype(np.float32) / 255.0, growth_px) > 0.5).astype(np.uint8) * 255
        sam2_local = cv2.bitwise_and(sam2_u8, allow_growth)
        merged_u8 = cv2.bitwise_or(base_core, sam2_local)

        base_px = int((base_u8 > 0).sum())
        merged_px = int((merged_u8 > 0).sum())
        if merged_px < max(60, int(base_px * 0.55)):
            logger.info(
                "[SDPipeline] segface_priority fallback to base mask: base_px=%s merged_px=%s",
                base_px,
                merged_px,
            )
            return (base_u8 > 0).astype(np.float32)

        merged_u8 = cv2.morphologyEx(
            merged_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel),
        )
        return (merged_u8 > 0).astype(np.float32)

    def _refine_with_sam2(
        self,
        img_rgb: np.ndarray,           # H×W×3 RGB
        base_mask: np.ndarray,          # H×W float32
        face_bbox: Tuple[int, int, int, int],
        prompt_text: str,
        mask_refine_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, str, str]:
        """
        SAM2로 SegFace 마스크를 정밀 보정.

        Returns:
            (refined_mask H×W float32, source_name, used_refine_mode)
        """
        hair_length = self._classify_hair_length(prompt_text)
        refine_mode = self._resolve_mask_refine_mode(mask_refine_mode)
        if refine_mode == "segface_only":
            return self._dilate_hair_mask_for_length(base_mask, hair_length), "segface", "segface_only"
        if self._sam2_factory is None:
            return self._dilate_hair_mask_for_length(base_mask, hair_length), "segface", refine_mode

        try:
            predictor = self._sam2_factory()
            H, W = img_rgb.shape[:2]
            x1, y1, x2, y2 = face_bbox

            bw = x2 - x1
            bh = y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # SAM2 bbox: 긴 머리 고려해서 하단을 base hair mask 최하단까지 확장
            hair_coords = np.argwhere(base_mask > 0.5)  # (N,2) [row, col]
            if len(hair_coords) > 0:
                hair_bottom = int(hair_coords[:, 0].max())
                bbox_bottom = min(H - 1, max(hair_bottom + 20, y2 + int(bh * 0.2)))
            else:
                bbox_bottom = min(H - 1, y2 + int(bh * 0.6))  # fallback: 얼굴 높이 60% 아래

            sam_bbox = np.array([
                max(0,     x1 - int(bw * 0.6)),
                max(0,     y1 - int(bh * 0.6)),
                min(W - 1, x2 + int(bw * 0.6)),
                bbox_bottom,
            ], dtype=np.float32)

            # Positive points: 정수리/옆머리 + 앞머리 + 긴 머리 흘러내리는 옆쪽
            hair_top_y  = max(5, y1 - int(bh * 0.25))   # 정수리
            side_y      = max(5, y1 - int(bh * 0.05))   # 귀 위쪽
            bangs_y     = max(5, y1 + int(bh * 0.10))   # 앞머리 (이마 위)
            long_hair_y = min(H - 5, y2 + int(bh * 0.4)) # 턱 아래 긴 머리
            pos_pts = np.array([
                [cx,                    hair_top_y],   # 정수리 중앙
                [cx - int(bw * 0.25),   hair_top_y],   # 정수리 왼쪽
                [cx + int(bw * 0.25),   hair_top_y],   # 정수리 오른쪽
                [x1 - int(bw * 0.05),   side_y],       # 왼쪽 옆머리
                [x2 + int(bw * 0.05),   side_y],       # 오른쪽 옆머리
                [cx - int(bw * 0.15),   bangs_y],      # 앞머리 왼쪽
                [cx + int(bw * 0.15),   bangs_y],      # 앞머리 오른쪽
                [x1 - int(bw * 0.2),    long_hair_y],  # 왼쪽 긴 머리
                [x2 + int(bw * 0.2),    long_hair_y],  # 오른쪽 긴 머리
            ], dtype=np.float32)
            pos_pts[:, 0] = np.clip(pos_pts[:, 0], 0, W - 1)
            pos_pts[:, 1] = np.clip(pos_pts[:, 1], 0, H - 1)
            tail_zone_u8 = np.zeros((H, W), dtype=np.uint8)
            tail_x1 = max(0, x1 - int(bw * 0.78))
            tail_x2 = min(W, x2 + int(bw * 0.78))
            tail_y1 = max(0, int(y2 - bh * 0.02))
            tail_y2 = min(H, int(y2 + bh * (0.86 if hair_length == "short" else 0.92)))
            if tail_x1 < tail_x2 and tail_y1 < tail_y2:
                tail_zone_u8[tail_y1:tail_y2, tail_x1:tail_x2] = 255
                tail_seed_u8 = cv2.bitwise_and(
                    (base_mask > 0.35).astype(np.uint8) * 255,
                    tail_zone_u8,
                )
                if int((tail_seed_u8 > 0).sum()) > 0:
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tail_seed_u8, 8)
                    extra_tail_pts: list[list[float]] = []
                    seen: set[tuple[int, int]] = set()
                    for idx in range(1, num_labels):
                        area = int(stats[idx, cv2.CC_STAT_AREA])
                        if area < max(16, int(bw * bh * 0.003)):
                            continue
                        comp_cx = float(centroids[idx][0])
                        comp_cy = float(centroids[idx][1])
                        key = (int(round(comp_cx / 8.0)), int(round(comp_cy / 8.0)))
                        if key in seen:
                            continue
                        seen.add(key)
                        extra_tail_pts.append([comp_cx, comp_cy])
                    if extra_tail_pts:
                        tail_candidates = np.asarray(extra_tail_pts, dtype=np.float32)
                        ranked_indices = sorted(
                            range(len(tail_candidates)),
                            key=lambda idx_pt: (
                                -abs(float(tail_candidates[idx_pt][0]) - float(cx)),
                                -float(tail_candidates[idx_pt][1]),
                                idx_pt,
                            ),
                        )
                        chosen_indices: list[int] = []
                        used_bins: set[tuple[str, int, int]] = set()
                        for idx_pt in ranked_indices:
                            pt_x, pt_y = tail_candidates[idx_pt]
                            side_key = (
                                "left"
                                if float(pt_x) < float(cx) - bw * 0.08
                                else "right"
                                if float(pt_x) > float(cx) + bw * 0.08
                                else "center"
                            )
                            uniq_key = (
                                side_key,
                                int(round(float(pt_x) / 10.0)),
                                int(round(float(pt_y) / 10.0)),
                            )
                            if uniq_key in used_bins:
                                continue
                            used_bins.add(uniq_key)
                            chosen_indices.append(idx_pt)
                            if len(chosen_indices) >= 4:
                                break
                        tail_pts = tail_candidates[chosen_indices].astype(np.float32)
                        tail_pts[:, 0] = np.clip(tail_pts[:, 0], 0, W - 1)
                        tail_pts[:, 1] = np.clip(tail_pts[:, 1], 0, H - 1)
                        pos_pts = np.concatenate([pos_pts, tail_pts], axis=0)

            # Negative points: 얼굴 격자 9점 + 목/상체 중앙 (몸통 잡지 않도록)
            neck_y   = min(H - 5, y2 + int(bh * 0.15))
            body_y   = min(H - 5, y2 + int(bh * 0.5))
            neg_pts = np.array([
                # 얼굴 상단부 (이마)
                [x1 + int(bw * 0.25), y1 + int(bh * 0.25)],
                [cx,                   y1 + int(bh * 0.25)],
                [x2 - int(bw * 0.25), y1 + int(bh * 0.25)],
                # 얼굴 중앙부 (눈/코)
                [x1 + int(bw * 0.25), cy],
                [cx,                   cy],
                [x2 - int(bw * 0.25), cy],
                # 얼굴 하단부 (입/턱)
                [x1 + int(bw * 0.25), y2 - int(bh * 0.15)],
                [cx,                   y2 - int(bh * 0.15)],
                [x2 - int(bw * 0.25), y2 - int(bh * 0.15)],
                # 목/상체 중앙 (긴 머리가 옆으로 흘러도 몸통 중앙은 제외)
                [cx,  neck_y],
                [cx,  body_y],
            ], dtype=np.float32)
            # 이미지 범위 클램프
            neg_pts[:, 0] = np.clip(neg_pts[:, 0], 0, W - 1)
            neg_pts[:, 1] = np.clip(neg_pts[:, 1], 0, H - 1)

            point_coords = np.concatenate([pos_pts, neg_pts], axis=0)
            point_labels = np.concatenate([
                np.ones(len(pos_pts),  dtype=np.int32),
                np.zeros(len(neg_pts), dtype=np.int32),
            ])

            # SAM2 predict (multimask=True → 가장 face overlap 적은 마스크 선택)
            predictor.set_image(img_rgb)
            prediction = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=sam_bbox[None, :],
                multimask_output=True,
            )

            # predict() 반환 형태: dict | (masks, scores, logits) tuple
            if isinstance(prediction, dict):
                masks = prediction.get("masks")
            elif isinstance(prediction, (tuple, list)):
                # (masks, iou_scores, low_res_logits) 형태로 반환
                masks = prediction[0]
                # 드물게 masks 자체가 또 tuple/list인 경우 unwrap
                while isinstance(masks, (tuple, list)):
                    masks = masks[0]
            else:
                masks = prediction

            if masks is not None:
                # numpy/tensor → numpy 변환
                if hasattr(masks, "cpu"):
                    masks_np = masks.cpu().numpy()
                else:
                    masks_np = np.asarray(masks)

                # shape 정규화: (N,H,W) or (H,W)
                if masks_np.ndim == 2:
                    masks_np = masks_np[np.newaxis]  # → (1,H,W)
                elif masks_np.ndim != 3 or masks_np.shape[0] == 0:
                    raise ValueError(f"Unexpected SAM2 mask shape: {masks_np.shape}")

                # multimask: SegFace의 base_mask와 가장 일치하는(IoU가 높은) 마스크를 선택
                best_mask = None
                best_iou = -1.0
                
                # base_mask (SegFace 예측 결과)
                base_f = (base_mask > 0.5).astype(np.float32)
                base_sum = base_f.sum()
                
                for m in masks_np:
                    m_f = (m > 0.5).astype(np.float32)
                    if m_f.shape != (H, W):
                        m_f = cv2.resize(m_f, (W, H), interpolation=cv2.INTER_LINEAR)
                        m_f = (m_f > 0.5).astype(np.float32)
                    
                    # Compute IoU with base_mask
                    intersection = (m_f * base_f).sum()
                    union = m_f.sum() + base_sum - intersection
                    iou = intersection / (union + 1e-6)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = m_f

                refined_np = np.clip(best_mask.astype(np.float32), 0.0, 1.0)

                if hair_length in ("short", "medium"):
                    allow_growth_px = 11 if hair_length == "short" else 15
                    allow_growth = self._dilate_mask_with_px(base_f, allow_growth_px)
                    allow_growth_u8 = (allow_growth > 0.5).astype(np.uint8) * 255

                    if len(hair_coords) > 0:
                        base_bottom = int(hair_coords[:, 0].max())
                    else:
                        base_bottom = int(y2)

                    growth_limit_y = min(
                        H,
                        base_bottom + int(bh * (0.14 if hair_length == "short" else 0.18)),
                    )
                    tail_pad_x = int(bw * (0.52 if hair_length == "short" else 0.58))
                    growth_window = np.zeros((H, W), dtype=np.uint8)
                    growth_window[:growth_limit_y, :] = 255
                    tail_x1 = max(0, x1 - tail_pad_x)
                    tail_x2 = min(W, x2 + tail_pad_x)
                    if tail_x1 < tail_x2 and growth_limit_y < H:
                        growth_window[growth_limit_y:, tail_x1:tail_x2] = 255

                    refined_u8 = (refined_np > 0.5).astype(np.uint8) * 255
                    refined_u8 = cv2.bitwise_and(refined_u8, allow_growth_u8)
                    refined_u8 = cv2.bitwise_and(refined_u8, growth_window)
                    refined_np = (refined_u8 > 0).astype(np.float32)
                    logger.info(
                        "[SDPipeline] SAM2 conservative cap: hair_length=%s base_px=%.0f refined_px=%.0f",
                        hair_length,
                        float(base_f.sum()),
                        float(refined_np.sum()),
                    )
                else:
                    # long hair는 SAM2의 확장을 더 넓게 허용한다.
                    base_mask_dilated = self._dilate_mask(base_mask)
                    refined_np = np.clip(refined_np * base_mask_dilated, 0.0, 1.0)
                
                if refined_np.sum() < 300:
                    logger.warning("[SDPipeline] SAM2 결과가 너무 작아 SegFace로 폴백")
                    return self._dilate_hair_mask_for_length(base_mask, hair_length), "segface", refine_mode

                if refine_mode == "segface_priority":
                    refined_np = self._merge_segface_priority_mask(
                        base_mask=base_mask,
                        sam2_mask=refined_np,
                        hair_length=hair_length,
                    )
                    return self._dilate_hair_mask_for_length(refined_np, hair_length), "sam2_soft", "segface_priority"

                return self._dilate_hair_mask_for_length(refined_np, hair_length), "sam2", "sam2"

        except Exception as e:
            logger.warning(f"[SDPipeline] SAM2 refine failed, falling back to SegFace: {e}")

        return self._dilate_hair_mask_for_length(base_mask, hair_length), "segface", refine_mode

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """마스크 dilate (경계 확장)"""
        px = self.config.mask_dilate_px
        if px <= 0:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return np.clip(dilated, 0.0, 1.0).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # SD Input Preparation
    # ──────────────────────────────────────────────────────────────────────────

    def _prepare_sd_inputs(
        self,
        img_rgb: np.ndarray,     # H×W×3 RGB
        hair_mask: np.ndarray,   # H×W float32
        mask_edge_suppression: float = 1.0,  # 0.0=엣지 보존, 1.0=마스크 내부 엣지 완전 제거
        canny_suppress_mask: Optional[np.ndarray] = None,  # H×W float32 — 이 영역의 canny edge도 제거
    ) -> Tuple[Image.Image, Image.Image, Image.Image, float, Tuple[int, int]]:
        """
        Letter-box resize → 512×512.

        Args:
            canny_suppress_mask: short/medium에서 사용. 기존 long-hair 영역의 canny edge를
                                 추가로 제거하여 ControlNet이 원본 긴머리 윤곽을 따라가지 않게 함.

        Returns:
            img_512:    PIL RGB 512×512 (full image)
            mask_512:   PIL L  512×512 (흰색=inpaint)
            canny_512:  PIL RGB 512×512 (ControlNet conditioning)
            scale:      resize 비율
            pad:        (pad_left, pad_top) pixels
        """
        H, W = img_rgb.shape[:2]
        scale = SD_SIZE / max(H, W)
        new_w, new_h = int(W * scale), int(H * scale)
        pad_l = (SD_SIZE - new_w) // 2
        pad_t = (SD_SIZE - new_h) // 2

        # ── image letterbox
        img_rs = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((SD_SIZE, SD_SIZE, 3), dtype=np.uint8)
        canvas[pad_t:pad_t + new_h, pad_l:pad_l + new_w] = img_rs

        # ── mask letterbox
        msk_rs = cv2.resize(hair_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
        msk_canvas = np.zeros((SD_SIZE, SD_SIZE), dtype=np.float32)
        msk_canvas[pad_t:pad_t + new_h, pad_l:pad_l + new_w] = msk_rs

        # ── Canny edge
        # 기본(헤어 생성): 마스크 내부 엣지 강하게 제거
        # 배경 복원(fill): 일부 엣지를 남겨 texture/구조 연속성 확보
        gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)
        suppress = float(np.clip(mask_edge_suppression, 0.0, 1.0))
        hair_hard = (msk_canvas > 0.5).astype(np.float32)

        # canny_suppress_mask가 있으면 해당 영역의 edge도 완전 제거
        # → LaMa 잔여 블러 윤곽이 ControlNet에 전달되지 않음
        if canny_suppress_mask is not None:
            sup_rs = cv2.resize(canny_suppress_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
            sup_canvas = np.zeros((SD_SIZE, SD_SIZE), dtype=np.float32)
            sup_canvas[pad_t:pad_t + new_h, pad_l:pad_l + new_w] = sup_rs
            # dilate: 경계 blur 잔여물까지 제거
            k_sup = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            sup_hard = cv2.dilate(
                (sup_canvas > 0.3).astype(np.uint8), k_sup, iterations=1
            ).astype(np.float32)
            # 기존 hair_hard와 합쳐서 최종 suppression 영역
            hair_hard = np.clip(hair_hard + sup_hard, 0.0, 1.0)
            logger.info(
                f"[SDPipeline] canny suppress 확장: "
                f"gen_mask pixels={int((msk_canvas > 0.5).sum())}, "
                f"total suppress pixels={int((hair_hard > 0.5).sum())}"
            )

        canny_f = canny.astype(np.float32) * (1.0 - hair_hard * suppress)
        canny_rgb = cv2.cvtColor(canny_f.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        img_512   = Image.fromarray(canvas)
        mask_512  = Image.fromarray((msk_canvas * 255).astype(np.uint8), mode="L")
        canny_512 = Image.fromarray(canny_rgb)

        return img_512, mask_512, canny_512, scale, (pad_l, pad_t)

    def _crop_face(
        self,
        img_pil: Image.Image,
        face_bbox: Tuple[int, int, int, int],
    ) -> Image.Image:
        """IP-Adapter용 얼굴 crop (얼굴만 — 머리카락 최소화)

        padding을 아래쪽은 넉넉히(턱/목 포함), 위쪽/옆은 최소화(머리카락 제외)
        IP-Adapter가 원본 헤어 스타일을 conditioning하면 숏컷 변환이 안 됨.
        """
        x1, y1, x2, y2 = face_bbox
        W, H = img_pil.size
        bw, bh = x2 - x1, y2 - y1
        # 위/옆은 패딩 최소화(0.05) → 머리카락 포함 억제
        # 아래는 패딩 넉넉히(0.2) → 턱/목 포함 → 얼굴 identity 안정화
        pad_side = int(bw * 0.05)
        pad_top  = int(bh * 0.05)
        pad_bot  = int(bh * 0.20)
        crop = img_pil.crop((
            max(0, x1 - pad_side),
            max(0, y1 - pad_top),
            min(W, x2 + pad_side),
            min(H, y2 + pad_bot),
        ))
        return crop.resize((224, 224), Image.LANCZOS)

from pipeline_sd_components import (
    bind_loading_methods_to_pipeline,
    bind_postprocess_methods_to_pipeline,
    bind_prompt_methods_to_pipeline,
)

bind_loading_methods_to_pipeline(MirrAISDPipeline)
bind_prompt_methods_to_pipeline(MirrAISDPipeline)
bind_postprocess_methods_to_pipeline(MirrAISDPipeline)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True)
    parser.add_argument("--hairstyle", default="wolf cut, layered")
    parser.add_argument("--color",     default="auburn")
    parser.add_argument("--top-k",     type=int, default=3)
    parser.add_argument("--output",    default="./sd_output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    pipe = MirrAISDPipeline()
    pipe.load()

    results = pipe.run(img, args.hairstyle, args.color, args.top_k)

    os.makedirs(args.output, exist_ok=True)
    for r in results:
        path = os.path.join(args.output, f"rank{r.rank}_seed{r.seed}_{r.mask_used}.jpg")
        cv2.imwrite(path, r.image)
        print(f"저장: {path}")
