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

# ── HuggingFace 모델 ID ────────────────────────────────────────────────────────
SD_INPAINT_MODEL_ID   = "runwayml/stable-diffusion-inpainting"
CONTROLNET_MODEL_ID   = "lllyasviel/control_v11p_sd15_canny"
IP_ADAPTER_REPO_ID    = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT     = "ip-adapter-plus-face_sd15.bin"
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
    "undercut", "bowl", "chin length", "chin-length",
    "above ear", "above shoulder", "ear length", "single",
    "단발", "숏컷", "픽시",
])
_MEDIUM_HAIR_KEYWORDS = frozenset([
    "lob", "midi", "medium", "shoulder length", "shoulder-length",
    "collarbone", "clavicle", "mid length", "mid-length",
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
        lora_path: Optional[str] = None,
        lora_scale: Optional[float] = None,
    ) -> List[SDInpaintResult]:
        """
        헤어 스타일 변환 실행.

        Args:
            image:          입력 이미지 (BGR numpy)
            hairstyle_text: 사용자 헤어스타일 텍스트 (런타임에 llm_refined_trends 기반 보강)
            color_text:     헤어 컬러 텍스트
            top_k:          반환 결과 수 (기본 3)
            return_intermediates: 중간 산출물 디버그 이미지 포함 여부

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
        normalized_color_text = self._normalize_color_text(color_text)
        has_color_request = bool(normalized_color_text)
        target_hair_lab = self._resolve_target_hair_lab(normalized_color_text) if has_color_request else None
        if not has_color_request:
            logger.info("[SDPipeline] color_text 미지정 → 원본 머리 톤 유지 모드")
        elif target_hair_lab is None:
            logger.info("[SDPipeline] color_text 파싱 실패 → 색상 재정렬은 스킵")

        H, W = image_bgr.shape[:2]
        debug_images_common: Optional[Dict[str, np.ndarray]] = {} if return_intermediates else None
        debug_data_common: Optional[Dict[str, Any]] = {} if return_intermediates else None
        if debug_data_common is not None and trend_request is not None:
            debug_data_common["trend_resolution"] = trend_request.to_debug_dict()

        def _store_mask(name: str, mask: np.ndarray) -> None:
            if debug_images_common is None:
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
        cloth_mask = self._sanitize_cloth_mask(cloth_mask, hair_mask_base, face_bbox)
        _store_mask("segface_hair_mask", hair_mask_base)
        _store_mask("segface_face_region_mask", face_region_mask)
        _store_mask("segface_cloth_mask", cloth_mask)
        segface_debug = self._last_segface_mask_debug or {}
        custom_hair_mask = segface_debug.get("custom_hair_mask")
        base_hair_mask = segface_debug.get("base_hair_mask")
        base_hair_support_mask = segface_debug.get("base_hair_support_mask")
        glasses_mask = segface_debug.get("glasses_mask")
        if isinstance(custom_hair_mask, np.ndarray):
            _store_mask("segface_custom_hair_mask", custom_hair_mask)
        if isinstance(base_hair_mask, np.ndarray):
            _store_mask("segface_base_hair_mask", base_hair_mask)
        if isinstance(base_hair_support_mask, np.ndarray):
            _store_mask("segface_base_hair_support_mask", base_hair_support_mask)
        if isinstance(glasses_mask, np.ndarray):
            _store_mask("segface_glasses_mask", glasses_mask)
        if debug_data_common is not None and segface_debug.get("meta"):
            debug_data_common["segface_mask_debug"] = segface_debug["meta"]

        # ── Step 3: SAM2 refinement ───────────────────────────────────────────
        hair_mask, mask_source = self._refine_with_sam2(
            img_rgb, hair_mask_base, face_bbox, effective_hairstyle_text
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
        logger.info(f"[SDPipeline] 헤어 길이 분류: {hair_length}")
        if hair_length == "short" and len(seeds) < 5:
            extra = 5 - len(seeds)
            seeds.extend(random.randint(0, 2**31 - 1) for _ in range(extra))
            logger.info(
                f"[SDPipeline] short internal candidate expansion: requested={requested_top_k}, internal={len(seeds)}"
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
        hair_mask = np.clip(hair_mask - cloth_mask_dilated, 0.0, 1.0)
        _store_mask("segface_cloth_mask_dilated", cloth_mask_dilated)
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
            hair_below[:, :max(0, head_x1 - 20)] = 0
            hair_below[:, min(W, head_x2 + 20):] = 0
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
                )
                below_bob_cloth_restore_for_post = self._build_short_below_bob_cloth_restore_mask(
                    removal_mask=removal_mask_for_post,
                    cloth_mask=cloth_mask_dilated,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                    support_mask=lower_tail_support_for_post,
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
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                )
            composite_bangs_release_mask = np.zeros((H, W), dtype=np.float32)
            if float(bangs_restore_for_sd.sum()) > 0.0:
                soft_bangs_generation_mask = self._build_soft_bangs_generation_mask(
                    bangs_restore_for_sd,
                    face_bbox=face_bbox,
                )
                composite_bangs_release_mask = np.clip(
                    soft_bangs_generation_mask.astype(np.float32) * 1.15,
                    0.0,
                    1.0,
                )
                gen_mask = np.maximum(
                    gen_mask,
                    np.clip(soft_bangs_generation_mask.astype(np.float32), 0.0, 1.0),
                )
                _store_mask("pipeline_bangs_generation_soft_mask", soft_bangs_generation_mask)
                _store_mask("pipeline_bangs_composite_release_mask", composite_bangs_release_mask)

            _store_mask("pipeline_short_removal_mask", removal_mask_for_post)
            _store_mask("pipeline_short_generation_mask", gen_mask)
            _store_mask("pipeline_short_below_bob_generation_block_mask", below_bob_generation_block_for_post)
            _store_mask("pipeline_short_below_bob_cloth_restore_mask", below_bob_cloth_restore_for_post)

            logger.info(
                f"[SDPipeline] 전략2: removal_px={removal_mask_for_post.sum():.0f}, "
                f"gen_px={gen_mask.sum():.0f}, cutoff_y={cutoff_y}, "
                f"head_box=({head_x1},{head_y1})-({head_x2},{head_y2})"
            )

            bg_mode = self.config.bg_fill_mode
            logger.info(f"[SDPipeline] bg_fill_mode={bg_mode}")
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
                        img_rgb_cleaned = self._sd_refine_removed_region(
                            base_rgb=img_rgb_cleaned,
                            removal_mask=removal_mask,
                            face_bbox=face_bbox,
                            face_crop_pil=face_crop_fill,
                            protect_mask=protect_mask_for_sd,
                            cloth_mask=cloth_mask_dilated,
                            hair_length=hair_length,
                            seed=fill_seed,
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
                        cloth_mask=cloth_mask_dilated,
                        candidate_mask=preclean_side_candidate_mask,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        hair_length=hair_length,
                        final_hair_mask=None,
                    )
                    direct_preclean_side_restore_mask = self._build_direct_short_column_restore_mask(
                        removal_mask=removal_mask_for_post,
                        cloth_mask=cloth_mask_dilated,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        hair_length=hair_length,
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
                        cloth_mask=cloth_mask_dilated,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y,
                        hair_length=hair_length,
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
                )
                if float(long_soft_bangs_mask.sum()) > 0.0:
                    long_soft_bangs_u8 = cv2.dilate(
                        (np.clip(long_soft_bangs_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13)),
                        iterations=1,
                    )
                    long_soft_bangs_mask = cv2.GaussianBlur(
                        long_soft_bangs_u8.astype(np.float32) / 255.0,
                        (0, 0),
                        sigmaX=2.6,
                        sigmaY=3.0,
                    ).astype(np.float32)
                    long_soft_bangs_mask = np.clip(long_soft_bangs_mask * 1.10, 0.0, 1.0)
                    hair_mask_for_sd = np.maximum(
                        hair_mask_for_sd.astype(np.float32),
                        long_soft_bangs_mask,
                    ).astype(np.float32)
                    composite_bangs_release_mask = np.maximum(
                        composite_bangs_release_mask,
                        np.clip(long_soft_bangs_mask * 1.28, 0.0, 1.0),
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
            effective_hairstyle_text, normalized_color_text, hair_length
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

            candidates.append({
                "seed": seed,
                "image_bgr": composited_bgr,
                "preview_bgr": gen_preview_bgr,
                "color_distance": color_distance,
                "color_score": color_score,
                "tail_penalty": tail_penalty,
                "accessory_penalty": accessory_penalty,
                "gen_idx": gen_idx,
            })

        if has_color_request and target_hair_lab is not None and len(candidates) > 1:
            sortable_count = sum(c["color_distance"] is not None for c in candidates)
            if sortable_count >= 2:
                candidates.sort(
                    key=lambda c: (
                        c["accessory_penalty"] is None,
                        c["accessory_penalty"] if c["accessory_penalty"] is not None else 1e9,
                        c["color_distance"] is None,
                        c["color_distance"] if c["color_distance"] is not None else 1e9,
                        c["gen_idx"],
                    )
                )
                logger.info("[SDPipeline] 컬러 유사도 기준으로 결과 재정렬 완료")
            else:
                logger.info("[SDPipeline] 컬러 유사도 재정렬 스킵 (유효 샘플 부족)")
        elif len(candidates) > 1:
            accessory_sortable = sum(c["accessory_penalty"] is not None for c in candidates)
            if accessory_sortable >= 2:
                candidates.sort(
                    key=lambda c: (
                        c["accessory_penalty"] is None,
                        c["accessory_penalty"] if c["accessory_penalty"] is not None else 1e9,
                        c["gen_idx"],
                    )
                )
                logger.info("[SDPipeline] accessory penalty ranking applied")

        if hair_length == "short" and len(candidates) > 1:
            tail_sortable = sum(c["tail_penalty"] is not None for c in candidates)
            if tail_sortable >= 2:
                candidates.sort(
                    key=lambda c: (
                        c["accessory_penalty"] is None,
                        c["accessory_penalty"] if c["accessory_penalty"] is not None else 1e9,
                        c["tail_penalty"] is None,
                        c["tail_penalty"] if c["tail_penalty"] is not None else 1e9,
                        c["color_distance"] is None,
                        c["color_distance"] if c["color_distance"] is not None else 1e9,
                        c["gen_idx"],
                    )
                )
                logger.info("[SDPipeline] short tail penalty ranking applied")

        results: List[SDInpaintResult] = []
        for rank, cand in enumerate(candidates[:requested_top_k]):
            if debug_images_common is not None and rank == 0:
                debug_images_common["sd_generated_rank0_512"] = cand["preview_bgr"]
            final_bgr = cand["image_bgr"]
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
                    cloth_refine_mask = self._build_post_cloth_refine_mask(
                        removal_mask=removal_mask_for_post,
                        cloth_mask=cloth_mask_dilated,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        protect_mask=protect_mask_for_sd,
                        final_hair_mask=final_hair_mask,
                        artifact_cleanup_mask=artifact_cleanup_mask_for_post,
                    )
                    cloth_refine_u8 = (cloth_refine_mask > 0.08).astype(np.uint8) * 255
                    if int((cloth_refine_u8 > 0).sum()) >= 100:
                        final_rgb = self._sd_refine_removed_region(
                            base_rgb=final_rgb,
                            removal_mask=cloth_refine_mask,
                            face_bbox=face_bbox,
                            face_crop_pil=face_crop_pil,
                            protect_mask=protect_mask_for_sd,
                            cloth_mask=cloth_mask_dilated,
                            hair_length=hair_length,
                            seed=int(cand["seed"]) + 1701,
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
                            reference_mask=cloth_mask_dilated,
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
                try:
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    final_hair_mask, _, _ = self._segface_hair_mask(final_rgb, face_bbox)
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
                    side_column_restore_mask = self._build_side_column_cloth_restore_mask(
                        img_rgb=final_rgb,
                        cloth_mask=cloth_mask_dilated,
                        candidate_mask=side_column_candidate_mask,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
                        final_hair_mask=final_hair_mask,
                    )
                    direct_side_column_restore_mask = self._build_direct_short_column_restore_mask(
                        removal_mask=removal_mask_for_post,
                        cloth_mask=cloth_mask_dilated,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
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
                                refine_mode="cloth",
                            )
                        if hair_length == "short":
                            final_rgb = self._cleanup_region_with_cloth_restore(
                                source_rgb=img_rgb,
                                current_rgb=final_rgb,
                                cleanup_mask=side_column_restore_mask,
                                cloth_mask=cloth_mask_dilated,
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
                        cloth_mask=cloth_mask_dilated,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
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
                        cloth_mask=cloth_mask_dilated,
                        removal_mask=removal_mask_for_post,
                        face_bbox=face_bbox,
                        cutoff_y=cutoff_y_for_post,
                        hair_length=hair_length,
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

    # ──────────────────────────────────────────────────────────────────────────
    # Model Loading
    # ──────────────────────────────────────────────────────────────────────────

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
        candidate_dirs: List[Path] = [
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

    @staticmethod
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

    @staticmethod
    def _as_state_dict(candidate: Any) -> Optional[Dict[str, torch.Tensor]]:
        if not isinstance(candidate, dict) or not candidate:
            return None
        state_dict = {
            str(k): v for k, v in candidate.items()
            if torch.is_tensor(v)
        }
        return state_dict or None

    @staticmethod
    def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {
            (key[len(prefix):] if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }

    @staticmethod
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

    # ──────────────────────────────────────────────────────────────────────────
    # Segmentation: SegFace + SAM2
    # ──────────────────────────────────────────────────────────────────────────

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
        is_landscape = W > H

        if (
            face_h_ratio >= float(self.config.standardize_face_height_ratio_min)
            and face_w_ratio >= float(self.config.standardize_face_width_ratio_min)
            and not is_landscape
        ):
            return meta

        target_w = int(getattr(self.config, "standardized_width", 768))
        target_h = int(getattr(self.config, "standardized_height", 1024))
        target_aspect = target_w / max(float(target_h), 1.0)

        cx = 0.5 * (x1 + x2)
        crop_top = int(round(y1 - face_h * 0.95))
        crop_bottom = int(round(y2 + face_h * 2.15))
        crop_h = max(crop_bottom - crop_top, face_h + 1)
        crop_w = max(int(round(crop_h * target_aspect)), face_w + 1)
        crop_left = int(round(cx - crop_w * 0.5))
        crop_right = crop_left + crop_w

        if crop_left < 0:
            crop_right -= crop_left
            crop_left = 0
        if crop_right > W:
            crop_left -= (crop_right - W)
            crop_right = W
        if crop_left < 0:
            crop_left = 0

        if crop_top < 0:
            crop_bottom -= crop_top
            crop_top = 0
        if crop_bottom > H:
            crop_top -= (crop_bottom - H)
            crop_bottom = H
        if crop_top < 0:
            crop_top = 0

        crop_left = max(0, min(crop_left, W - 1))
        crop_top = max(0, min(crop_top, H - 1))
        crop_right = max(crop_left + 1, min(crop_right, W))
        crop_bottom = max(crop_top + 1, min(crop_bottom, H))

        crop_rgb = img_rgb[crop_top:crop_bottom, crop_left:crop_right]
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
                "is_landscape": bool(is_landscape),
            },
            "crop_box": [int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)],
            "original_shape": [int(H), int(W)],
            "standardized_shape": [int(target_h), int(target_w)],
            "image_rgb": standardized_rgb,
        })
        return meta

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

    def _sanitize_cloth_mask(
        self,
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
            int(forehead_y + face_h * (0.48 if hair_length == "short" else 0.42 if hair_length == "medium" else 0.44)),
        )
        center_half = max(
            18,
            int(face_w * (0.50 if hair_length == "short" else 0.50 if hair_length == "medium" else 0.54)),
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
        support_bottom = min(H, int(forehead_y + face_h * (0.22 if hair_length == "short" else 0.16)))
        support_x1 = max(0, cx - max(24, int(face_w * (0.52 if hair_length == "short" else 0.48))))
        support_x2 = min(W, cx + max(24, int(face_w * (0.52 if hair_length == "short" else 0.48))))
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
        max_component_area = max(160, int(face_w * face_h * (0.36 if hair_length == "short" else 0.30)))
        max_component_width = max(34, int(face_w * (1.02 if hair_length == "short" else 0.92)))
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
                (7, 11) if hair_length == "short" else (7, 11),
            ),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, corridor)
        return (keep_u8 > 0).astype(np.float32)

    def _build_soft_bangs_generation_mask(
        self,
        bangs_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        H, W = bangs_mask.shape[:2]
        base = (np.clip(bangs_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255
        if int((base > 0).sum()) < 8:
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        base = cv2.morphologyEx(
            base,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)),
        )
        base = cv2.dilate(
            base,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)),
            iterations=1,
        )
        ys = np.where(base > 0)[0]
        if ys.size == 0:
            return np.zeros((H, W), dtype=np.float32)

        top = int(ys.min())
        bottom = int(ys.max()) + 1
        cx = int(0.5 * (x1 + x2))
        side_keep = max(18, int(face_w * 0.46))
        x_min = max(0, cx - side_keep)
        x_max = min(W, cx + side_keep)
        band_top = max(0, int(top - face_h * 0.08))
        band_bottom = min(H, int(bottom + face_h * 0.08))
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
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)),
        )
        soft_u8 = cv2.dilate(
            soft_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
            iterations=1,
        )
        if int((soft_u8 > 0).sum()) < 8:
            return np.zeros((H, W), dtype=np.float32)

        alpha = cv2.GaussianBlur(
            soft_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=2.2,
            sigmaY=2.6,
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
        tail_end = min(H, band_bottom + max(6, int(face_h * 0.06)))
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
        x_fade = np.clip(1.0 - (x_dist ** 1.55) * 0.52, 0.44, 1.0).astype(np.float32)
        alpha = alpha * x_fade[np.newaxis, :]
        return np.clip(alpha, 0.0, 0.68).astype(np.float32)

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
            band_bottom = min(H, int(np.max(left_eye_pts[:, 1].max(), right_eye_pts[:, 1].max()) + face_h * 0.06))
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

    # ──────────────────────────────────────────────────────────────────────────
    # Hair Length Classification
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_hair_length(hairstyle_text: str) -> str:
        """헤어스타일 텍스트 → 'short' | 'medium' | 'long'"""
        text = hairstyle_text.lower()
        for kw in _SHORT_HAIR_KEYWORDS:
            if kw in text:
                return "short"
        for kw in _MEDIUM_HAIR_KEYWORDS:
            if kw in text:
                return "medium"
        return "long"

    # ──────────────────────────────────────────────────────────────────────────
    # Color Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_color_text(color_text: str) -> str:
        text = str(color_text or "").strip()
        lowered = text.lower()
        if lowered in _NO_COLOR_HINTS:
            return ""
        return text

    @staticmethod
    def _normalize_hairstyle_prompt_text(hairstyle_text: str, hair_length: str) -> str:
        raw = " ".join(str(hairstyle_text or "").strip().split())
        if not raw:
            return ""
        if hair_length != "short":
            return raw

        lowered = raw.lower()
        hints: List[str] = []
        if "hush" in lowered or "layer" in lowered:
            hints.append("soft internal bob layers above the jawline")
            hints.append("rounded jaw-length bob silhouette")
        if "blunt" in lowered:
            hints.append("clean blunt bob outline")
        if "bang" in lowered or "fringe" in lowered:
            hints.append("soft see-through bangs")
        if any(token in lowered for token in ("wave", "wavy", "curl", "curly")):
            hints.append("light natural texture")
        if any(token in lowered for token in ("straight", "sleek")):
            hints.append("sleek straight finish")
        if "tuck" in lowered:
            hints.append("tucked nape silhouette")
        else:
            hints.append("tucked inward ends at the jawline")
        if any(token in lowered for token in ("bob", "short", "chin")):
            hints.append("clear neckline and shoulders")
            hints.append("no lower side tails below the jawline")

        base_style = "strict short chin-length bob haircut with a compact side silhouette"
        if "pixie" in lowered or "buzz" in lowered:
            base_style = "strict short cropped haircut"

        parts = [base_style]
        for hint in hints:
            if hint not in parts:
                parts.append(hint)
        return ", ".join(parts)

    @staticmethod
    def _resolve_target_hair_lab(color_text: str) -> Optional[np.ndarray]:
        query = str(color_text or "").strip().lower()
        if not query:
            return None
        for keyword, rgb in _HAIR_COLOR_TARGET_RGB:
            if keyword in query:
                rgb_np = np.array([[list(rgb)]], dtype=np.uint8)
                lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
                return lab
        return None

    def _estimate_hair_color_distance(
        self,
        img_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        target_lab: np.ndarray,
    ) -> Optional[float]:
        hair_mask, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
        hair_u8 = (hair_mask > 0.45).astype(np.uint8) * 255
        if int((hair_u8 > 0).sum()) < 80:
            return None

        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        hair_pixels = lab[hair_u8 > 0]
        if hair_pixels.shape[0] < 50:
            return None

        # 극단적인 shadow 영역 영향 완화
        if hair_pixels.shape[0] > 200:
            l_vals = hair_pixels[:, 0]
            keep = l_vals > np.percentile(l_vals, 15.0)
            if np.any(keep):
                hair_pixels = hair_pixels[keep]

        med = np.median(hair_pixels, axis=0)
        d_l = abs(float(med[0] - target_lab[0]))
        d_a = abs(float(med[1] - target_lab[1]))
        d_b = abs(float(med[2] - target_lab[2]))
        # 색조(a,b)를 더 강하게 반영
        return 0.25 * d_l + 0.85 * d_a + 0.85 * d_b

    def _estimate_short_tail_penalty(
        self,
        img_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        removal_mask: np.ndarray,
    ) -> Optional[float]:
        H, W = img_rgb.shape[:2]
        if removal_mask.shape != (H, W):
            return None

        tail_hint = self._build_side_tail_cleanup_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length="short",
        )
        face_w = max(int(face_bbox[2] - face_bbox[0]), 1)
        face_h = max(int(face_bbox[3] - face_bbox[1]), 1)
        deep_start = min(H, int(cutoff_y + face_h * 0.08))
        zone = tail_hint.copy()
        zone[:deep_start, :] = 0.0
        broad_zone = np.zeros((H, W), dtype=np.float32)
        broad_left = max(0, int(face_bbox[0] - face_w * 0.92))
        broad_right = min(W, int(face_bbox[2] + face_w * 0.92))
        broad_bottom = min(H, int(cutoff_y + face_h * 1.14))
        if deep_start < broad_bottom and broad_left < broad_right:
            broad_zone[deep_start:broad_bottom, broad_left:broad_right] = 1.0
            zone = np.maximum(zone, broad_zone * 0.38)
        if float(zone.sum()) < 20.0:
            zone = removal_mask.copy().astype(np.float32)
            zone[:deep_start, :] = 0.0
            if float(broad_zone.sum()) > 0.0:
                zone = np.maximum(zone, broad_zone * 0.38)
        if float(zone.sum()) < 20.0:
            return None

        hair_now, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
        hair_now[:deep_start, :] = 0.0

        zone_bool = zone > 0.08
        if int(zone_bool.sum()) < 20:
            return None

        hair_penalty = float(np.mean(hair_now[zone_bool]))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        darkness = np.clip((118.0 - gray) / 118.0, 0.0, 1.0)
        dark_penalty = float(np.mean(darkness[zone_bool]))
        deep_penalty = 0.0
        deep_zone_bool = broad_zone > 0.0
        if int(deep_zone_bool.sum()) >= 20:
            deep_penalty = float(np.mean(hair_now[deep_zone_bool]))
        return 0.60 * hair_penalty + 0.15 * dark_penalty + 0.25 * deep_penalty

    def _estimate_accessory_penalty(
        self,
        img_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> Optional[float]:
        H, W = img_rgb.shape[:2]
        _ = self._segface_hair_mask(img_rgb, face_bbox)
        segface_debug = self._last_segface_mask_debug or {}
        earring_mask = segface_debug.get("earring_mask")
        necklace_mask = segface_debug.get("necklace_mask")
        if not isinstance(earring_mask, np.ndarray) or not isinstance(necklace_mask, np.ndarray):
            return None
        if earring_mask.shape != (H, W) or necklace_mask.shape != (H, W):
            return None

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(y1 - face_h * 0.16))
        bottom = min(H, int(y2 + face_h * 0.92))
        left = max(0, int(x1 - face_w * 0.92))
        right = min(W, int(x2 + face_w * 0.92))
        if top >= bottom or left >= right:
            return None
        corridor_u8[top:bottom, left:right] = 255

        earring_u8 = cv2.dilate(
            (np.clip(earring_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        necklace_u8 = cv2.dilate(
            (np.clip(necklace_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        earring_u8 = cv2.bitwise_and(earring_u8, corridor_u8)
        necklace_u8 = cv2.bitwise_and(necklace_u8, corridor_u8)

        earring_px = int((earring_u8 > 0).sum())
        necklace_px = int((necklace_u8 > 0).sum())
        if earring_px < 4 and necklace_px < 8:
            return 0.0

        norm = float(max(face_w * face_h, 1))
        earring_penalty = min(float(earring_px) / norm * 28.0, 1.0)
        necklace_penalty = min(float(necklace_px) / norm * 16.0, 1.0)
        return 0.74 * earring_penalty + 0.26 * necklace_penalty

    def _preserve_original_hair_tone(
        self,
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        src_hair, _, _ = self._segface_hair_mask(source_rgb, face_bbox)
        tgt_hair, _, _ = self._segface_hair_mask(target_rgb, face_bbox)

        src_mask = (src_hair > 0.45)
        tgt_mask = (tgt_hair > 0.45)
        if int(src_mask.sum()) < 100 or int(tgt_mask.sum()) < 100:
            return target_rgb

        src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        src_mean = src_lab[src_mask].mean(axis=0)
        tgt_mean = tgt_lab[tgt_mask].mean(axis=0)

        tuned_lab = tgt_lab.copy()
        vals = tuned_lab[tgt_mask]
        vals[:, 1] = np.clip(vals[:, 1] + (src_mean[1] - tgt_mean[1]) * 0.78, 0.0, 255.0)
        vals[:, 2] = np.clip(vals[:, 2] + (src_mean[2] - tgt_mean[2]) * 0.78, 0.0, 255.0)
        vals[:, 0] = np.clip(vals[:, 0] + (src_mean[0] - tgt_mean[0]) * 0.32, 0.0, 255.0)
        tuned_lab[tgt_mask] = vals

        tuned_rgb = cv2.cvtColor(tuned_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        alpha = cv2.GaussianBlur(tgt_hair.astype(np.float32), (0, 0), sigmaX=3.0, sigmaY=3.0)
        alpha = np.clip(alpha * 0.70, 0.0, 1.0)[..., np.newaxis]
        out = tuned_rgb.astype(np.float32) * alpha + target_rgb.astype(np.float32) * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        hairstyle_text: str,
        color_text: str,
        hair_length: str = "long",
    ) -> Tuple[str, str, float]:
        """
        Returns:
            positive_prompt, negative_prompt, guidance_scale
        """
        normalized_color = MirrAISDPipeline._normalize_color_text(color_text)
        normalized_style = MirrAISDPipeline._normalize_hairstyle_prompt_text(
            hairstyle_text,
            hair_length,
        )
        parts = []
        if normalized_style:
            parts.append(normalized_style)
        if normalized_color:
            parts.append(f"{normalized_color.strip()} hair color")
        style = ", ".join(parts) if parts else "natural hairstyle"

        # ── 길이별 positive/negative 보강 ────────────────────────────────────
        if hair_length == "short":
            pos_suffix = (
                ", strict short jaw-length bob silhouette, compact side shape, tucked nape line, "
                "ends stopping at or above the jawline, fully visible neck and shoulders, "
                "hair clearly above the shoulders, no side strands touching clothing, "
                "no shoulder-grazing side sections, no chest-length strands, "
                "no long lower tails below the chin line"
            )
            neg_prefix = (
                "very long hair, medium hair, medium length hair, medium-length hair, shoulder-length hair, "
                "shoulder grazing hair, shoulder-grazing hair, collarbone-length hair, lob, "
                "flowing long hair, hair below shoulders, waist-length hair, side long locks over chest, "
                "long hush cut, long wolf cut, mullet tails, long layers below jawline, "
                "hair touching shoulders, hair covering collar, chest-length strands, neckline covered by hair, "
                "hair below jawline, hair below neckline, dangling lower tails, long side tails, nape tails, "
                "strands touching clothes, side locks on shoulders, hair covering blouse, "
                "overly voluminous hair, puffy hair, oversized bob, wide helmet shape, bulky side volume, "
                "blunt horizontal cut line, helmet hair, bowl-shaped edge, "
            )
            guidance = 11.2
        elif hair_length == "medium":
            pos_suffix = (
                ", medium length hair, shoulder-length hair, "
                "hair just above or at shoulder"
            )
            neg_prefix = "very long hair, very short hair, "
            guidance = 8.5
        else:
            pos_suffix = ""
            neg_prefix = ""
            guidance = 7.5

        color_pos_hint = ""
        color_neg_hint = ""
        lowered_color = normalized_color.lower()
        if "ash" in lowered_color:
            color_pos_hint = ", cool-toned ash color, smoky neutral undertone, no brassiness"
            color_neg_hint = "warm orange cast, yellow brassiness, copper tint, reddish tint, "
        elif normalized_color:
            color_pos_hint = ", consistent natural hair color tone, coherent root-to-end color"

        positive_parts = [
            f"professional portrait photo of a person with {style}{pos_suffix}",
        ]
        if color_pos_hint:
            positive_parts.append(color_pos_hint.lstrip(", ").strip())
        positive_parts.extend([
            "same outfit, preserved shirt or blouse fabric texture, clean neckline and collar continuity, natural sleeve folds",
            "photorealistic, high quality, natural lighting, 8k",
            "studio photography, sharp focus, beautiful hair",
        ])
        positive = ", ".join(positive_parts)
        negative_base = _NEGATIVE_BASE + ", " + _COMMON_STYLE_BLOCK_NEGATIVE
        negative = neg_prefix + color_neg_hint + negative_base

        return positive, negative, guidance

    # ──────────────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate(
        self,
        img_512: Image.Image,
        mask_512: Image.Image,
        canny_512: Image.Image,
        face_crop_pil: Image.Image,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        seeds: List[int],
        hair_length: str = "long",
    ) -> List[Image.Image]:
        """
        모든 seed를 단일 배치 forward pass로 생성 (순차 대비 ~절반 시간).

        diffusers는 generator를 리스트로 받으면 num_images_per_prompt 개의
        이미지를 각자 다른 seed로 한 번의 파이프라인 실행에 처리함.
        """
        # 숏컷/중단발 변환 시 IP-Adapter scale을 낮춤
        # → 원본 긴머리 identity가 생성에 과도하게 영향주는 것 방지
        if hair_length == "short":
            ip_scale = 0.01
            control_scale = min(self.config.controlnet_conditioning_scale, 0.08)
        elif hair_length == "medium":
            ip_scale = 0.18
            control_scale = min(self.config.controlnet_conditioning_scale, 0.20)
        else:
            ip_scale = self.config.ip_adapter_scale  # long은 기본값 유지
            control_scale = self.config.controlnet_conditioning_scale

        self._sd_pipe.set_ip_adapter_scale(ip_scale)
        logger.info(
            f"[SDPipeline] ip_adapter_scale={ip_scale}, "
            f"controlnet_scale={control_scale} (hair_length={hair_length})"
        )

        n = len(seeds)
        generators = [
            torch.Generator(device=self.device).manual_seed(s) for s in seeds
        ]
        logger.info(f"[SDPipeline] 배치 생성 시작 (n={n}, seeds={seeds})")

        with torch.inference_mode():
            out = self._sd_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img_512,
                mask_image=mask_512,
                control_image=canny_512,
                ip_adapter_image=[face_crop_pil],
                height=SD_SIZE,
                width=SD_SIZE,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=control_scale,
                num_images_per_prompt=n,
                generator=generators,
                strength=1.0,
            )

        logger.info(f"[SDPipeline] 배치 생성 완료 → {len(out.images)}장")
        return out.images

    @staticmethod
    def _cv2_refine_cloth_region(
        base_rgb: np.ndarray,
        cloth_refine_mask: np.ndarray,
        reference_rgb: Optional[np.ndarray] = None,
        reference_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Clean residual shirt/blouse blur with a small deterministic inpaint pass.
        The mask is expected to already exclude hair and face-protect regions.
        """
        H, W = base_rgb.shape[:2]
        if cloth_refine_mask.shape != (H, W):
            return base_rgb

        ref_rgb = base_rgb
        if reference_rgb is not None and reference_rgb.shape[:2] == (H, W):
            ref_rgb = reference_rgb

        mask_u8 = (np.clip(cloth_refine_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        if int((mask_u8 > 0).sum()) < 60:
            return base_rgb

        mask_u8 = cv2.dilate(
            mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        ring_u8 = cv2.dilate(
            mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
            iterations=1,
        )
        ring_u8 = cv2.subtract(
            ring_u8,
            cv2.dilate(
                mask_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ),
        )
        telea = cv2.inpaint(base_rgb, mask_u8, 5, cv2.INPAINT_TELEA)
        ns = cv2.inpaint(base_rgb, mask_u8, 4, cv2.INPAINT_NS)
        refined = cv2.addWeighted(telea, 0.66, ns, 0.34, 0.0)
        mask_bool = mask_u8 > 0
        ring_bool = ring_u8 > 0
        if reference_rgb is not None and reference_rgb.shape[:2] == (H, W):
            ref_telea = cv2.inpaint(ref_rgb, mask_u8, 5, cv2.INPAINT_TELEA)
            ref_ns = cv2.inpaint(ref_rgb, mask_u8, 4, cv2.INPAINT_NS)
            ref_refined = cv2.addWeighted(ref_telea, 0.62, ref_ns, 0.38, 0.0)
            refined = cv2.addWeighted(refined, 0.42, ref_refined, 0.58, 0.0)
        if int(ring_bool.sum()) >= 80 and int(mask_bool.sum()) >= 60:
            refined_lab = cv2.cvtColor(refined, cv2.COLOR_RGB2LAB).astype(np.float32)
            ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            ring_ref_bool = ring_bool.copy()
            if reference_mask is not None and reference_mask.shape == (H, W):
                ref_mask_u8 = cv2.dilate(
                    (np.clip(reference_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                    iterations=1,
                )
                masked_ring = np.logical_and(ring_ref_bool, ref_mask_u8 > 0)
                if int(masked_ring.sum()) >= 40:
                    ring_ref_bool = masked_ring
            ring_vals = ref_lab[ring_ref_bool]
            mask_vals = refined_lab[mask_bool]
            ring_mean = ring_vals.mean(axis=0)
            mask_mean = mask_vals.mean(axis=0)
            ring_std = ring_vals.std(axis=0)
            mask_std = np.maximum(mask_vals.std(axis=0), 1.0)
            tone_matched = mask_vals.copy()
            tone_matched[:, 0] = np.clip(
                (tone_matched[:, 0] - mask_mean[0]) * np.clip(ring_std[0] / mask_std[0], 0.82, 1.18)
                + mask_mean[0]
                + np.clip(ring_mean[0] - mask_mean[0], -16.0, 16.0) * 0.72,
                0.0,
                255.0,
            )
            tone_matched[:, 1] = np.clip(
                tone_matched[:, 1] + np.clip(ring_mean[1] - mask_mean[1], -5.0, 5.0) * 0.55,
                0.0,
                255.0,
            )
            tone_matched[:, 2] = np.clip(
                tone_matched[:, 2] + np.clip(ring_mean[2] - mask_mean[2], -5.0, 5.0) * 0.55,
                0.0,
                255.0,
            )
            refined_lab[mask_bool] = tone_matched
            refined = cv2.cvtColor(refined_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

            detail_reference_rgb = ref_rgb
            lowpass = cv2.GaussianBlur(detail_reference_rgb, (0, 0), sigmaX=3.2, sigmaY=3.2)
            detail_src = np.clip(
                detail_reference_rgb.astype(np.float32) - lowpass.astype(np.float32) + 128.0,
                0.0,
                255.0,
            ).astype(np.uint8)
            detail_telea = cv2.inpaint(detail_src, mask_u8, 3, cv2.INPAINT_TELEA)
            detail_ns = cv2.inpaint(detail_src, mask_u8, 3, cv2.INPAINT_NS)
            detail_fill = cv2.addWeighted(detail_telea, 0.70, detail_ns, 0.30, 0.0)
            detail_signed = detail_fill.astype(np.float32) - 128.0

            gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            ring_detail = float(np.mean(np.abs(lap[ring_bool]))) if int(ring_bool.sum()) > 0 else 0.0
            mask_detail = float(np.mean(np.abs(detail_signed[mask_bool]))) if int(mask_bool.sum()) > 0 else 0.0
            texture_gain = float(np.clip(ring_detail / max(mask_detail, 1.0), 0.65, 1.35))
            textured = np.clip(
                refined.astype(np.float32) + detail_signed * (0.46 * texture_gain),
                0.0,
                255.0,
            )
            refined = textured.astype(np.uint8)

        alpha = cv2.GaussianBlur(
            (mask_u8 > 0).astype(np.float32),
            (0, 0),
            sigmaX=2.4,
            sigmaY=2.4,
        )[..., np.newaxis]
        alpha = np.clip(alpha * 0.92, 0.0, 1.0)
        out = refined.astype(np.float32) * alpha + base_rgb.astype(np.float32) * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _overlay_reference_cloth_fill(
        base_rgb: np.ndarray,
        reference_rgb: Optional[np.ndarray],
        fill_mask: np.ndarray,
        cloth_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        H, W = base_rgb.shape[:2]
        if reference_rgb is None or reference_rgb.shape[:2] != (H, W) or fill_mask.shape != (H, W):
            return base_rgb

        mask_u8 = (np.clip(fill_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_u8 = cv2.dilate(
                (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                iterations=1,
            )
            mask_u8 = cv2.bitwise_and(mask_u8, cloth_u8)
        if int((mask_u8 > 0).sum()) < 40:
            return base_rgb

        base_gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        ref_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        base_blur = cv2.GaussianBlur(base_gray, (0, 0), sigmaX=4.8, sigmaY=4.8)
        ref_blur = cv2.GaussianBlur(ref_gray, (0, 0), sigmaX=4.8, sigmaY=4.8)
        base_sat = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
        blackhat = cv2.morphologyEx(
            base_gray.astype(np.uint8),
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 21)),
        )

        residual_u8 = (
            (
                (
                    (ref_blur - base_blur > 12.0)
                    & (ref_gray - base_gray > 10.0)
                    & (base_sat < 142.0)
                )
                | (
                    (base_gray < 138.0)
                    & (ref_gray > 160.0)
                    & (blackhat > 8)
                )
            ).astype(np.uint8)
            * 255
        )
        residual_u8 = cv2.bitwise_and(residual_u8, mask_u8)
        if int((residual_u8 > 0).sum()) < 28:
            return base_rgb

        residual_u8 = cv2.morphologyEx(
            residual_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        residual_u8 = cv2.morphologyEx(
            residual_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
        )
        residual_u8 = cv2.dilate(
            residual_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
            iterations=1,
        )
        if int((residual_u8 > 0).sum()) < 28:
            return base_rgb

        alpha = cv2.GaussianBlur(
            (residual_u8 > 0).astype(np.float32),
            (0, 0),
            sigmaX=3.0,
            sigmaY=3.0,
        )[..., np.newaxis]
        delta = np.clip((ref_gray - base_gray) / 52.0, 0.0, 1.0)[..., np.newaxis]
        alpha = np.clip(alpha * (0.56 + 0.34 * delta), 0.0, 0.96)
        out = reference_rgb.astype(np.float32) * alpha + base_rgb.astype(np.float32) * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _blend_neighbor_cloth_tone(
        base_rgb: np.ndarray,
        fill_mask: np.ndarray,
        cloth_mask: Optional[np.ndarray] = None,
        reference_rgb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        H, W = base_rgb.shape[:2]
        if fill_mask.shape != (H, W):
            return base_rgb
        ref_rgb = base_rgb
        if reference_rgb is not None and reference_rgb.shape[:2] == (H, W):
            ref_rgb = reference_rgb

        mask_u8 = (np.clip(fill_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        if int((mask_u8 > 0).sum()) < 60:
            return base_rgb

        ring_u8 = cv2.dilate(
            mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
            iterations=1,
        )
        ring_u8 = cv2.subtract(
            ring_u8,
            cv2.dilate(
                mask_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            ),
        )
        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_u8 = cv2.dilate(
                (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                iterations=1,
            )
            ring_u8 = cv2.bitwise_and(ring_u8, cloth_u8)
        if int((ring_u8 > 0).sum()) < 80:
            return base_rgb

        hsv = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        bright_u8 = ((gray > 152.0) & (sat < 118.0)).astype(np.uint8) * 255
        bright_ring_u8 = cv2.bitwise_and(ring_u8, bright_u8)
        sample_u8 = bright_ring_u8 if int((bright_ring_u8 > 0).sum()) >= 60 else ring_u8

        sample_pixels = ref_rgb[sample_u8 > 0]
        if sample_pixels.size == 0:
            return base_rgb
        fill_rgb = np.median(sample_pixels, axis=0).astype(np.float32)

        filled = base_rgb.astype(np.float32).copy()
        filled[mask_u8 > 0] = fill_rgb
        alpha = cv2.GaussianBlur(
            (mask_u8 > 0).astype(np.float32),
            (0, 0),
            sigmaX=4.2,
            sigmaY=4.2,
        )[..., np.newaxis]
        alpha = np.clip(alpha * 0.94, 0.0, 1.0)
        out = filled * alpha + base_rgb.astype(np.float32) * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _sd_refine_removed_region(
        self,
        base_rgb: np.ndarray,          # H×W×3 RGB (cv2 inpaint 1차 결과)
        removal_mask: np.ndarray,      # H×W float32 (긴머리 제거 영역)
        face_bbox: Tuple[int, int, int, int],
        face_crop_pil: Image.Image,    # IP-Adapter conditioning face
        protect_mask: Optional[np.ndarray],  # H×W float32 (얼굴 보호)
        cloth_mask: Optional[np.ndarray],    # H×W float32 (의상 영역)
        hair_length: str,
        seed: int,
        refine_mode: str = "generic",
    ) -> np.ndarray:
        """
        긴머리 제거 후 남는 어색한 영역(목/어깨/배경)을 SD로 한 번 더 정리.
        """
        H, W = base_rgb.shape[:2]
        removal_mask = self._resize_mask_to_shape(removal_mask, (H, W))
        protect_mask = self._resize_mask_to_shape(protect_mask, (H, W))
        cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
        if removal_mask.shape != (H, W):
            raise ValueError(f"removal_mask shape mismatch: {removal_mask.shape} vs {(H, W)}")

        # removal 영역 중심으로만 SD를 적용하기 위해 그대로 letterbox 변환
        fill_mask = (removal_mask > 0.5).astype(np.float32)
        img_512, mask_512, canny_512, scale, pad = self._prepare_sd_inputs(
            base_rgb,
            fill_mask,
            mask_edge_suppression=0.45,
        )

        if refine_mode == "cloth":
            fill_prompt = (
                "professional portrait photo, preserve the original shirt or blouse shape, "
                "realistic clothing fabric texture continuity, coherent folds and seams, "
                "clean neck and shoulders, no hair strands in masked region, photorealistic details"
            )
            fill_guidance = 6.8 if hair_length == "short" else 7.0
            fill_negative = (
                "hair strands, loose dangling hair, long hair, blur, blurry cloth, smudged cloth, "
                "melted fabric, duplicate collar, broken neckline, extra folds, extra buttons, "
                "warped shirt, warped blouse, deformed neck, artifacts, cartoon, painting, "
                f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
            )
        elif refine_mode == "short_tail" and hair_length == "short":
            fill_prompt = (
                "professional portrait photo, neat compact short jaw-length bob haircut, "
                "clean side silhouette above the shoulders, visible neck and shoulders, "
                "same shirt or blouse preserved, realistic clothing fabric texture continuity, "
                "clean neckline, no hair below jawline, no shoulder-length side hair, "
                "no dangling strands in masked region, photorealistic details"
            )
            fill_guidance = 8.2
            fill_negative = (
                "long hair, shoulder-length hair, medium hair, lob haircut, hair below jawline, "
                "hair touching shoulders, dangling side tails, loose strands, extra hair mass, "
                "warped shirt, warped blouse, melted fabric, deformed neck, artifacts, blurry, "
                "smudged texture, cartoon, painting, "
                f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
            )
        elif hair_length == "short":
            fill_prompt = (
                "professional portrait photo, clean natural neck and shoulders, "
                "same shirt or blouse preserved, realistic clothing fabric texture continuity, "
                "coherent neckline, collar and sleeve folds, coherent background, "
                "short-hair silhouette maintained, no long hair below jawline, "
                "no loose dangling strands in masked region, photorealistic details"
            )
            fill_guidance = 7.1
            fill_negative = (
                "long hair, hair below chin, hair below shoulders, loose hair strands, "
                "wavy hair, straight long hair, wig, ponytail, braid, bangs, side locks, "
                "deformed neck, artifacts, blurry, smudged texture, melted details, cartoon, painting, "
                f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
            )
        else:
            fill_prompt = (
                "professional portrait photo, clean neck and shoulders, "
                "natural skin and clothing texture continuity, coherent background, "
                "no loose long hair strands in masked region, photorealistic details"
            )
            fill_guidance = 7.6
            fill_negative = (
                "long hair, hair below chin, hair below shoulders, loose hair strands, "
                "wavy hair, straight long hair, wig, ponytail, braid, bangs, side locks, "
                "deformed neck, artifacts, blurry, smudged texture, melted details, cartoon, painting, "
                f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
            )

        # 배경 복원은 identity 영향이 과하면 긴머리가 다시 생길 수 있어 scale을 낮춘다.
        self._sd_pipe.set_ip_adapter_scale(0.0)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        if refine_mode == "cloth":
            fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.16), 0.10, 0.24))
            fill_steps = max(20, self.config.num_inference_steps - 8)
            fill_strength = 0.84
        elif refine_mode == "short_tail" and hair_length == "short":
            fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.14), 0.10, 0.20))
            fill_steps = max(22, self.config.num_inference_steps - 6)
            fill_strength = 0.90
        else:
            fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.18), 0.12, 0.30))
            fill_steps = max(24, self.config.num_inference_steps - 4)
            fill_strength = 0.88

        with torch.inference_mode():
            out = self._sd_pipe(
                prompt=fill_prompt,
                negative_prompt=fill_negative,
                image=img_512,
                mask_image=mask_512,
                control_image=canny_512,
                ip_adapter_image=[face_crop_pil],
                height=SD_SIZE,
                width=SD_SIZE,
                num_inference_steps=fill_steps,
                guidance_scale=fill_guidance,
                controlnet_conditioning_scale=fill_control,
                num_images_per_prompt=1,
                generator=generator,
                strength=fill_strength,
            )

        gen_np = np.array(out.images[0])  # 512×512 RGB

        # letterbox 역변환
        pad_l, pad_t = pad
        new_w = int(W * scale)
        new_h = int(H * scale)
        gen_cropped = gen_np[pad_t:pad_t + new_h, pad_l:pad_l + new_w]
        gen_orig = cv2.resize(gen_cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

        alpha = cv2.GaussianBlur(fill_mask, (0, 0), sigmaX=7.0, sigmaY=7.0)
        alpha = np.clip(alpha, 0.0, 1.0)

        # 중앙 편향을 완화해 side 잔존 영역도 자연스럽게 복원한다.
        x1, y1, x2, y2 = face_bbox
        cx = 0.5 * (x1 + x2)
        face_w = max(float(x2 - x1), 1.0)
        sigma_x = max(face_w * 1.45, 44.0)
        xs = np.arange(W, dtype=np.float32)
        center_weight = np.exp(-0.5 * ((xs - cx) / sigma_x) ** 2)
        alpha = alpha * (0.65 + 0.35 * center_weight[np.newaxis, :])

        # 의상 영역은 과도한 hallucination을 줄이기 위해 SD 블렌딩 가중치를 낮춘다.
        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_w = np.clip(cloth_mask.astype(np.float32), 0.0, 1.0)
            alpha = alpha * (1.0 - 0.18 * cloth_w)

        # 얼굴은 기존 픽셀 고정
        if refine_mode == "cloth":
            alpha = cv2.GaussianBlur(fill_mask, (0, 0), sigmaX=6.0, sigmaY=6.0)
            alpha = np.clip(alpha, 0.0, 1.0)
            if cloth_mask is not None and cloth_mask.shape == (H, W):
                cloth_w = np.clip(cloth_mask.astype(np.float32), 0.0, 1.0)
                alpha = np.clip(alpha * (0.94 + 0.18 * cloth_w), 0.0, 1.0)

        if protect_mask is not None:
            protect_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            protect = cv2.dilate(protect_mask.astype(np.float32), protect_k)
            alpha = alpha * (1.0 - np.clip(protect, 0.0, 1.0))

        alpha = alpha[..., np.newaxis]
        refined = (
            gen_orig.astype(np.float32) * alpha
            + base_rgb.astype(np.float32) * (1.0 - alpha)
        )
        return np.clip(refined, 0, 255).astype(np.uint8)

    def _filter_short_center_cleanup_mask(
        self,
        mask_u8: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        *,
        anchor_u8: Optional[np.ndarray] = None,
        top_scale: float = 0.02,
        bottom_scale: float = 0.82,
        half_w_scale: float = 0.24,
        shrink_half_w_scale: float = 0.18,
        max_area_scale: float = 0.10,
        max_width_scale: float = 0.34,
        min_height_scale: float = 0.10,
        center_allow_scale: float = 0.18,
        max_total_scale: float = 0.05,
    ) -> np.ndarray:
        H, W = mask_u8.shape[:2]
        filtered_u8 = (mask_u8 > 0).astype(np.uint8) * 255
        if int((filtered_u8 > 0).sum()) == 0:
            return np.zeros((H, W), dtype=np.uint8)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        gate_u8 = np.zeros((H, W), dtype=np.uint8)
        gate_half_w = max(14, int(face_w * half_w_scale))
        gate_left = max(0, cx - gate_half_w)
        gate_right = min(W, cx + gate_half_w)
        gate_top = max(0, int(cutoff_y + face_h * top_scale))
        gate_bottom = min(H, int(cutoff_y + face_h * bottom_scale))
        if gate_top >= gate_bottom or gate_left >= gate_right:
            return np.zeros((H, W), dtype=np.uint8)
        gate_u8[gate_top:gate_bottom, gate_left:gate_right] = 255
        filtered_u8 = cv2.bitwise_and(filtered_u8, gate_u8)
        if int((filtered_u8 > 0).sum()) == 0:
            return np.zeros((H, W), dtype=np.uint8)

        anchor_local_u8 = np.zeros((H, W), dtype=np.uint8)
        if anchor_u8 is not None and anchor_u8.shape == (H, W):
            anchor_local_u8 = cv2.bitwise_and((anchor_u8 > 0).astype(np.uint8) * 255, gate_u8)

        max_component_area = max(96, int(face_w * face_h * max_area_scale))
        max_component_width = max(24, int(face_w * max_width_scale))
        min_component_height = max(12, int(face_h * min_height_scale))
        center_allow = max(14, int(face_w * center_allow_scale))
        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filtered_u8, 8)
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            comp_u8 = (labels == idx).astype(np.uint8) * 255
            comp_cx = x + (w * 0.5)
            if area < 14 or area > max_component_area:
                continue
            if w > max_component_width or h < min_component_height:
                continue
            if (y + h) > gate_bottom:
                continue
            if abs(comp_cx - cx) > center_allow:
                continue
            if w > max(20, int(face_w * 0.28)) and h < max(20, int(face_h * 0.22)):
                continue
            if int((anchor_local_u8 > 0).sum()) > 0:
                anchor_overlap = int((cv2.bitwise_and(comp_u8, anchor_local_u8) > 0).sum())
                if anchor_overlap < max(4, int(area * 0.04)):
                    continue
            keep_u8 = cv2.bitwise_or(keep_u8, comp_u8)

        if int((keep_u8 > 0).sum()) == 0:
            return np.zeros((H, W), dtype=np.uint8)

        max_total_px = max(72, int(face_w * face_h * max_total_scale))
        current_px = int((keep_u8 > 0).sum())
        if current_px > max_total_px:
            shrink_u8 = np.zeros((H, W), dtype=np.uint8)
            shrink_half_w = max(12, int(face_w * shrink_half_w_scale))
            shrink_left = max(0, cx - shrink_half_w)
            shrink_right = min(W, cx + shrink_half_w)
            shrink_top = max(gate_top, int(cutoff_y + face_h * max(top_scale, 0.04)))
            shrink_bottom = min(H, int(cutoff_y + face_h * max(bottom_scale - 0.06, 0.10)))
            if shrink_top < shrink_bottom and shrink_left < shrink_right:
                shrink_u8[shrink_top:shrink_bottom, shrink_left:shrink_right] = 255
                keep_u8 = cv2.bitwise_and(keep_u8, shrink_u8)

        return keep_u8

    def _build_post_cloth_refine_mask(
        self,
        removal_mask: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
        protect_mask: Optional[np.ndarray] = None,
        final_hair_mask: Optional[np.ndarray] = None,
        artifact_cleanup_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        H, W = removal_mask.shape[:2]
        if cloth_mask is None or cloth_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255
        overlap_u8 = cv2.bitwise_and(removal_u8, cloth_u8)
        if int((overlap_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        overlap_u8 = cv2.dilate(
            overlap_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 15) if hair_length == "short" else (11, 19),
            ),
            iterations=1,
        )

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(cutoff_y + face_h * (0.05 if hair_length == "short" else 0.05)))
        bottom = min(H, int(cutoff_y + face_h * (0.54 if hair_length == "short" else 1.52)))
        left = max(0, int(x1 - face_w * (0.72 if hair_length == "short" else 1.18)))
        right = min(W, int(x2 + face_w * (0.72 if hair_length == "short" else 1.18)))
        if top < bottom and left < right:
            corridor_u8[top:bottom, left:right] = 255

        mask_u8 = cv2.bitwise_and(overlap_u8, corridor_u8)
        mask_u8 = cv2.morphologyEx(
            mask_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 11) if hair_length == "short" else (9, 13),
            ),
        )

        artifact_bonus_u8 = np.zeros((H, W), dtype=np.uint8)
        if artifact_cleanup_mask is not None and artifact_cleanup_mask.shape == (H, W):
            artifact_bonus_u8 = (
                (np.clip(artifact_cleanup_mask.astype(np.float32), 0.0, 1.0) > 0.18).astype(np.uint8) * 255
            )
            artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, corridor_u8)
            artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, cloth_u8)
            artifact_bonus_u8 = cv2.dilate(
                artifact_bonus_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
                iterations=1,
            )

        if protect_mask is not None and protect_mask.shape == (H, W):
            protect_u8 = cv2.dilate(
                (np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                iterations=1,
            )
            mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(protect_u8))
            artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, cv2.bitwise_not(protect_u8))

        if final_hair_mask is not None and final_hair_mask.shape == (H, W):
            final_hair_u8 = cv2.dilate(
                (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
                iterations=1,
            )
            mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(final_hair_u8))
            artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, cv2.bitwise_not(final_hair_u8))

        if hair_length == "short":
            short_gate_u8 = np.zeros((H, W), dtype=np.uint8)
            gate_half_w = max(14, int(face_w * 0.22))
            gate_left = max(0, cx - gate_half_w)
            gate_right = min(W, cx + gate_half_w)
            gate_top = max(top, int(cutoff_y + face_h * 0.04))
            gate_bottom = min(H, int(cutoff_y + face_h * 0.76))
            if gate_top < gate_bottom and gate_left < gate_right:
                short_gate_u8[gate_top:gate_bottom, gate_left:gate_right] = 255
            if int((short_gate_u8 > 0).sum()) > 0:
                mask_u8 = cv2.bitwise_and(mask_u8, short_gate_u8)
                artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, short_gate_u8)

            max_bottom = min(H, int(cutoff_y + face_h * 0.74))
            if max_bottom < H:
                mask_u8[max_bottom:, :] = 0
                artifact_bonus_u8[max_bottom:, :] = 0

            max_component_area = max(96, int(face_w * face_h * 0.08))
            max_component_width = max(24, int(face_w * 0.34))
            min_component_height = max(12, int(face_h * 0.10))
            center_allow = max(14, int(face_w * 0.18))
            filtered_u8 = np.zeros((H, W), dtype=np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
            for idx in range(1, num_labels):
                x = int(stats[idx, cv2.CC_STAT_LEFT])
                y = int(stats[idx, cv2.CC_STAT_TOP])
                w = int(stats[idx, cv2.CC_STAT_WIDTH])
                h = int(stats[idx, cv2.CC_STAT_HEIGHT])
                area = int(stats[idx, cv2.CC_STAT_AREA])
                comp_u8 = (labels == idx).astype(np.uint8) * 255
                comp_cx = x + (w * 0.5)
                artifact_overlap = int((cv2.bitwise_and(comp_u8, artifact_bonus_u8) > 0).sum())
                if area < 14 or area > max_component_area:
                    continue
                if w > max_component_width or h < min_component_height:
                    continue
                if (y + h) > max_bottom:
                    continue
                if abs(comp_cx - cx) > center_allow:
                    continue
                if artifact_overlap < max(4, int(area * 0.04)):
                    continue
                if w > max(20, int(face_w * 0.26)) and h < max(20, int(face_h * 0.22)):
                    continue
                filtered_u8 = cv2.bitwise_or(filtered_u8, comp_u8)
            mask_u8 = filtered_u8

        if int((artifact_bonus_u8 > 0).sum()) > 0:
            if hair_length == "short":
                artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, mask_u8)
            mask_u8 = cv2.bitwise_or(mask_u8, artifact_bonus_u8)
            mask_u8 = cv2.morphologyEx(
                mask_u8,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
            )
            if hair_length == "short":
                max_total_px = max(72, int(face_w * face_h * 0.05))
                current_px = int((mask_u8 > 0).sum())
                if current_px > max_total_px:
                    shrink_u8 = np.zeros((H, W), dtype=np.uint8)
                    shrink_half_w = max(12, int(face_w * 0.18))
                    shrink_left = max(0, cx - shrink_half_w)
                    shrink_right = min(W, cx + shrink_half_w)
                    shrink_top = max(top, int(cutoff_y + face_h * 0.06))
                    shrink_bottom = min(H, int(cutoff_y + face_h * 0.72))
                    if shrink_top < shrink_bottom and shrink_left < shrink_right:
                        shrink_u8[shrink_top:shrink_bottom, shrink_left:shrink_right] = 255
                    mask_u8 = cv2.bitwise_and(mask_u8, shrink_u8)

        if int((mask_u8 > 0).sum()) < 60:
            return np.zeros((H, W), dtype=np.float32)
        return (mask_u8 > 0).astype(np.float32)

    def _build_generation_protect_mask(
        self,
        protect_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        hair_length: str,
    ) -> np.ndarray:
        """
        SD 생성/합성에 사용할 얼굴 보호 마스크.
        short/medium 헤어에서는 목까지 네모나게 막히면 bob 라인이 끊겨 보여서,
        턱 아래는 빠르게 감쇠시키고 중앙 목 부분만 좁게 남긴다.
        """
        mask = np.clip(protect_mask.astype(np.float32), 0.0, 1.0).copy()
        if hair_length not in ("short", "medium"):
            return mask

        H, W = mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        fade_start = max(0, int(y2 - face_h * 0.04))
        fade_len = max(10, int(face_h * (0.22 if hair_length == "short" else 0.28)))
        fade_end = min(H, fade_start + fade_len)
        if fade_end > fade_start:
            ramp = np.linspace(1.0, 0.0, fade_end - fade_start, dtype=np.float32)
            mask[fade_start:fade_end, :] *= ramp[:, np.newaxis]
        if fade_end < H:
            mask[fade_end:, :] = 0.0

        neck_half = max(18, int(face_w * (0.32 if hair_length == "short" else 0.38)))
        neck_x1 = max(0, cx - neck_half)
        neck_x2 = min(W, cx + neck_half)
        neck_y1 = max(0, int(y2 - face_h * 0.02))
        neck_y2 = min(H, int(y2 + face_h * (0.18 if hair_length == "short" else 0.26)))
        if neck_x1 < neck_x2 and neck_y1 < neck_y2:
            neck_guard_u8 = np.zeros((H, W), dtype=np.uint8)
            guard_center = (
                cx,
                int(y2 + face_h * (0.05 if hair_length == "short" else 0.08)),
            )
            guard_axes = (
                max(10, int(face_w * (0.18 if hair_length == "short" else 0.24))),
                max(8, int(face_h * (0.09 if hair_length == "short" else 0.13))),
            )
            cv2.ellipse(neck_guard_u8, guard_center, guard_axes, 0, 0, 360, 255, -1)
            neck_guard = cv2.GaussianBlur(
                neck_guard_u8.astype(np.float32) / 255.0,
                (0, 0),
                sigmaX=4.5,
                sigmaY=4.5,
            )
            neck_guard[:neck_y1, :] = 0.0
            if neck_y2 > neck_y1:
                ramp = np.linspace(1.0, 0.0, neck_y2 - neck_y1, dtype=np.float32)
                neck_guard[neck_y1:neck_y2, :] *= ramp[:, np.newaxis]
            if neck_y2 < H:
                neck_guard[neck_y2:, :] = 0.0
            mask = np.maximum(
                mask,
                neck_guard * (0.58 if hair_length == "short" else 0.50),
            )

        if hair_length == "short":
            erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.erode(mask, erode_k, iterations=1)

        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    def _build_removal_protect_mask(
        self,
        protect_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        hair_length: str,
    ) -> np.ndarray:
        """
        긴 머리 제거(pre-clean) 단계에서 사용할 얼굴 보호 마스크.
        생성 단계보다 목 중앙 보호를 훨씬 약하게 두어, 목 앞쪽으로 내려온 머리 가닥은
        제거 대상으로 남기고 얼굴/턱 주변만 보수적으로 보호한다.
        """
        mask = np.clip(protect_mask.astype(np.float32), 0.0, 1.0).copy()
        if hair_length not in ("short", "medium"):
            return mask

        H, W = mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        face_core_u8 = np.zeros((H, W), dtype=np.uint8)
        face_core_center = (
            cx,
            int(y1 + face_h * (0.44 if hair_length == "short" else 0.47)),
        )
        face_core_axes = (
            max(16, int(face_w * (0.44 if hair_length == "short" else 0.48))),
            max(18, int(face_h * (0.56 if hair_length == "short" else 0.60))),
        )
        cv2.ellipse(face_core_u8, face_core_center, face_core_axes, 0, 0, 360, 255, -1)
        face_core = cv2.GaussianBlur(
            face_core_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=4.0,
            sigmaY=4.0,
        )
        mask = np.minimum(mask, np.clip(face_core * 1.15, 0.0, 1.0))

        fade_start = max(0, int(y2 - face_h * 0.10))
        fade_end = min(H, int(y2 + face_h * (0.005 if hair_length == "short" else 0.02)))
        if fade_end > fade_start:
            ramp = np.linspace(1.0, 0.0, fade_end - fade_start, dtype=np.float32)
            mask[fade_start:fade_end, :] *= ramp[:, np.newaxis]
        cutoff_y = min(H, int(y2 + face_h * (0.015 if hair_length == "short" else 0.03)))
        if cutoff_y < H:
            mask[cutoff_y:, :] = 0.0

        neck_cut_u8 = np.zeros((H, W), dtype=np.uint8)
        neck_cut_center = (
            cx,
            int(y2 + face_h * (0.04 if hair_length == "short" else 0.06)),
        )
        neck_cut_axes = (
            max(12, int(face_w * (0.18 if hair_length == "short" else 0.22))),
            max(10, int(face_h * (0.12 if hair_length == "short" else 0.16))),
        )
        cv2.ellipse(neck_cut_u8, neck_cut_center, neck_cut_axes, 0, 0, 360, 255, -1)
        neck_cut = cv2.GaussianBlur(
            neck_cut_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=3.6,
            sigmaY=3.6,
        )
        mask = np.clip(mask - neck_cut, 0.0, 1.0)

        erode_k = (5, 5) if hair_length == "short" else (7, 7)
        mask = cv2.erode(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erode_k),
            iterations=1,
        )

        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    def _build_neckline_preserve_mask(
        self,
        face_bbox: Tuple[int, int, int, int],
        cloth_mask: Optional[np.ndarray],
        hair_length: str,
    ) -> np.ndarray:
        """
        목 중앙과 상의 neckline을 보존하기 위한 소프트 마스크.
        short hair에서 턱 아래 피부/옷 경계가 네모나게 끊기는 현상을 줄인다.
        """
        if cloth_mask is None:
            return np.zeros((1, 1), dtype=np.float32)

        H, W = cloth_mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        mask = np.zeros((H, W), dtype=np.uint8)
        neck_half = max(16, int(face_w * (0.19 if hair_length == "short" else 0.25)))
        x_left = max(0, cx - neck_half)
        x_right = min(W, cx + neck_half)
        y_top = max(0, int(y2 - face_h * 0.01))
        y_mid = min(H, int(y2 + face_h * (0.11 if hair_length == "short" else 0.16)))
        y_bottom = min(H, int(y2 + face_h * (0.23 if hair_length == "short" else 0.32)))

        if x_left < x_right and y_top < y_bottom:
            neck_poly = np.array(
                [
                    [cx - max(8, int(neck_half * 0.48)), y_top],
                    [cx + max(8, int(neck_half * 0.48)), y_top],
                    [cx + max(10, int(neck_half * 0.88)), y_mid],
                    [cx + max(8, int(neck_half * 0.72)), y_bottom],
                    [cx - max(8, int(neck_half * 0.72)), y_bottom],
                    [cx - max(10, int(neck_half * 0.88)), y_mid],
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(mask, neck_poly, 255)
            ellipse_center = (
                cx,
                int(y2 + face_h * (0.07 if hair_length == "short" else 0.10)),
            )
            ellipse_axes = (
                max(10, int(neck_half * 0.85)),
                max(8, int(face_h * (0.10 if hair_length == "short" else 0.14))),
            )
            cv2.ellipse(mask, ellipse_center, ellipse_axes, 0, 0, 360, 255, -1)

        if cloth_mask.shape == (H, W):
            cloth_u8 = (np.clip(cloth_mask, 0.0, 1.0) > 0.22).astype(np.uint8) * 255
            cloth_u8 = cv2.dilate(
                cloth_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(cloth_u8))

        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=7.0, sigmaY=7.0)
        if hair_length == "short":
            mask_f = np.clip(mask_f * 0.74, 0.0, 1.0)
        return np.clip(mask_f, 0.0, 1.0).astype(np.float32)

    def _build_short_lateral_neck_preserve_mask(
        self,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        cloth_mask: Optional[np.ndarray],
        hair_length: str,
    ) -> np.ndarray:
        if hair_length != "short" or cloth_mask is None:
            return np.zeros((1, 1), dtype=np.float32)

        H, W = cloth_mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        mask_u8 = np.zeros((H, W), dtype=np.uint8)
        centers = (
            (
                int(cx - face_w * 0.44),
                int(y2 + face_h * 0.11),
            ),
            (
                int(cx + face_w * 0.44),
                int(y2 + face_h * 0.11),
            ),
        )
        axes = (
            max(16, int(face_w * 0.26)),
            max(14, int(face_h * 0.16)),
        )
        for center in centers:
            cv2.ellipse(mask_u8, center, axes, 0, 0, 360, 255, -1)

        band_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(y2 - face_h * 0.03))
        bottom = min(H, int(cutoff_y + face_h * 0.26))
        left = max(0, int(x1 - face_w * 0.36))
        right = min(W, int(x2 + face_w * 0.36))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        band_u8[top:bottom, left:right] = 255
        mask_u8 = cv2.bitwise_and(mask_u8, band_u8)

        center_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
        center_half = max(14, int(face_w * 0.18))
        center_keepout_u8[
            top:bottom,
            max(0, cx - center_half):min(W, cx + center_half),
        ] = 255
        mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(center_keepout_u8))

        if int((mask_u8 > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)

        mask_f = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (0, 0), sigmaX=5.0, sigmaY=5.0)
        return np.clip(mask_f * 0.92, 0.0, 1.0).astype(np.float32)

    def _build_shoulder_protect_mask(
        self,
        cloth_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
    ) -> np.ndarray:
        """
        어깨선(옷 상단 경계) 보호 마스크 생성.
        short/medium 후처리에서 어깨 라인 훼손을 줄이기 위해 사용한다.
        """
        H, W = cloth_mask.shape[:2]
        if cloth_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        cloth_u8 = (cloth_mask > 0.35).astype(np.uint8) * 255
        if int((cloth_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        y_start = max(0, int(cutoff_y - face_h * 0.08))
        y_end = min(H, int(cutoff_y + face_h * 1.05))
        band = np.zeros((H, W), dtype=np.uint8)
        band[y_start:y_end, :] = cloth_u8[y_start:y_end, :]
        if int((band > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)

        edge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge = cv2.morphologyEx(band, cv2.MORPH_GRADIENT, edge_k)
        edge = cv2.dilate(
            edge,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )

        x_min = max(0, int(x1 - face_w * 1.30))
        x_max = min(W, int(x2 + face_w * 1.30))
        if x_min >= x_max:
            return np.zeros((H, W), dtype=np.float32)

        corridor = np.zeros((H, W), dtype=np.uint8)
        corridor[:, x_min:x_max] = 255
        edge = cv2.bitwise_and(edge, corridor)

        center_half = max(18, int(face_w * 0.45))
        center_zone = np.zeros((H, W), dtype=np.uint8)
        center_zone[:, max(0, cx - center_half):min(W, cx + center_half)] = 255
        side_edge = cv2.bitwise_and(edge, cv2.bitwise_not(center_zone))
        if int((side_edge > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)

        side_edge = cv2.GaussianBlur(
            side_edge.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=3.0,
            sigmaY=3.0,
        )
        return np.clip(side_edge, 0.0, 1.0).astype(np.float32)

    def _build_torso_cloth_preserve_mask(
        self,
        face_bbox: Tuple[int, int, int, int],
        cloth_mask: Optional[np.ndarray],
        cutoff_y: int,
        hair_length: str,
    ) -> np.ndarray:
        if cloth_mask is None:
            return np.zeros((1, 1), dtype=np.float32)

        H, W = cloth_mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.18).astype(np.uint8) * 255
        if int((cloth_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        preserve_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(cutoff_y + face_h * (0.06 if hair_length == "short" else 0.10)))
        bottom = min(H, int(cutoff_y + face_h * (0.96 if hair_length == "short" else 1.35)))
        half_w = max(18, int(face_w * (0.58 if hair_length == "short" else 0.68)))
        if top < bottom:
            preserve_u8[top:bottom, max(0, cx - half_w):min(W, cx + half_w)] = 255

        ellipse_center = (
            cx,
            min(H - 1, int(y2 + face_h * (0.24 if hair_length == "short" else 0.32))),
        )
        ellipse_axes = (
            max(14, int(face_w * (0.44 if hair_length == "short" else 0.52))),
            max(12, int(face_h * (0.22 if hair_length == "short" else 0.32))),
        )
        cv2.ellipse(preserve_u8, ellipse_center, ellipse_axes, 0, 0, 360, 255, -1)

        preserve_u8 = cv2.bitwise_and(preserve_u8, cloth_u8)
        preserve_u8 = cv2.morphologyEx(
            preserve_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        preserve_u8 = cv2.dilate(
            preserve_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        if hair_length == "short":
            center_release_u8 = np.zeros((H, W), dtype=np.uint8)
            release_top = max(0, int(cutoff_y + face_h * 0.10))
            release_bottom = min(H, int(cutoff_y + face_h * 1.34))
            release_half = max(10, int(face_w * 0.13))
            if release_top < release_bottom:
                center_release_u8[
                    release_top:release_bottom,
                    max(0, cx - release_half):min(W, cx + release_half)
                ] = 255
                center_release_f = cv2.GaussianBlur(
                    center_release_u8.astype(np.float32) / 255.0,
                    (0, 0),
                    sigmaX=4.0,
                    sigmaY=6.0,
                )
                preserve_u8 = cv2.bitwise_and(
                    preserve_u8,
                    cv2.bitwise_not((center_release_f > 0.12).astype(np.uint8) * 255),
                )

        preserve = cv2.GaussianBlur(
            preserve_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=6.0,
            sigmaY=6.0,
        )
        return np.clip(preserve * (0.92 if hair_length == "short" else 0.68), 0.0, 1.0).astype(np.float32)

    def _build_bright_cloth_preserve_mask(
        self,
        img_rgb: np.ndarray,
        removal_mask: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
    ) -> np.ndarray:
        H, W = removal_mask.shape[:2]
        if cloth_mask is None or cloth_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)

        removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255
        zone_u8 = cv2.bitwise_and(removal_u8, cloth_u8)
        if int((zone_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(cutoff_y + face_h * 0.04))
        bottom = min(H, int(cutoff_y + face_h * (1.22 if hair_length == "short" else 1.46)))
        left = max(0, int(x1 - face_w * (1.14 if hair_length == "short" else 1.08)))
        right = min(W, int(x2 + face_w * (1.14 if hair_length == "short" else 1.08)))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)

        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        sat = hsv[:, :, 1]
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
        bright_u8 = (
            (gray > (176 if hair_length == "short" else 170))
            & (blur > (182 if hair_length == "short" else 176))
            & (sat < (58 if hair_length == "short" else 64))
        ).astype(np.uint8) * 255
        preserve_u8 = cv2.bitwise_and(bright_u8, zone_u8)
        if int((preserve_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        preserve_u8 = cv2.morphologyEx(
            preserve_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        preserve_u8 = cv2.morphologyEx(
            preserve_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
        )
        preserve_u8 = cv2.dilate(
            preserve_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
            iterations=1,
        )

        filtered_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(preserve_u8, 8)
        min_area = max(40, int(face_w * face_h * 0.006))
        max_area = max(2400, int(face_w * face_h * 0.22))
        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            comp_cx = float(centroids[idx][0])
            if area < min_area or area > max_area:
                continue
            if (y + h) < int(cutoff_y + face_h * 0.10):
                continue
            if abs(comp_cx - 0.5 * (x1 + x2)) > max(14, int(face_w * 0.14)):
                filtered_u8[labels == idx] = 255

        if int((filtered_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        preserve = cv2.GaussianBlur(
            filtered_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=5.0,
            sigmaY=5.0,
        )
        return np.clip(preserve, 0.0, 1.0).astype(np.float32)

    def _filter_short_torso_box_mask(
        self,
        img_rgb: Optional[np.ndarray],
        removal_mask: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        support_mask: Optional[np.ndarray],
        center_support_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
    ) -> np.ndarray:
        H, W = removal_mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        original_px = int((removal_u8 > 0).sum())
        if original_px < 40:
            return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8)
        else:
            cloth_u8 = np.zeros((H, W), dtype=np.uint8)

        support_hint_u8 = np.zeros((H, W), dtype=np.uint8)
        if support_mask is not None and support_mask.shape == (H, W):
            support_hint_u8 = cv2.bitwise_or(
                support_hint_u8,
                cv2.dilate(
                    (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 17)),
                    iterations=1,
                ),
            )
        if center_support_mask is not None and center_support_mask.shape == (H, W):
            support_hint_u8 = cv2.bitwise_or(
                support_hint_u8,
                cv2.dilate(
                    (np.clip(center_support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 19)),
                    iterations=1,
                ),
            )

        dark_evidence_u8 = np.zeros((H, W), dtype=np.uint8)
        bright_cloth_evidence_u8 = np.zeros((H, W), dtype=np.uint8)
        if img_rgb is not None and img_rgb.shape[:2] == (H, W):
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
            sat = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
            blackhat = cv2.morphologyEx(
                gray.astype(np.uint8),
                cv2.MORPH_BLACKHAT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
            )
            dark_evidence_u8 = (
                (
                    ((gray < 168.0) & ((blur - gray) > 2.4))
                    | (blackhat > 8)
                ).astype(np.uint8)
                * 255
            )
            dark_zone_u8 = np.zeros((H, W), dtype=np.uint8)
            dark_top = max(0, int(cutoff_y - face_h * 0.04))
            dark_bottom = min(H, int(cutoff_y + face_h * 0.96))
            dark_left = max(0, int(x1 - face_w * 1.12))
            dark_right = min(W, int(x2 + face_w * 1.12))
            if dark_top < dark_bottom and dark_left < dark_right:
                dark_zone_u8[dark_top:dark_bottom, dark_left:dark_right] = 255
                dark_evidence_u8 = cv2.bitwise_and(dark_evidence_u8, dark_zone_u8)
            if int((cloth_u8 > 0).sum()) > 0:
                dark_evidence_u8 = cv2.bitwise_and(dark_evidence_u8, cloth_u8.astype(np.uint8) * 255)
            bright_cloth_evidence_u8 = (
                (
                    (gray > 178.0)
                    & (blur > 182.0)
                    & (sat < 60.0)
                ).astype(np.uint8)
                * 255
            )
            bright_zone_u8 = np.zeros((H, W), dtype=np.uint8)
            bright_top = max(0, int(cutoff_y + face_h * 0.02))
            bright_bottom = min(H, int(cutoff_y + face_h * 1.04))
            bright_left = max(0, int(x1 - face_w * 1.16))
            bright_right = min(W, int(x2 + face_w * 1.16))
            if bright_top < bright_bottom and bright_left < bright_right:
                bright_zone_u8[bright_top:bright_bottom, bright_left:bright_right] = 255
                bright_cloth_evidence_u8 = cv2.bitwise_and(bright_cloth_evidence_u8, bright_zone_u8)
            if int((cloth_u8 > 0).sum()) > 0:
                bright_cloth_evidence_u8 = cv2.bitwise_and(
                    bright_cloth_evidence_u8,
                    cloth_u8.astype(np.uint8) * 255,
                )

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(removal_u8, 8)
        boxy_width = max(26, int(face_w * 0.30))
        boxy_height = max(24, int(face_h * 0.30))
        boxy_area = max(120, int(face_w * face_h * 0.075))
        center_half = max(16, int(face_w * 0.24))
        low_top = int(cutoff_y + face_h * 0.16)
        low_bottom = int(cutoff_y + face_h * 0.82)
        center_strand_width = max(16, int(face_w * 0.18))
        center_strand_height = max(30, int(face_h * 0.30))

        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 16:
                continue

            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            bottom = y + h
            comp_mask = labels == idx
            cloth_overlap = int(cloth_u8[comp_mask].sum())
            overlap_ratio = float(cloth_overlap) / float(area) if area > 0 else 0.0
            comp_cx = float(centroids[idx][0])
            comp_u8 = comp_mask.astype(np.uint8) * 255
            support_overlap = int((cv2.bitwise_and(comp_u8, support_hint_u8) > 0).sum())
            dark_overlap = int((cv2.bitwise_and(comp_u8, dark_evidence_u8) > 0).sum())
            dark_ratio = float(dark_overlap) / float(area) if area > 0 else 0.0
            bright_overlap = int((cv2.bitwise_and(comp_u8, bright_cloth_evidence_u8) > 0).sum())
            bright_ratio = float(bright_overlap) / float(area) if area > 0 else 0.0

            is_boxy = w >= boxy_width and h >= boxy_height and area >= boxy_area
            is_center_box = abs(comp_cx - cx) <= center_half and w >= max(22, int(face_w * 0.28))
            is_center_strand = (
                abs(comp_cx - cx) <= center_half
                and w <= center_strand_width
                and h >= center_strand_height
                and bottom >= int(cutoff_y + face_h * 0.30)
            )
            is_low = y >= low_top or bottom >= low_bottom
            is_side_component = abs(comp_cx - cx) >= max(22, int(face_w * 0.28))
            is_side_blob = (
                is_side_component
                and overlap_ratio >= 0.30
                and w >= max(20, int(face_w * 0.22))
                and area >= max(72, int(face_w * face_h * 0.022))
            )
            is_bright_side_blob = (
                is_side_component
                and overlap_ratio >= 0.26
                and bright_ratio >= 0.18
                and w >= max(24, int(face_w * 0.28))
                and area >= max(96, int(face_w * face_h * 0.024))
                and bottom >= int(cutoff_y + face_h * 0.14)
            )
            if is_center_strand:
                keep_u8[comp_mask] = 255
                continue
            if is_bright_side_blob:
                if support_overlap < max(16, int(area * 0.08)):
                    if dark_ratio < 0.16 or bright_ratio > (dark_ratio * 1.8 + 0.06):
                        continue
            if is_side_blob and (dark_ratio < 0.09 or dark_overlap < max(8, int(area * 0.04))):
                if support_overlap < max(14, int(area * 0.08)):
                    continue
                if h < max(28, int(face_h * 0.42)):
                    continue
            if overlap_ratio > 0.46 and (is_boxy or is_center_box or is_low):
                if support_overlap >= max(18, int(area * 0.08)):
                    supported_u8 = cv2.bitwise_and(comp_u8, support_hint_u8)
                    supported_u8 = cv2.dilate(
                        supported_u8,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 19)),
                        iterations=1,
                    )
                    supported_u8 = cv2.bitwise_and(supported_u8, comp_u8)
                    keep_u8 = cv2.bitwise_or(keep_u8, supported_u8)
                continue
            if is_low and is_boxy and support_overlap < max(10, int(area * 0.05)):
                continue
            if is_side_component and overlap_ratio > 0.34 and dark_ratio < 0.07 and area >= max(80, int(face_w * face_h * 0.028)):
                continue
            if (
                is_side_component
                and overlap_ratio > 0.30
                and bright_ratio >= 0.22
                and dark_ratio < 0.11
                and support_overlap < max(12, int(area * 0.06))
                and area >= max(110, int(face_w * face_h * 0.030))
            ):
                continue

            keep_u8[comp_mask] = 255

        if int((support_hint_u8 > 0).sum()) > 0:
            deep_torso_u8 = np.zeros((H, W), dtype=np.uint8)
            deep_x1 = max(0, int(x1 - face_w * 1.05))
            deep_x2 = min(W, int(x2 + face_w * 1.05))
            deep_y1 = max(0, int(cutoff_y + face_h * 0.22))
            if deep_x1 < deep_x2 and deep_y1 < H:
                deep_torso_u8[deep_y1:, deep_x1:deep_x2] = 255
                if int((cloth_u8 > 0).sum()) > 0:
                    deep_torso_u8 = cv2.bitwise_and(deep_torso_u8, cloth_u8.astype(np.uint8) * 255)
                precise_keep_u8 = cv2.dilate(
                    support_hint_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 13)),
                    iterations=1,
                )
                precise_keep_u8 = cv2.bitwise_and(precise_keep_u8, deep_torso_u8)
                keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(deep_torso_u8))
                keep_u8 = cv2.bitwise_or(
                    keep_u8,
                    cv2.bitwise_and(removal_u8, precise_keep_u8),
                )

        kept_px = int((keep_u8 > 0).sum())
        if kept_px <= 0:
            return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

        filtered = removal_mask.astype(np.float32) * (keep_u8.astype(np.float32) / 255.0)
        filtered = cv2.GaussianBlur(filtered, (0, 0), sigmaX=1.1, sigmaY=1.1)
        return np.clip(filtered, 0.0, 1.0).astype(np.float32)

    def _build_micro_cloth_artifact_mask(
        self,
        img_rgb: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
        final_hair_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if hair_length not in ("short", "medium"):
            return np.zeros(img_rgb.shape[:2], dtype=np.float32)

        H, W = img_rgb.shape[:2]
        if cloth_mask is None or cloth_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        zone_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(cutoff_y - face_h * 0.04))
        bottom = min(H, int(cutoff_y + face_h * (0.82 if hair_length == "short" else 0.92)))
        left = max(0, int(x1 - face_w * (1.04 if hair_length == "short" else 1.12)))
        right = min(W, int(x2 + face_w * (1.04 if hair_length == "short" else 1.12)))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        zone_u8[top:bottom, left:right] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, cloth_u8)
        if int((zone_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
        blackhat = cv2.morphologyEx(
            gray.astype(np.uint8),
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 15) if hair_length == "short" else (7, 13)),
        )
        candidate_u8 = (
            (
                ((gray < (172.0 if hair_length == "short" else 166.0)) & ((blur - gray) > (2.5 if hair_length == "short" else 2.8)))
                | (blackhat > (8 if hair_length == "short" else 9))
            ).astype(np.uint8)
            * 255
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, zone_u8)
        if final_hair_mask is not None and final_hair_mask.shape == (H, W):
            final_hair_u8 = cv2.dilate(
                (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 15)),
                iterations=1,
            )
            candidate_u8 = cv2.bitwise_and(candidate_u8, cv2.bitwise_not(final_hair_u8))
        candidate_u8 = cv2.morphologyEx(
            candidate_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        if int((candidate_u8 > 0).sum()) < 8:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
        max_area = max(160, int(face_w * face_h * 0.018))
        max_width = max(24, int(face_w * 0.20))
        max_height = max(72, int(face_h * 0.62))
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom_y = y + h
            comp_cx = float(centroids[idx][0])
            if area < 6 or area > max_area:
                continue
            if w > max_width or h > max_height:
                continue
            if bottom_y > int(cutoff_y + face_h * 0.88):
                continue
            fill_ratio = float(area) / float(max(w * h, 1))
            if fill_ratio > 0.72 and area > max(42, int(face_w * face_h * 0.006)):
                continue
            if abs(comp_cx - cx) > max(44, int(face_w * 0.56)) and area > max(28, int(face_w * face_h * 0.004)):
                continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 8:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, zone_u8)
        return (keep_u8 > 0).astype(np.float32)

    def _restrict_short_removal_to_tail_lanes(
        self,
        removal_mask: np.ndarray,
        support_mask: Optional[np.ndarray],
        center_support_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
    ) -> np.ndarray:
        if hair_length != "short":
            return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

        H, W = removal_mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)

        removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        original_px = int((removal_u8 > 0).sum())
        if original_px < 60:
            return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

        side_seed_u8 = np.zeros((H, W), dtype=np.uint8)
        if support_mask is not None and support_mask.shape == (H, W):
            side_seed_u8 = (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255

        center_seed_u8 = np.zeros((H, W), dtype=np.uint8)
        if center_support_mask is not None and center_support_mask.shape == (H, W):
            center_seed_u8 = (np.clip(center_support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255

        if int((side_seed_u8 > 0).sum()) < 24 and int((center_seed_u8 > 0).sum()) < 12:
            return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

        side_lane_u8 = cv2.dilate(
            side_seed_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 41)),
            iterations=1,
        )
        center_lane_u8 = cv2.dilate(
            center_seed_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 41)),
            iterations=1,
        )
        lane_u8 = cv2.bitwise_or(side_lane_u8, center_lane_u8)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(cutoff_y))
        bottom = min(H, int(cutoff_y + face_h * 1.35))
        left = max(0, int(x1 - face_w * 0.95))
        right = min(W, int(x2 + face_w * 0.95))
        if top >= bottom or left >= right:
            return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)
        corridor_u8[top:bottom, left:right] = 255
        lane_u8 = cv2.bitwise_and(lane_u8, corridor_u8)
        if int((lane_u8 > 0).sum()) < 120:
            return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

        filtered_u8 = cv2.bitwise_and(removal_u8, lane_u8)
        filtered_u8 = cv2.morphologyEx(
            filtered_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
        )
        filtered_px = int((filtered_u8 > 0).sum())
        if filtered_px < max(180, int(original_px * 0.16)):
            return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

        filtered = removal_mask.astype(np.float32) * (filtered_u8.astype(np.float32) / 255.0)
        filtered = cv2.GaussianBlur(filtered, (0, 0), sigmaX=1.0, sigmaY=1.2)
        return np.clip(filtered, 0.0, 1.0).astype(np.float32)

    def _build_shoulder_cloth_restore_mask(
        self,
        removal_mask: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
        protect_mask: Optional[np.ndarray] = None,
        final_hair_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if hair_length not in ("short", "medium"):
            return np.zeros(removal_mask.shape[:2], dtype=np.float32)

        H, W = removal_mask.shape[:2]
        if cloth_mask is None or cloth_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        base_u8 = (
            (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255
        )
        cloth_u8 = (
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255
        )
        base_u8 = cv2.bitwise_and(base_u8, cloth_u8)
        if int((base_u8 > 0).sum()) < 60:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(cutoff_y - face_h * 0.02))
        bottom = min(H, int(cutoff_y + face_h * (1.10 if hair_length == "short" else 1.22)))
        left = max(0, int(x1 - face_w * 1.18))
        right = min(W, int(x2 + face_w * 1.18))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255
        base_u8 = cv2.bitwise_and(base_u8, corridor_u8)

        center_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
        keepout_half = max(14, int(face_w * 0.20))
        keepout_top = max(0, int(cutoff_y - face_h * 0.02))
        keepout_bottom = min(H, int(cutoff_y + face_h * 1.18))
        if keepout_top < keepout_bottom:
            center_keepout_u8[
                keepout_top:keepout_bottom,
                max(0, cx - keepout_half):min(W, cx + keepout_half),
            ] = 255
            base_u8 = cv2.bitwise_and(base_u8, cv2.bitwise_not(center_keepout_u8))

        base_u8 = cv2.morphologyEx(
            base_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        base_u8 = cv2.morphologyEx(
            base_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
        )
        if int((base_u8 > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(base_u8, 8)
        min_area = max(60, int(face_w * face_h * 0.008))
        max_area = max(1800, int(face_w * face_h * 0.42))
        max_width = max(88, int(face_w * 1.12))
        min_height = max(36, int(face_h * 0.18))
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            comp_cx = float(centroids[idx][0])
            if area < min_area or area > max_area:
                continue
            if w > max_width or h < min_height:
                continue
            if abs(comp_cx - cx) < max(12, int(face_w * 0.14)):
                continue
            if (y + h) < int(cutoff_y + face_h * 0.08):
                continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.morphologyEx(
            keep_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 23)),
        )
        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 23)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
        keep_u8 = cv2.bitwise_and(keep_u8, cloth_u8)
        if protect_mask is not None and protect_mask.shape == (H, W):
            protect_u8 = cv2.dilate(
                (np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                iterations=1,
            )
            keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(protect_u8))
        if final_hair_mask is not None and final_hair_mask.shape == (H, W):
            final_hair_u8 = cv2.dilate(
                (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
                iterations=1,
            )
            keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(final_hair_u8))
        if int((keep_u8 > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)
        return (keep_u8 > 0).astype(np.float32)

    def _restore_cloth_overlap_from_source(
        self,
        source_rgb: np.ndarray,
        current_rgb: np.ndarray,
        restore_mask: np.ndarray,
        final_hair_mask: Optional[np.ndarray] = None,
        tone_reference_rgb: Optional[np.ndarray] = None,
        tone_reference_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        H, W = current_rgb.shape[:2]
        if source_rgb.shape[:2] != (H, W) or restore_mask.shape != (H, W):
            return current_rgb

        mask_u8 = (np.clip(restore_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        if int((mask_u8 > 0).sum()) < 40:
            return current_rgb
        if final_hair_mask is not None and final_hair_mask.shape == (H, W):
            final_hair_u8 = cv2.dilate(
                (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 15)),
                iterations=1,
            )
            mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(final_hair_u8))
        if int((mask_u8 > 0).sum()) < 40:
            return current_rgb

        mask_u8 = cv2.dilate(
            mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        ns = cv2.inpaint(source_rgb, mask_u8, 5, cv2.INPAINT_NS)
        telea = cv2.inpaint(source_rgb, mask_u8, 4, cv2.INPAINT_TELEA)
        refill = cv2.addWeighted(telea, 0.64, ns, 0.36, 0.0)

        ring_u8 = cv2.dilate(
            mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            iterations=1,
        )
        ring_u8 = cv2.subtract(
            ring_u8,
            cv2.dilate(
                mask_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ),
        )
        ring_bool = ring_u8 > 0
        mask_bool = mask_u8 > 0
        if int(ring_bool.sum()) >= 80 and int(mask_bool.sum()) >= 40:
            refill_lab = cv2.cvtColor(refill, cv2.COLOR_RGB2LAB).astype(np.float32)
            reference_rgb = source_rgb
            if tone_reference_rgb is not None and tone_reference_rgb.shape[:2] == (H, W):
                reference_rgb = tone_reference_rgb
            reference_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            ring_ref_bool = ring_bool.copy()
            if tone_reference_mask is not None and tone_reference_mask.shape == (H, W):
                ref_mask_u8 = cv2.dilate(
                    (np.clip(tone_reference_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                    iterations=1,
                )
                masked_ring = np.logical_and(ring_ref_bool, ref_mask_u8 > 0)
                if int(masked_ring.sum()) >= 40:
                    ring_ref_bool = masked_ring
            ring_vals = reference_lab[ring_ref_bool]
            mask_vals = refill_lab[mask_bool]
            ring_mean = ring_vals.mean(axis=0)
            mask_mean = mask_vals.mean(axis=0)
            ring_std = ring_vals.std(axis=0)
            mask_std = np.maximum(mask_vals.std(axis=0), 1.0)
            tone_matched = mask_vals.copy()
            tone_matched[:, 0] = np.clip(
                (tone_matched[:, 0] - mask_mean[0]) * np.clip(ring_std[0] / mask_std[0], 0.84, 1.16)
                + mask_mean[0]
                + np.clip(ring_mean[0] - mask_mean[0], -14.0, 14.0) * 0.72,
                0.0,
                255.0,
            )
            tone_matched[:, 1] = np.clip(
                tone_matched[:, 1] + np.clip(ring_mean[1] - mask_mean[1], -5.0, 5.0) * 0.45,
                0.0,
                255.0,
            )
            tone_matched[:, 2] = np.clip(
                tone_matched[:, 2] + np.clip(ring_mean[2] - mask_mean[2], -5.0, 5.0) * 0.45,
                0.0,
                255.0,
            )
            refill_lab[mask_bool] = tone_matched
            refill = cv2.cvtColor(refill_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        alpha = cv2.GaussianBlur(
            (mask_u8 > 0).astype(np.float32),
            (0, 0),
            sigmaX=2.6,
            sigmaY=2.6,
        )[..., np.newaxis]
        alpha = np.clip(alpha * 0.96, 0.0, 1.0)
        out = refill.astype(np.float32) * alpha + current_rgb.astype(np.float32) * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _cleanup_region_with_cloth_restore(
        self,
        *,
        source_rgb: np.ndarray,
        current_rgb: np.ndarray,
        cleanup_mask: np.ndarray,
        cloth_mask: Optional[np.ndarray] = None,
        final_hair_mask: Optional[np.ndarray] = None,
        ignore_final_hair_for_cloth_restore: bool = False,
        cleanup_dark_tail: bool = True,
    ) -> np.ndarray:
        H, W = current_rgb.shape[:2]
        if source_rgb.shape[:2] != (H, W) or cleanup_mask.shape != (H, W):
            return current_rgb

        mask_u8 = (np.clip(cleanup_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        if int((mask_u8 > 0).sum()) < 40:
            return current_rgb

        cleaned = self._lama_inpaint(current_rgb, mask_u8)
        if cleanup_dark_tail:
            cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, mask_u8)

        if cloth_mask is None or cloth_mask.shape != (H, W):
            return cleaned

        cloth_cleanup_mask = np.clip(
            cleanup_mask.astype(np.float32)
            * (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.float32),
            0.0,
            1.0,
        )
        cloth_cleanup_u8 = (cloth_cleanup_mask > 0.08).astype(np.uint8) * 255
        if int((cloth_cleanup_u8 > 0).sum()) < 30:
            return cleaned

        restore_final_hair_mask = None if ignore_final_hair_for_cloth_restore else final_hair_mask
        reference_fill_u8 = cv2.dilate(
            cloth_cleanup_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 25)),
            iterations=1,
        )
        if cloth_mask is not None and cloth_mask.shape == (H, W):
            wide_cloth_u8 = cv2.dilate(
                (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
                iterations=1,
            )
            reference_fill_u8 = cv2.bitwise_and(reference_fill_u8, wide_cloth_u8)
        reference_fill_mask = reference_fill_u8.astype(np.float32) / 255.0
        reference_fill_rgb = self._restore_cloth_overlap_from_source(
            source_rgb=source_rgb,
            current_rgb=source_rgb,
            restore_mask=reference_fill_mask,
            final_hair_mask=None,
            tone_reference_rgb=source_rgb,
            tone_reference_mask=cloth_mask,
        )
        source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        source_sat = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
        visible_cloth_u8 = cv2.bitwise_and(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.bitwise_not(
                cv2.dilate(
                    cloth_cleanup_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
                    iterations=1,
                )
            ),
        )
        plain_gray = 0.0
        plain_sat = 255.0
        use_plain_cloth_force = False
        plain_fill_rgb: Optional[np.ndarray] = None
        if int((visible_cloth_u8 > 0).sum()) >= 80:
            plain_gray = float(np.median(source_gray[visible_cloth_u8 > 0]))
            plain_sat = float(np.median(source_sat[visible_cloth_u8 > 0]))
            use_plain_cloth_force = plain_gray >= 168.0 and plain_sat <= 84.0
        if use_plain_cloth_force:
            plain_fill_rgb = source_rgb.copy()
            plain_fill_color = np.median(source_rgb[visible_cloth_u8 > 0], axis=0).astype(np.uint8)
            plain_fill_rgb[reference_fill_u8 > 0] = plain_fill_color
            plain_fill_rgb = self._restore_reference_region(
                reference_fill_rgb,
                plain_fill_rgb,
                reference_fill_mask,
                strength=0.98,
            )
            reference_fill_rgb = self._blend_neighbor_cloth_tone(
                plain_fill_rgb,
                reference_fill_mask,
                cloth_mask=cloth_mask,
                reference_rgb=source_rgb,
            )
            reference_fill_rgb = self._cv2_refine_cloth_region(
                reference_fill_rgb,
                reference_fill_mask,
                reference_rgb=source_rgb,
                reference_mask=cloth_mask,
            )
        cleaned = self._restore_cloth_overlap_from_source(
            source_rgb=source_rgb,
            current_rgb=cleaned,
            restore_mask=cloth_cleanup_mask,
            final_hair_mask=restore_final_hair_mask,
            tone_reference_rgb=source_rgb,
            tone_reference_mask=cloth_mask,
        )
        if use_plain_cloth_force:
            cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY).astype(np.float32)
            cleaned_sat = cv2.cvtColor(cleaned, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
            cleanup_bool = cloth_cleanup_u8 > 0
            cleanup_dark_ratio = 0.0
            if int(cleanup_bool.sum()) >= 40:
                cleanup_dark_ratio = float(
                    np.mean(
                        (
                            (cleaned_gray < plain_gray - 14.0)
                            & (cleaned_sat < 138.0)
                        )[cleanup_bool]
                    )
                )
            if cleanup_dark_ratio >= 0.14:
                cleaned = self._restore_reference_region(
                    cleaned,
                    plain_fill_rgb if plain_fill_rgb is not None else reference_fill_rgb,
                    cloth_cleanup_mask,
                    strength=float(np.clip(0.88 + cleanup_dark_ratio * 0.24, 0.88, 0.98)),
                )
        cleaned = self._overlay_reference_cloth_fill(
            cleaned,
            reference_fill_rgb,
            cloth_cleanup_mask,
            cloth_mask=cloth_mask,
        )
        cleaned = self._blend_neighbor_cloth_tone(
            cleaned,
            cloth_cleanup_mask,
            cloth_mask=cloth_mask,
            reference_rgb=reference_fill_rgb,
        )
        cleaned = self._cv2_refine_cloth_region(
            cleaned,
            cloth_cleanup_mask,
            reference_rgb=reference_fill_rgb,
            reference_mask=cloth_mask,
        )
        if use_plain_cloth_force:
            cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY).astype(np.float32)
            cleaned_sat = cv2.cvtColor(cleaned, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
            ref_gray = cv2.cvtColor(reference_fill_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            dark_residual_u8 = (
                (
                    (
                        (cleaned_gray + 12.0 < ref_gray)
                        & (cleaned_sat < 132.0)
                    )
                    | (
                        (cleaned_gray < plain_gray - 18.0)
                        & (cleaned_sat < 124.0)
                    )
                ).astype(np.uint8)
                * 255
            )
            dark_residual_u8 = cv2.bitwise_and(dark_residual_u8, cloth_cleanup_u8)
            dark_residual_u8 = cv2.morphologyEx(
                dark_residual_u8,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
            )
            dark_residual_u8 = cv2.dilate(
                dark_residual_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
                iterations=1,
            )
            if int((dark_residual_u8 > 0).sum()) >= 24:
                cleaned = self._restore_reference_region(
                    cleaned,
                    plain_fill_rgb if plain_fill_rgb is not None else reference_fill_rgb,
                    dark_residual_u8.astype(np.float32) / 255.0,
                    strength=0.98,
                )
        cleaned = self._overlay_reference_cloth_fill(
            cleaned,
            reference_fill_rgb,
            cloth_cleanup_mask,
            cloth_mask=cloth_mask,
        )
        return cleaned

    def _build_side_column_cloth_restore_mask(
        self,
        img_rgb: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        candidate_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
        final_hair_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        H, W = img_rgb.shape[:2]
        if cloth_mask is None or candidate_mask is None:
            return np.zeros((H, W), dtype=np.float32)
        if cloth_mask.shape != (H, W) or candidate_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        candidate_u8 = (np.clip(candidate_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        zone_u8 = cv2.bitwise_and(candidate_u8, cloth_u8)
        if int((zone_u8 > 0).sum()) < 80:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(cutoff_y + face_h * (0.10 if hair_length == "short" else 0.20)))
        bottom = min(H, int(cutoff_y + face_h * (1.74 if hair_length == "short" else 1.34)))
        left = max(0, int(x1 - face_w * (1.42 if hair_length == "short" else 1.12)))
        right = min(W, int(x2 + face_w * (1.42 if hair_length == "short" else 1.12)))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)

        if int((zone_u8 > 0).sum()) < 60:
            return np.zeros((H, W), dtype=np.float32)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        bright_smooth_u8 = (
            (gray > (194.0 if hair_length == "short" else 188.0))
            & (blur > (198.0 if hair_length == "short" else 192.0))
            & (sat < (92.0 if hair_length == "short" else 96.0))
            & (lap < (24.0 if hair_length == "short" else 26.0))
        ).astype(np.uint8) * 255
        dark_smooth_u8 = (
            (gray > (58.0 if hair_length == "short" else 72.0))
            & (gray < (188.0 if hair_length == "short" else 172.0))
            & (sat < (100.0 if hair_length == "short" else 100.0))
            & (lap < (24.0 if hair_length == "short" else 20.0))
        ).astype(np.uint8) * 255
        smooth_u8 = cv2.bitwise_or(bright_smooth_u8, dark_smooth_u8)
        zone_u8 = cv2.bitwise_and(zone_u8, smooth_u8)

        if final_hair_mask is not None and final_hair_mask.shape == (H, W):
            final_hair_u8 = cv2.dilate(
                (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24).astype(np.uint8) * 255,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (7, 11) if hair_length == "short" else (11, 17),
                ),
                iterations=1,
            )
            if hair_length == "short":
                final_hair_u8[min(H, int(cutoff_y + face_h * 0.80)):, :] = 0
            zone_u8 = cv2.bitwise_and(zone_u8, cv2.bitwise_not(final_hair_u8))
        if int((zone_u8 > 0).sum()) < 60:
            return np.zeros((H, W), dtype=np.float32)

        zone_u8 = cv2.morphologyEx(
            zone_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        zone_u8 = cv2.morphologyEx(
            zone_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 19)),
        )

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
        min_area = max(120, int(face_w * face_h * 0.014))
        max_area = max(7000, int(face_w * face_h * 0.34))
        min_height = max(36, int(face_h * 0.12))
        max_width = max(120, int(face_w * (0.78 if hair_length == "short" else 0.54)))
        max_offset = max(120, int(face_w * (1.38 if hair_length == "short" else 0.60)))
        short_center_offset = max(18, int(face_w * 0.18))
        short_center_width = max(92, int(face_w * 0.42))
        short_center_area = max(2400, int(face_w * face_h * 0.15))
        short_side_width = max(132, int(face_w * 0.86))
        short_side_area = max(12000, int(face_w * face_h * 0.52))
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            comp_cx = float(centroids[idx][0])
            offset = abs(comp_cx - cx)
            bottom_y = y + h
            if area < min_area or area > max_area:
                continue
            if h < min_height or w > max_width:
                continue
            if y < int(cutoff_y + face_h * (0.08 if hair_length == "short" else 0.22)):
                continue
            if bottom_y < int(cutoff_y + face_h * (0.34 if hair_length == "short" else 0.44)):
                continue
            if offset > max_offset:
                continue
            if hair_length == "short":
                if offset <= short_center_offset:
                    if w > short_center_width or area > short_center_area:
                        continue
                else:
                    if w > short_side_width or area > short_side_area:
                        continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 60:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (19, 31) if hair_length == "short" else (9, 17),
            ),
            iterations=1,
        )
        loose_cloth_u8 = cv2.dilate(
            cloth_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )
        keep_gate_u8 = cv2.bitwise_or(loose_cloth_u8, candidate_u8)
        keep_u8 = cv2.bitwise_and(keep_u8, keep_gate_u8)
        keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
        if int((keep_u8 > 0).sum()) < 60:
            return np.zeros((H, W), dtype=np.float32)

        return cv2.GaussianBlur(
            keep_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=4.0,
            sigmaY=6.0,
        ).astype(np.float32)

    def _build_direct_short_column_restore_mask(
        self,
        removal_mask: Optional[np.ndarray],
        cloth_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
    ) -> np.ndarray:
        if removal_mask is not None:
            base_shape = removal_mask.shape[:2]
        elif cloth_mask is not None:
            base_shape = cloth_mask.shape[:2]
        else:
            base_shape = (1, 1)
        if hair_length != "short":
            return np.zeros(base_shape, dtype=np.float32)

        if removal_mask is None or cloth_mask is None:
            return np.zeros(base_shape, dtype=np.float32)

        H, W = removal_mask.shape[:2]
        if cloth_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        zone_u8 = cv2.bitwise_and(removal_u8, cloth_u8)
        if int((zone_u8 > 0).sum()) < 80:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(cutoff_y - face_h * 0.10))
        bottom = min(H, int(cutoff_y + face_h * 1.82))
        left = max(0, int(x1 - face_w * 1.48))
        right = min(W, int(x2 + face_w * 1.48))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
        if int((zone_u8 > 0).sum()) < 80:
            return np.zeros((H, W), dtype=np.float32)

        zone_u8 = cv2.morphologyEx(
            zone_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        zone_u8 = cv2.morphologyEx(
            zone_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
        )

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
        min_area = max(120, int(face_w * face_h * 0.010))
        max_area = max(22000, int(face_w * face_h * 0.72))
        min_height = max(72, int(face_h * 0.28))
        max_width = max(176, int(face_w * 1.00))
        max_offset = max(360, int(face_w * 1.62))
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
            if bottom_y < int(cutoff_y + face_h * 0.24):
                continue
            if bottom_y > int(cutoff_y + face_h * 1.82):
                continue
            if abs(comp_cx - cx) > max_offset:
                continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 80:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 33)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(
            keep_u8,
            cv2.dilate(
                cloth_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
                iterations=1,
            ),
        )
        keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
        if int((keep_u8 > 0).sum()) < 80:
            return np.zeros((H, W), dtype=np.float32)

        return cv2.GaussianBlur(
            keep_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=4.2,
            sigmaY=7.4,
        ).astype(np.float32)

    def _build_short_below_bob_cloth_restore_mask(
        self,
        removal_mask: Optional[np.ndarray],
        cloth_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
        support_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if removal_mask is not None:
            base_shape = removal_mask.shape[:2]
        elif cloth_mask is not None:
            base_shape = cloth_mask.shape[:2]
        else:
            base_shape = (1, 1)
        if hair_length != "short":
            return np.zeros(base_shape, dtype=np.float32)
        if removal_mask is None or cloth_mask is None:
            return np.zeros(base_shape, dtype=np.float32)

        H, W = removal_mask.shape[:2]
        if cloth_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        zone_u8 = cv2.bitwise_and(removal_u8, cloth_u8)

        if support_mask is not None and support_mask.shape == (H, W):
            support_u8 = cv2.dilate(
                (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
                iterations=1,
            )
            zone_u8 = cv2.bitwise_or(zone_u8, cv2.bitwise_and(support_u8, cloth_u8))

        if int((zone_u8 > 0).sum()) < 60:
            return np.zeros((H, W), dtype=np.float32)

        bob_floor = max(
            0,
            min(
                int(y2 + face_h * 0.10),
                int(cutoff_y + face_h * 0.08),
            ),
        )
        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        left = max(0, int(x1 - face_w * 1.26))
        right = min(W, int(x2 + face_w * 1.26))
        bottom = min(H, int(cutoff_y + face_h * 1.78))
        if bob_floor >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[bob_floor:bottom, left:right] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
        if int((zone_u8 > 0).sum()) < 60:
            return np.zeros((H, W), dtype=np.float32)

        lane_u8 = np.zeros((H, W), dtype=np.uint8)
        side_inner_gap = max(14, int(face_w * 0.08))
        left_lane_right = max(left + 1, int(cx - side_inner_gap))
        right_lane_left = min(right - 1, int(cx + side_inner_gap))
        lane_u8[bob_floor:bottom, left:left_lane_right] = 255
        lane_u8[bob_floor:bottom, right_lane_left:right] = 255
        center_lane_top = min(bottom, int(cutoff_y + face_h * 0.34))
        center_half = max(18, int(face_w * 0.20))
        center_x1 = max(left, int(cx - center_half))
        center_x2 = min(right, int(cx + center_half))
        if center_lane_top < bottom and center_x1 < center_x2:
            lane_u8[center_lane_top:bottom, center_x1:center_x2] = 255

        zone_u8 = cv2.bitwise_and(zone_u8, lane_u8)
        if int((zone_u8 > 0).sum()) < 60:
            return np.zeros((H, W), dtype=np.float32)

        zone_u8 = cv2.morphologyEx(
            zone_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        zone_u8 = cv2.morphologyEx(
            zone_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
        )

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
        min_area = max(36, int(face_w * face_h * 0.002))
        max_area = max(12000, int(face_w * face_h * 0.28))
        min_height = max(18, int(face_h * 0.10))
        max_width = max(152, int(face_w * 0.98))
        center_keepout = max(16, int(face_w * 0.16))
        deep_center_bottom = int(y2 + face_h * 0.52)

        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom_y = y + h
            comp_cx = float(centroids[idx][0])
            offset = abs(comp_cx - cx)
            if area < min_area or area > max_area:
                continue
            if h < min_height or w > max_width:
                continue
            if bottom_y < int(bob_floor + face_h * 0.10):
                continue
            if offset < center_keepout and bottom_y < deep_center_bottom:
                continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 23)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, lane_u8)
        if int((keep_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        return cv2.GaussianBlur(
            keep_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=3.0,
            sigmaY=5.4,
        ).astype(np.float32)

    def _build_short_below_bob_generation_block_mask(
        self,
        removal_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
        support_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if removal_mask is None:
            return np.zeros((1, 1), dtype=np.float32)
        H, W = removal_mask.shape[:2]
        if hair_length != "short":
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        removal_u8 = cv2.dilate(
            (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
            iterations=1,
        )
        zone_u8 = removal_u8.copy()
        if support_mask is not None and support_mask.shape == (H, W):
            support_u8 = cv2.dilate(
                (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21)),
                iterations=1,
            )
            zone_u8 = cv2.bitwise_or(zone_u8, support_u8)

        bob_floor = max(
            0,
            min(
                int(y2 + face_h * 0.02),
                int(cutoff_y + face_h * 0.04),
            ),
        )
        bottom = min(H, int(cutoff_y + face_h * 1.46))
        left = max(0, int(x1 - face_w * 1.22))
        right = min(W, int(x2 + face_w * 1.22))
        if bob_floor >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        corridor_u8[bob_floor:bottom, left:right] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
        if int((zone_u8 > 0).sum()) < 60:
            return np.zeros((H, W), dtype=np.float32)

        lane_u8 = np.zeros((H, W), dtype=np.uint8)
        side_inner_gap = max(12, int(face_w * 0.06))
        left_lane_right = max(left + 1, int(cx - side_inner_gap))
        right_lane_left = min(right - 1, int(cx + side_inner_gap))
        lane_u8[bob_floor:bottom, left:left_lane_right] = 255
        lane_u8[bob_floor:bottom, right_lane_left:right] = 255
        center_lane_top = min(bottom, int(cutoff_y + face_h * 0.28))
        center_half = max(20, int(face_w * 0.22))
        center_x1 = max(left, int(cx - center_half))
        center_x2 = min(right, int(cx + center_half))
        if center_lane_top < bottom and center_x1 < center_x2:
            lane_u8[center_lane_top:bottom, center_x1:center_x2] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, lane_u8)
        if int((zone_u8 > 0).sum()) < 60:
            return np.zeros((H, W), dtype=np.float32)

        zone_u8 = cv2.morphologyEx(
            zone_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 27)),
        )
        zone_u8 = cv2.dilate(
            zone_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
            iterations=1,
        )
        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
        min_area = max(36, int(face_w * face_h * 0.0018))
        max_area = max(32000, int(face_w * face_h * 0.56))
        min_height = max(20, int(face_h * 0.10))
        max_width = max(228, int(face_w * 1.36))
        center_keepout = max(20, int(face_w * 0.18))
        deep_center_bottom = int(y2 + face_h * 0.34)

        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom_y = y + h
            comp_cx = float(centroids[idx][0])
            offset = abs(comp_cx - cx)
            if area < min_area or area > max_area:
                continue
            if h < min_height or w > max_width:
                continue
            if bottom_y < int(bob_floor + face_h * 0.10):
                continue
            if offset < center_keepout and bottom_y < deep_center_bottom:
                continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 33)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, lane_u8)
        return cv2.GaussianBlur(
            keep_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=2.4,
            sigmaY=4.2,
        ).astype(np.float32)

    def _build_preclean_side_column_cleanup_mask(
        self,
        img_rgb: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        base_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
    ) -> np.ndarray:
        H, W = img_rgb.shape[:2]
        if cloth_mask is None or base_mask is None:
            return np.zeros((H, W), dtype=np.float32)
        if cloth_mask.shape != (H, W) or base_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        base_u8 = (np.clip(base_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        zone_u8 = cv2.bitwise_and(base_u8, cloth_u8)
        if int((zone_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(cutoff_y + face_h * 0.04))
        bottom = min(H, int(cutoff_y + face_h * (1.18 if hair_length == "short" else 1.10)))
        left = max(0, int(x1 - face_w * 1.34))
        right = min(W, int(x2 + face_w * 1.34))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
        if int((zone_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        blackhat = cv2.morphologyEx(
            gray.astype(np.uint8),
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 21) if hair_length == "short" else (7, 17),
            ),
        )
        candidate_u8 = (
            (
                (
                    (gray < (176.0 if hair_length == "short" else 170.0))
                    & (blur < (182.0 if hair_length == "short" else 176.0))
                    & (sat < (118.0 if hair_length == "short" else 112.0))
                    & (lap < (28.0 if hair_length == "short" else 26.0))
                )
                | (blackhat > (9 if hair_length == "short" else 10))
            ).astype(np.uint8)
            * 255
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, zone_u8)
        candidate_u8 = cv2.morphologyEx(
            candidate_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        candidate_u8 = cv2.morphologyEx(
            candidate_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (5, 13) if hair_length == "short" else (5, 11),
            ),
        )
        if int((candidate_u8 > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
        max_area = max(2600, int(face_w * face_h * 0.12))
        max_width = max(44, int(face_w * 0.28))
        min_height = max(34, int(face_h * 0.18))
        max_offset = max(260, int(face_w * 1.14))
        center_offset = max(16, int(face_w * 0.14))
        center_max_width = max(58, int(face_w * 0.30))
        center_max_area = max(3400, int(face_w * face_h * 0.14))
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom_y = y + h
            comp_cx = float(centroids[idx][0])
            offset = abs(comp_cx - cx)
            if area < 12 or area > max_area:
                continue
            if w > max_width or h < min_height:
                continue
            if y < int(cutoff_y + face_h * 0.02):
                continue
            if bottom_y < int(cutoff_y + face_h * 0.34):
                continue
            if bottom_y > int(cutoff_y + face_h * 1.02):
                continue
            if offset > max_offset:
                continue
            fill_ratio = float(area) / float(max(w * h, 1))
            if fill_ratio > 0.92 and area > max(80, int(face_w * face_h * 0.008)):
                continue
            if offset <= center_offset and (w > center_max_width or area > center_max_area):
                continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (5, 15) if hair_length == "short" else (5, 11),
            ),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, zone_u8)
        if int((keep_u8 > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)

        return cv2.GaussianBlur(
            keep_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=1.8,
            sigmaY=3.2,
        ).astype(np.float32)

    def _build_preclean_cloth_hair_cleanup_mask(
        self,
        img_rgb: np.ndarray,
        removal_mask: Optional[np.ndarray],
        cloth_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
    ) -> np.ndarray:
        if hair_length not in ("short", "medium"):
            return np.zeros(img_rgb.shape[:2], dtype=np.float32)

        H, W = img_rgb.shape[:2]
        if removal_mask is None or cloth_mask is None:
            return np.zeros((H, W), dtype=np.float32)
        if removal_mask.shape != (H, W) or cloth_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (13, 13) if hair_length == "short" else (11, 11),
            ),
            iterations=1,
        )
        zone_u8 = cv2.bitwise_and(removal_u8, cloth_u8)
        if int((zone_u8 > 0).sum()) < 80:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(cutoff_y + face_h * 0.02))
        bottom = min(H, int(cutoff_y + face_h * (1.76 if hair_length == "short" else 1.08)))
        left = max(0, int(x1 - face_w * (1.46 if hair_length == "short" else 1.32)))
        right = min(W, int(x2 + face_w * (1.46 if hair_length == "short" else 1.32)))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
        if int((zone_u8 > 0).sum()) < 80:
            return np.zeros((H, W), dtype=np.float32)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        blackhat = cv2.morphologyEx(
            gray.astype(np.uint8),
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 21) if hair_length == "short" else (7, 17),
            ),
        )
        darkness_u8 = (
            (
                (
                    (gray < (208.0 if hair_length == "short" else 198.0))
                    & (blur < (214.0 if hair_length == "short" else 206.0))
                    & (sat < (142.0 if hair_length == "short" else 132.0))
                )
                | (blackhat > (7 if hair_length == "short" else 8))
            ).astype(np.uint8)
            * 255
        )
        guided_zone_u8 = cv2.bitwise_and(zone_u8, darkness_u8)
        if hair_length != "short" or int((guided_zone_u8 > 0).sum()) >= 120:
            zone_u8 = guided_zone_u8
        zone_u8 = cv2.morphologyEx(
            zone_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        zone_u8 = cv2.morphologyEx(
            zone_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (5, 11) if hair_length == "short" else (5, 9),
            ),
        )
        if int((zone_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
        min_area = max(80, int(face_w * face_h * 0.006))
        max_area = max(
            22000 if hair_length == "short" else 14000,
            int(face_w * face_h * (0.62 if hair_length == "short" else 0.26)),
        )
        min_height = max(44, int(face_h * (0.16 if hair_length == "short" else 0.16)))
        max_width = max(
            188 if hair_length == "short" else 136,
            int(face_w * (1.06 if hair_length == "short" else 0.62)),
        )
        max_offset = max(
            340 if hair_length == "short" else 280,
            int(face_w * (1.52 if hair_length == "short" else 1.22)),
        )
        center_guard_offset = max(18, int(face_w * 0.18))
        center_guard_width = max(108, int(face_w * 0.76))
        center_guard_area = max(5200, int(face_w * face_h * 0.34))
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
            if y < int(cutoff_y + face_h * 0.02):
                continue
            if bottom_y < int(cutoff_y + face_h * 0.30):
                continue
            if bottom_y > int(cutoff_y + face_h * (1.72 if hair_length == "short" else 1.16)):
                continue
            if abs(comp_cx - cx) > max_offset:
                continue
            fill_ratio = float(area) / float(max(w * h, 1))
            if (
                hair_length == "short"
                and abs(comp_cx - cx) <= center_guard_offset
                and (w > center_guard_width or area > center_guard_area)
            ):
                continue
            if fill_ratio > (0.96 if hair_length == "short" else 0.86) and area > max(180, int(face_w * face_h * 0.012)):
                continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (15, 31) if hair_length == "short" else (5, 11),
            ),
            iterations=1,
        )
        keep_gate_u8 = cv2.bitwise_or(
            cloth_u8,
            cv2.dilate(
                removal_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (11, 19) if hair_length == "short" else (7, 13),
                ),
                iterations=1,
            ),
        )
        keep_u8 = cv2.bitwise_and(keep_u8, keep_gate_u8)
        keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
        if int((keep_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        return cv2.GaussianBlur(
            keep_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=3.6 if hair_length == "short" else 2.0,
            sigmaY=6.2 if hair_length == "short" else 3.4,
        ).astype(np.float32)

    def _build_residual_strand_cleanup_mask(
        self,
        img_rgb: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        removal_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
        final_hair_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if hair_length not in ("short", "medium"):
            return np.zeros(img_rgb.shape[:2], dtype=np.float32)

        H, W = img_rgb.shape[:2]
        if cloth_mask is None or removal_mask is None:
            return np.zeros((H, W), dtype=np.float32)
        if cloth_mask.shape != (H, W) or removal_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        zone_u8 = cv2.bitwise_and(cloth_u8, removal_u8)
        if int((zone_u8 > 0).sum()) < 30:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(cutoff_y + face_h * (0.16 if hair_length == "short" else 0.12)))
        bottom = min(H, int(cutoff_y + face_h * (1.18 if hair_length == "short" else 1.06)))
        left = max(0, int(x1 - face_w * 1.18))
        right = min(W, int(x2 + face_w * 1.18))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
        if int((zone_u8 > 0).sum()) < 30:
            return np.zeros((H, W), dtype=np.float32)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=4.2, sigmaY=4.2)
        blackhat = cv2.morphologyEx(
            gray.astype(np.uint8),
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 13) if hair_length == "short" else (5, 11)),
        )
        candidate_u8 = (
            (
                ((gray < (192.0 if hair_length == "short" else 188.0)) & ((blur - gray) > (1.4 if hair_length == "short" else 1.7)))
                | (blackhat > (5 if hair_length == "short" else 6))
            ).astype(np.uint8)
            * 255
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, zone_u8)
        if final_hair_mask is not None and final_hair_mask.shape == (H, W):
            final_hair_u8 = cv2.dilate(
                (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 19)),
                iterations=1,
            )
            candidate_u8 = cv2.bitwise_and(candidate_u8, cv2.bitwise_not(final_hair_u8))
        candidate_u8 = cv2.morphologyEx(
            candidate_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        if int((candidate_u8 > 0).sum()) < 8:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
        max_area = max(900, int(face_w * face_h * 0.060))
        max_width = max(24, int(face_w * 0.18))
        max_height = max(96, int(face_h * 0.42))
        max_offset = max(160, int(face_w * 0.80))
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom_y = y + h
            comp_cx = float(centroids[idx][0])
            if area < 8 or area > max_area:
                continue
            if w > max_width or h > max_height:
                continue
            if bottom_y < int(cutoff_y + face_h * 0.30):
                continue
            if abs(comp_cx - cx) > max_offset:
                continue
            fill_ratio = float(area) / float(max(w * h, 1))
            if fill_ratio > 0.74 and area > 36:
                continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 8:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, zone_u8)
        if int((keep_u8 > 0).sum()) < 8:
            return np.zeros((H, W), dtype=np.float32)

        return cv2.GaussianBlur(
            keep_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=1.8,
            sigmaY=2.8,
        ).astype(np.float32)

    def _build_final_hair_lane_cleanup_mask(
        self,
        final_hair_mask: Optional[np.ndarray],
        cloth_mask: Optional[np.ndarray],
        removal_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
    ) -> np.ndarray:
        if hair_length not in ("short", "medium"):
            shape = final_hair_mask.shape[:2] if isinstance(final_hair_mask, np.ndarray) else (0, 0)
            return np.zeros(shape, dtype=np.float32)
        if final_hair_mask is None or cloth_mask is None or removal_mask is None:
            shape = final_hair_mask.shape[:2] if isinstance(final_hair_mask, np.ndarray) else (0, 0)
            return np.zeros(shape, dtype=np.float32)

        H, W = final_hair_mask.shape[:2]
        if cloth_mask.shape != (H, W) or removal_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.22).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
            iterations=1,
        )
        cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        zone_u8 = cv2.bitwise_and(hair_u8, cloth_u8)
        zone_u8 = cv2.bitwise_and(zone_u8, removal_u8)
        if int((zone_u8 > 0).sum()) < 24:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(cutoff_y + face_h * 0.14))
        bottom = min(H, int(cutoff_y + face_h * 1.22))
        left = max(0, int(x1 - face_w * 1.04))
        right = min(W, int(x2 + face_w * 1.04))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
        if int((zone_u8 > 0).sum()) < 24:
            return np.zeros((H, W), dtype=np.float32)

        zone_u8 = cv2.morphologyEx(
            zone_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        zone_u8 = cv2.morphologyEx(
            zone_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11)),
        )

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
        max_area = max(2200, int(face_w * face_h * 0.10))
        max_width = max(58, int(face_w * 0.30))
        min_height = max(24, int(face_h * 0.12))
        max_offset = max(240, int(face_w * 1.00))
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom_y = y + h
            comp_cx = float(centroids[idx][0])
            if area < 20 or area > max_area:
                continue
            if w > max_width or h < min_height:
                continue
            if bottom_y < int(cutoff_y + face_h * 0.26):
                continue
            if abs(comp_cx - cx) > max_offset:
                continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 24:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 13)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
        return cv2.GaussianBlur(
            keep_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=1.6,
            sigmaY=2.8,
        ).astype(np.float32)

    def _build_short_bob_tail_suppress_mask(
        self,
        img_rgb: np.ndarray,
        final_hair_mask: Optional[np.ndarray],
        cloth_mask: Optional[np.ndarray],
        removal_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
    ) -> np.ndarray:
        if hair_length != "short":
            shape = final_hair_mask.shape[:2] if isinstance(final_hair_mask, np.ndarray) else (0, 0)
            return np.zeros(shape, dtype=np.float32)
        if final_hair_mask is None:
            shape = final_hair_mask.shape[:2] if isinstance(final_hair_mask, np.ndarray) else (0, 0)
            return np.zeros(shape, dtype=np.float32)

        H, W = final_hair_mask.shape[:2]
        if img_rgb.shape[:2] != (H, W):
            return np.zeros((H, W), dtype=np.float32)
        if removal_mask is not None and removal_mask.shape != (H, W):
            removal_mask = None
        if cloth_mask is not None and cloth_mask.shape != (H, W):
            cloth_mask = None

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.14).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21)),
            iterations=1,
        )
        removal_u8 = None
        if removal_mask is not None:
            removal_u8 = cv2.dilate(
                (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (47, 47)),
                iterations=1,
            )
        cloth_u8 = None
        if cloth_mask is not None:
            cloth_u8 = cv2.dilate(
                (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
                iterations=1,
            )

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(max(y2 + face_h * 0.12, cutoff_y + face_h * 0.08)))
        bottom = min(H, int(cutoff_y + face_h * 1.56))
        left = max(0, int(x1 - face_w * 1.38))
        right = min(W, int(x2 + face_w * 1.38))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255

        lane_u8 = np.zeros((H, W), dtype=np.uint8)
        left_lane_left = max(left, int(x1 - face_w * 0.72))
        left_lane_right = min(right, int(x1 + face_w * 0.14))
        right_lane_left = max(left, int(x2 - face_w * 0.14))
        right_lane_right = min(right, int(x2 + face_w * 0.72))
        if left_lane_left < left_lane_right:
            lane_u8[top:bottom, left_lane_left:left_lane_right] = 255
        if right_lane_left < right_lane_right:
            lane_u8[top:bottom, right_lane_left:right_lane_right] = 255
        center_lane_top = min(bottom, int(y2 + face_h * 0.34))
        center_half = max(16, int(face_w * 0.12))
        center_x1 = max(left, int(cx - center_half))
        center_x2 = min(right, int(cx + center_half))
        if center_lane_top < bottom and center_x1 < center_x2:
            lane_u8[center_lane_top:bottom, center_x1:center_x2] = 255

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=4.8, sigmaY=6.4)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        blackhat = cv2.morphologyEx(
            gray.astype(np.uint8),
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 19)),
        )
        dark_tail_u8 = (
            (
                ((gray < 168.0) & (blur < 176.0) & (sat < 124.0))
                | (blackhat > 9)
            ).astype(np.uint8)
            * 255
        )

        candidate_u8 = cv2.bitwise_and(hair_u8, dark_tail_u8)
        if removal_u8 is not None:
            candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(dark_tail_u8, removal_u8))
        else:
            candidate_u8 = cv2.bitwise_or(candidate_u8, dark_tail_u8)
        if cloth_u8 is not None:
            cloth_support_u8 = cv2.dilate(
                cloth_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)),
                iterations=1,
            )
            deep_lane_u8 = lane_u8.copy()
            deep_lane_u8[:max(0, int(y2 + face_h * 0.12)), :] = 0
            candidate_u8 = cv2.bitwise_and(
                candidate_u8,
                cv2.bitwise_or(cloth_support_u8, deep_lane_u8),
            )
        candidate_u8 = cv2.bitwise_and(candidate_u8, corridor_u8)
        candidate_u8 = cv2.bitwise_and(candidate_u8, lane_u8)
        if int((candidate_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        lower_start = min(H, int(y2 + face_h * 0.18))
        if lower_start < H:
            candidate_u8[:lower_start, :] = 0
        candidate_u8 = cv2.morphologyEx(
            candidate_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        candidate_u8 = cv2.morphologyEx(
            candidate_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
        )
        if int((candidate_u8 > 0).sum()) < 36:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
        center_keepout = max(12, int(face_w * 0.12))
        min_height = max(16, int(face_h * 0.08))
        min_area = max(24, int(face_w * face_h * 0.0011))
        max_area = max(24000, int(face_w * face_h * 0.48))
        max_width = max(146, int(face_w * 0.94))
        deep_bottom = int(y2 + face_h * 0.24)
        deepest_bottom = int(y2 + face_h * 0.38)
        max_offset = max(float(face_w * 1.30), 1.0)

        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom_y = y + h
            comp_cx = float(centroids[idx][0])
            offset = abs(comp_cx - cx)
            if area < min_area or h < min_height:
                continue
            if area > max_area or w > max_width:
                continue
            if bottom_y < deep_bottom:
                continue

            side_ratio = float(np.clip((offset - center_keepout) / max(max_offset - center_keepout, 1.0), 0.0, 1.0))
            truncate_y = int(y2 + face_h * (0.06 + 0.10 * side_ratio))
            truncate_y = max(truncate_y, lower_start)
            comp_u8 = (labels == idx).astype(np.uint8) * 255
            if truncate_y > 0:
                comp_u8[:truncate_y, :] = 0
            lane_overlap = int((cv2.bitwise_and(comp_u8, lane_u8) > 0).sum())
            if cloth_u8 is not None:
                cloth_overlap = int((cv2.bitwise_and(comp_u8, cloth_u8) > 0).sum())
                if cloth_overlap < 10 and lane_overlap < max(18, int(area * 0.10)) and bottom_y < int(cutoff_y + face_h * 0.72):
                    continue
                if offset < center_keepout and bottom_y < deepest_bottom and cloth_overlap < 12:
                    continue
                if cloth_overlap > 0:
                    comp_u8 = cv2.dilate(
                        comp_u8,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 17)),
                        iterations=1,
                    )
            elif offset < center_keepout and bottom_y < deepest_bottom:
                continue
            keep_u8 = cv2.bitwise_or(keep_u8, comp_u8)

        if int((keep_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.morphologyEx(
            keep_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 19)),
        )
        keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
        if int((keep_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        return cv2.GaussianBlur(
            keep_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=4.0,
            sigmaY=6.2,
        ).astype(np.float32)

    def _build_short_final_side_lane_refine_mask(
        self,
        img_rgb: np.ndarray,
        final_hair_mask: Optional[np.ndarray],
        cloth_mask: Optional[np.ndarray],
        removal_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
    ) -> np.ndarray:
        if hair_length != "short":
            return np.zeros(img_rgb.shape[:2], dtype=np.float32)

        H, W = img_rgb.shape[:2]
        if removal_mask is None:
            return np.zeros((H, W), dtype=np.float32)
        if removal_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)
        if final_hair_mask is not None and final_hair_mask.shape != (H, W):
            final_hair_mask = None

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(max(float(y2) - face_h * 0.08, cutoff_y + face_h * 0.04)))
        bottom = min(H, int(y2 + face_h * 1.16))
        left = max(0, int(x1 - face_w * 1.26))
        right = min(W, int(x2 + face_w * 1.26))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255

        hair_u8 = np.zeros((H, W), dtype=np.uint8)
        if final_hair_mask is not None:
            hair_u8 = cv2.dilate(
                (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 19)),
                iterations=1,
            )
        removal_u8 = cv2.dilate(
            (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            iterations=1,
        )
        candidate_u8 = cv2.bitwise_and(hair_u8, corridor_u8)
        candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(hair_u8, removal_u8))

        side_tail_u8 = (
            self._build_side_tail_cleanup_mask(
                removal_mask=removal_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            ) > 0.08
        ).astype(np.uint8) * 255
        candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(side_tail_u8, corridor_u8))

        short_bob_tail_u8 = np.zeros((H, W), dtype=np.uint8)
        if final_hair_mask is not None:
            short_bob_tail_u8 = (
                self._build_short_bob_tail_suppress_mask(
                    img_rgb=img_rgb,
                    final_hair_mask=final_hair_mask,
                    cloth_mask=cloth_mask,
                    removal_mask=removal_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                ) > 0.08
            ).astype(np.uint8) * 255
            candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(short_bob_tail_u8, corridor_u8))

        dark_tail_u8 = (
            self._build_dark_tail_residual_mask(
                img_rgb=img_rgb,
                removal_mask=removal_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            ) > 0.08
        ).astype(np.uint8) * 255
        if int((dark_tail_u8 > 0).sum()) > 0:
            dark_tail_u8 = cv2.dilate(
                dark_tail_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 17)),
                iterations=1,
            )
            candidate_u8 = cv2.bitwise_or(
                candidate_u8,
                cv2.bitwise_and(cv2.bitwise_and(dark_tail_u8, removal_u8), corridor_u8),
            )

        cloth_u8 = np.zeros((H, W), dtype=np.uint8)
        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_u8 = cv2.dilate(
                (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
                iterations=1,
            )
            lower_tail_u8 = (
                self._build_short_lower_tail_cleanup_mask(
                    img_rgb=img_rgb,
                    cloth_mask=cloth_mask,
                    removal_mask=removal_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                ) > 0.08
            ).astype(np.uint8) * 255
            candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(lower_tail_u8, corridor_u8))

        if int((candidate_u8 > 0).sum()) > 0 and int((cloth_u8 > 0).sum()) > 0:
            cloth_support_u8 = cv2.bitwise_and(
                cv2.dilate(
                    candidate_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 25)),
                    iterations=1,
                ),
                cloth_u8,
            )
            candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(cloth_support_u8, corridor_u8))

        candidate_u8 = cv2.morphologyEx(
            candidate_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21)),
        )
        candidate_u8 = cv2.morphologyEx(
            candidate_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, corridor_u8)
        if int((candidate_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
        min_area = max(28, int(face_w * face_h * 0.0014))
        max_area = max(24000, int(face_w * face_h * 0.44))
        min_height = max(18, int(face_h * 0.08))
        max_width = max(196, int(face_w * 1.18))
        center_keepout = max(14, int(face_w * 0.12))
        deep_start = int(y2 + face_h * 0.10)
        deep_center_start = int(y2 + face_h * 0.32)

        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom_y = y + h
            comp_cx = float(centroids[idx][0])
            offset = abs(comp_cx - cx)
            if area < min_area or area > max_area:
                continue
            if h < min_height or w > max_width:
                continue
            if y < top or bottom_y < deep_start:
                continue
            cloth_overlap = int((cv2.bitwise_and((labels == idx).astype(np.uint8) * 255, cloth_u8) > 0).sum())
            if offset < center_keepout and bottom_y < deep_center_start and cloth_overlap < 10:
                continue
            if offset < center_keepout and w > max(92, int(face_w * 0.56)) and cloth_overlap < 10:
                continue

            comp_u8 = (labels == idx).astype(np.uint8) * 255
            if int((cloth_u8 > 0).sum()) > 0 and cloth_overlap > 0:
                comp_u8 = cv2.dilate(
                    comp_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21)),
                    iterations=1,
                )
            keep_u8 = cv2.bitwise_or(keep_u8, cv2.bitwise_and(comp_u8, corridor_u8))

        if int((keep_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.morphologyEx(
            keep_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21)),
        )
        keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
        if int((keep_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        return cv2.GaussianBlur(
            keep_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=3.4,
            sigmaY=5.8,
        ).astype(np.float32)

    def _build_short_lower_tail_cleanup_mask(
        self,
        img_rgb: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        removal_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
        final_hair_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if hair_length != "short":
            return np.zeros(img_rgb.shape[:2], dtype=np.float32)

        H, W = img_rgb.shape[:2]
        if cloth_mask is None or removal_mask is None:
            return np.zeros((H, W), dtype=np.float32)
        if cloth_mask.shape != (H, W) or removal_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)
        if final_hair_mask is not None and final_hair_mask.shape != (H, W):
            final_hair_mask = None

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )
        removal_u8 = cv2.dilate(
            (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )
        hair_u8 = np.zeros((H, W), dtype=np.uint8)
        if final_hair_mask is not None:
            hair_u8 = cv2.dilate(
                (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.14).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 27)),
                iterations=1,
            )
        zone_u8 = cv2.bitwise_and(cloth_u8, cv2.bitwise_or(removal_u8, hair_u8))
        if int((zone_u8 > 0).sum()) < 80:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, min(int(cutoff_y - face_h * 0.02), int(y2 - face_h * 0.04)))
        bottom = min(H, int(cutoff_y + face_h * 1.82))
        left = max(0, int(x1 - face_w * 1.52))
        right = min(W, int(x2 + face_w * 1.52))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
        if int((zone_u8 > 0).sum()) < 80:
            return np.zeros((H, W), dtype=np.float32)

        lane_u8 = np.zeros((H, W), dtype=np.uint8)
        left_lane_left = max(left, int(x1 - face_w * 0.78))
        left_lane_right = min(right, int(x1 + face_w * 0.16))
        right_lane_left = max(left, int(x2 - face_w * 0.16))
        right_lane_right = min(right, int(x2 + face_w * 0.78))
        if left_lane_left < left_lane_right:
            lane_u8[top:bottom, left_lane_left:left_lane_right] = 255
        if right_lane_left < right_lane_right:
            lane_u8[top:bottom, right_lane_left:right_lane_right] = 255
        center_lane_top = min(bottom, int(y2 + face_h * 0.42))
        center_half = max(14, int(face_w * 0.10))
        center_x1 = max(left, int(cx - center_half))
        center_x2 = min(right, int(cx + center_half))
        if center_lane_top < bottom and center_x1 < center_x2:
            lane_u8[center_lane_top:bottom, center_x1:center_x2] = 255
        lane_zone_u8 = cv2.bitwise_and(zone_u8, lane_u8)
        if int((lane_zone_u8 > 0).sum()) >= 60:
            zone_u8 = lane_zone_u8

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        blackhat = cv2.morphologyEx(
            gray.astype(np.uint8),
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 23)),
        )

        ring_u8 = cv2.subtract(
            cv2.dilate(zone_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (39, 39)), iterations=1),
            cv2.dilate(zone_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1),
        )
        ring_u8 = cv2.bitwise_and(ring_u8, cloth_u8)
        ring_u8 = cv2.bitwise_and(ring_u8, cv2.bitwise_not(cv2.bitwise_or(removal_u8, hair_u8)))
        ring_u8 = cv2.bitwise_and(ring_u8, corridor_u8)
        if int((ring_u8 > 0).sum()) < 60:
            ring_u8 = cv2.bitwise_and(cv2.bitwise_and(cloth_u8, corridor_u8), cv2.bitwise_not(zone_u8))
        if int((ring_u8 > 0).sum()) < 60:
            return np.zeros((H, W), dtype=np.float32)

        ref_gray = float(np.median(gray[ring_u8 > 0]))
        ref_sat = float(np.median(sat[ring_u8 > 0]))
        if ref_gray < 142.0:
            fallback_ring_u8 = cv2.bitwise_and(cloth_u8, corridor_u8)
            fallback_ring_u8 = cv2.bitwise_and(fallback_ring_u8, cv2.bitwise_not(zone_u8))
            if int((fallback_ring_u8 > 0).sum()) >= 60:
                ref_gray = max(ref_gray, float(np.median(gray[fallback_ring_u8 > 0])))
                ref_sat = min(ref_sat, float(np.median(sat[fallback_ring_u8 > 0])))
        if ref_gray < 120.0 and int((hair_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        tone_ceiling = max(108.0, ref_gray - 16.0)
        blur_ceiling = max(118.0, ref_gray - 8.0)
        sat_ceiling = max(118.0, ref_sat + 24.0)
        candidate_u8 = (
            (
                (
                    (gray < tone_ceiling)
                    & (blur < blur_ceiling)
                    & (sat < sat_ceiling)
                    & (lap < 52.0)
                )
                | (blackhat > 9)
            ).astype(np.uint8)
            * 255
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, zone_u8)
        if int((hair_u8 > 0).sum()) > 0:
            hair_dark_seed_u8 = cv2.bitwise_and(
                hair_u8,
                cv2.bitwise_and(
                    zone_u8,
                    cv2.bitwise_or(
                        ((gray < (tone_ceiling + 10.0)).astype(np.uint8) * 255),
                        (blackhat > 8).astype(np.uint8) * 255,
                    ),
                ),
            )
            candidate_u8 = cv2.bitwise_or(candidate_u8, hair_dark_seed_u8)
            candidate_u8 = cv2.bitwise_or(
                candidate_u8,
                cv2.bitwise_and((blackhat > 8).astype(np.uint8) * 255, hair_dark_seed_u8),
            )
        candidate_u8 = cv2.morphologyEx(
            candidate_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        candidate_u8 = cv2.morphologyEx(
            candidate_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 23)),
        )
        if int((candidate_u8 > 0).sum()) < 50:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
        min_area = max(80, int(face_w * face_h * 0.004))
        max_area = max(26000, int(face_w * face_h * 0.62))
        min_height = max(40, int(face_h * 0.16))
        max_width = max(138, int(face_w * 0.82))
        max_offset = max(360, int(face_w * 1.64))
        center_keepout = max(16, int(face_w * 0.16))
        center_max_width = max(96, int(face_w * 0.50))
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom_y = y + h
            comp_cx = float(centroids[idx][0])
            offset = abs(comp_cx - cx)
            if area < min_area or area > max_area:
                continue
            if h < min_height or w > max_width:
                continue
            if y < top:
                continue
            if bottom_y < int(cutoff_y + face_h * 0.28):
                continue
            if offset > max_offset:
                continue
            comp_u8 = (labels == idx).astype(np.uint8) * 255
            cloth_overlap = int((cv2.bitwise_and(comp_u8, cloth_u8) > 0).sum())
            hair_overlap = int((cv2.bitwise_and(comp_u8, hair_u8) > 0).sum())
            lane_overlap = int((cv2.bitwise_and(comp_u8, lane_u8) > 0).sum())
            if offset <= center_keepout and w > center_max_width and hair_overlap < 20:
                continue
            if lane_overlap < max(20, int(area * 0.12)) and hair_overlap < 20 and bottom_y < int(cutoff_y + face_h * 0.82):
                continue
            if cloth_overlap < 10 and hair_overlap < 10:
                continue
            if cloth_overlap > 0 or hair_overlap > 0:
                comp_u8 = cv2.dilate(
                    comp_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21)),
                    iterations=1,
                )
            keep_u8 = cv2.bitwise_or(keep_u8, cv2.bitwise_and(comp_u8, corridor_u8))

        if int((keep_u8 > 0).sum()) < 50:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 25)),
            iterations=1,
        )
        keep_u8 = cv2.morphologyEx(
            keep_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 19)),
        )
        keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
        if int((keep_u8 > 0).sum()) < 50:
            return np.zeros((H, W), dtype=np.float32)

        return cv2.GaussianBlur(
            keep_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=4.2,
            sigmaY=6.6,
        ).astype(np.float32)

    def _build_dark_lane_cleanup_mask(
        self,
        img_rgb: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        removal_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
    ) -> np.ndarray:
        if hair_length not in ("short", "medium"):
            return np.zeros(img_rgb.shape[:2], dtype=np.float32)

        H, W = img_rgb.shape[:2]
        if cloth_mask is None or removal_mask is None:
            return np.zeros((H, W), dtype=np.float32)
        if cloth_mask.shape != (H, W) or removal_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        zone_u8 = cv2.bitwise_and(cloth_u8, removal_u8)
        if int((zone_u8 > 0).sum()) < 30:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        top = max(0, int(cutoff_y + face_h * 0.18))
        bottom = min(H, int(cutoff_y + face_h * 1.24))
        left = max(0, int(x1 - face_w * 1.04))
        right = min(W, int(x2 + face_w * 1.04))
        if top >= bottom or left >= right:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[top:bottom, left:right] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
        if int((zone_u8 > 0).sum()) < 30:
            return np.zeros((H, W), dtype=np.float32)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        candidate_u8 = (
            (gray < (170.0 if hair_length == "short" else 164.0))
            & (sat < (95.0 if hair_length == "short" else 100.0))
            & (lap < (20.0 if hair_length == "short" else 22.0))
        ).astype(np.uint8) * 255
        candidate_u8 = cv2.bitwise_and(candidate_u8, zone_u8)
        candidate_u8 = cv2.morphologyEx(
            candidate_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        if int((candidate_u8 > 0).sum()) < 24:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
        max_area = max(3000, int(face_w * face_h * 0.12))
        max_width = max(82, int(face_w * 0.42))
        min_height = max(26, int(face_h * 0.12))
        max_offset = max(260, int(face_w * 1.08))
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom_y = y + h
            comp_cx = float(centroids[idx][0])
            if area < 24 or area > max_area:
                continue
            if w > max_width or h < min_height:
                continue
            if bottom_y < int(cutoff_y + face_h * 0.30):
                continue
            if abs(comp_cx - cx) > max_offset:
                continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 24:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 15)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
        return cv2.GaussianBlur(
            keep_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=1.8,
            sigmaY=3.0,
        ).astype(np.float32)

    def _build_side_tail_cleanup_mask(
        self,
        removal_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str = "short",
    ) -> np.ndarray:
        """
        bob 아래로 남는 side-tail blob만 분리한 cleanup 힌트 마스크.
        넓은 가로 band는 제외하고, 옆으로 내려오는 잔머리 성분만 유지한다.
        """
        H, W = removal_mask.shape[:2]
        if removal_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        base = np.clip(removal_mask.astype(np.float32), 0.0, 1.0).copy()
        top_keep = max(0, int(cutoff_y - face_h * 0.04))
        base[:top_keep, :] = 0.0

        base_thresh = 0.42 if hair_length == "short" else 0.48
        base_u8 = (base > base_thresh).astype(np.uint8) * 255
        if int((base_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        corridor_ratio = 1.45 if hair_length == "short" else 1.35
        x_min = max(0, int(x1 - face_w * corridor_ratio))
        x_max = min(W, int(x2 + face_w * corridor_ratio))
        if x_min >= x_max:
            return np.zeros((H, W), dtype=np.float32)

        corridor = np.zeros((H, W), dtype=np.uint8)
        corridor[:, x_min:x_max] = 255
        base_u8 = cv2.bitwise_and(base_u8, corridor)
        if int((base_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(base_u8, 8)
        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        width_cap = max(42, int(face_w * (1.55 if hair_length == "short" else 1.75)))
        min_height = max(18, int(face_h * (0.16 if hair_length == "short" else 0.20)))
        min_area = max(120, int(face_w * face_h * (0.003 if hair_length == "short" else 0.004)))
        center_keepout_half = max(18, int(face_w * 0.26))
        deep_start = min(H, int(cutoff_y + face_h * (0.14 if hair_length == "short" else 0.18)))
        deep_grace = max(20, int(face_h * 0.18))

        for idx in range(1, num_labels):
            x, y, w, h, area = stats[idx]
            if area < min_area or h < min_height or w > width_cap:
                continue

            comp_cx = float(centroids[idx][0])
            y_max = int(y + h)
            is_side_component = abs(comp_cx - cx) >= center_keepout_half
            is_deep_component = y_max >= (deep_start + deep_grace)
            if not is_side_component and not is_deep_component:
                continue

            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        close_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 13) if hair_length == "short" else (7, 11),
        )
        dilate_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 17) if hair_length == "short" else (9, 13),
        )
        keep_u8 = cv2.morphologyEx(keep_u8, cv2.MORPH_CLOSE, close_k)
        keep_u8 = cv2.dilate(keep_u8, dilate_k, iterations=1)
        keep_u8[:top_keep, :] = 0

        return (keep_u8 > 0).astype(np.float32)

    def _build_short_tail_core_mask(
        self,
        removal_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str = "short",
    ) -> np.ndarray:
        """
        side-tail cleanup 영역에서 중앙 폭만 남긴 좁은 core mask.
        넓은 어깨/옷 영역 재생성을 피하면서 하단 잔머리 꼬리만 다시 칠하도록 쓴다.
        """
        side_tail = self._build_side_tail_cleanup_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        H, W = side_tail.shape[:2]
        if removal_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        side_u8 = (side_tail > 0.08).astype(np.uint8) * 255
        if int((side_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        deep_start = min(
            H,
            int(cutoff_y + face_h * (0.04 if hair_length == "short" else 0.10)),
        )
        min_height = max(16, int(face_h * (0.10 if hair_length == "short" else 0.14)))
        min_area = max(60, int(face_w * face_h * (0.0012 if hair_length == "short" else 0.0018)))
        core_u8 = np.zeros((H, W), dtype=np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(side_u8, 8)
        for idx in range(1, num_labels):
            x, y, w, h, area = stats[idx]
            if area < min_area or h < min_height:
                continue

            comp_y1 = max(int(y), deep_start)
            comp_y2 = min(H, int(y + h))
            if comp_y2 - comp_y1 < min_height:
                continue

            comp_cx = float(centroids[idx][0])
            core_half = max(
                10,
                min(
                    int(round(w * (0.20 if hair_length == "short" else 0.26))),
                    int(face_w * (0.16 if hair_length == "short" else 0.22)),
                ),
            )
            rect_x1 = max(0, int(round(comp_cx - core_half)))
            rect_x2 = min(W, int(round(comp_cx + core_half)))
            if rect_x2 <= rect_x1:
                continue

            comp_u8 = (labels == idx).astype(np.uint8) * 255
            rect_u8 = np.zeros((H, W), dtype=np.uint8)
            rect_u8[comp_y1:comp_y2, rect_x1:rect_x2] = 255
            comp_core_u8 = cv2.bitwise_and(comp_u8, rect_u8)
            if int((comp_core_u8 > 0).sum()) < 24:
                continue

            comp_core_u8 = cv2.dilate(
                comp_core_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (9, 17) if hair_length == "short" else (7, 13),
                ),
                iterations=1,
            )
            core_u8 = cv2.bitwise_or(core_u8, comp_core_u8)

        return (core_u8 > 0).astype(np.float32)

    def _build_front_strand_cleanup_mask(
        self,
        removal_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str = "short",
        anchor_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        셔츠 앞쪽 중앙으로 떨어진 얇은 front strand를 위한 cleanup 마스크.
        side-tail 로직과 분리해서, 얼굴 중앙 아래의 가는 세로 성분만 보수적으로 남긴다.
        """
        H, W = removal_mask.shape[:2]
        if removal_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))
        target_cx = cx
        anchor_half_override = 0
        anchor_bottom = None
        lower_anchor_u8 = np.zeros((H, W), dtype=np.uint8)
        if anchor_mask is not None and anchor_mask.shape == (H, W):
            anchor_u8 = (np.clip(anchor_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
            anchor_u8[:max(0, int(cutoff_y - face_h * 0.06)), :] = 0
            if int((anchor_u8 > 0).sum()) > 0:
                lower_anchor_u8 = anchor_u8.copy()
                lower_anchor_u8[:min(H, int(cutoff_y + face_h * (0.18 if hair_length == "short" else 0.12))), :] = 0
                focus_u8 = lower_anchor_u8 if int((lower_anchor_u8 > 0).sum()) >= 10 else anchor_u8
                ys, xs = np.where(focus_u8 > 0)
                if ys.size and xs.size:
                    target_cx = int(np.clip(round(float(xs.mean())), 0, max(W - 1, 0)))
                    anchor_bottom = int(ys.max()) + 1
                    anchor_half_override = max(
                        14,
                        min(
                            max(28, int(face_w * 0.34)),
                            int(max(xs.max() - xs.min() + 1, 12) * 0.85),
                        ),
                    )

        zone = np.clip(removal_mask.astype(np.float32), 0.0, 1.0)
        zone[:max(0, int(cutoff_y - face_h * (0.10 if hair_length == "short" else 0.06))), :] = 0.0
        zone_u8 = (zone > 0.08).astype(np.uint8) * 255
        if int((zone_u8 > 0).sum()) < 30:
            return np.zeros((H, W), dtype=np.float32)

        front_half = max(14, int(face_w * (0.24 if hair_length == "short" else 0.18)))
        if anchor_half_override > 0:
            front_half = max(front_half, anchor_half_override)
        front_x1 = max(0, target_cx - front_half)
        front_x2 = min(W, target_cx + front_half)
        front_y1 = max(0, int(cutoff_y - face_h * (0.10 if hair_length == "short" else 0.06)))
        front_y2 = min(H, int(cutoff_y + face_h * (1.02 if hair_length == "short" else 0.82)))
        probe_y2 = min(H, int(cutoff_y + face_h * (1.34 if hair_length == "short" else 1.02)))
        if anchor_bottom is not None:
            front_y2 = min(H, max(front_y2, int(anchor_bottom + face_h * 0.10)))
        if front_x1 < front_x2 and front_y1 < probe_y2:
            front_probe_u8 = np.zeros((H, W), dtype=np.uint8)
            front_probe_u8[front_y1:probe_y2, front_x1:front_x2] = 255
            front_tail_u8 = cv2.bitwise_and(zone_u8, front_probe_u8)
            ys = np.where(front_tail_u8 > 0)[0]
            if ys.size:
                detected_bottom = int(ys.max()) + 1
                front_y2 = min(H, max(front_y2, int(detected_bottom + face_h * 0.06)))
        if front_x1 >= front_x2 or front_y1 >= front_y2:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        corridor_u8[front_y1:front_y2, front_x1:front_x2] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
        if int((zone_u8 > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)
        if hair_length == "short":
            filtered_u8 = np.zeros((H, W), dtype=np.uint8)
            anchor_lane_u8 = np.zeros((H, W), dtype=np.uint8)
            if int((lower_anchor_u8 > 0).sum()) > 0:
                anchor_lane_u8 = cv2.dilate(
                    lower_anchor_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 21)),
                    iterations=1,
                )
                anchor_lane_u8 = cv2.bitwise_and(anchor_lane_u8, corridor_u8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
            max_comp_width = max(22, int(face_w * 0.24))
            max_comp_area = max(160, int(face_w * face_h * 0.030))
            min_comp_height = max(18, int(face_h * 0.10))
            center_half = max(16, int(face_w * 0.22))
            if anchor_half_override > 0:
                center_half = max(center_half, anchor_half_override + 6)
            for idx in range(1, num_labels):
                x = int(stats[idx, cv2.CC_STAT_LEFT])
                y = int(stats[idx, cv2.CC_STAT_TOP])
                w = int(stats[idx, cv2.CC_STAT_WIDTH])
                h = int(stats[idx, cv2.CC_STAT_HEIGHT])
                area = int(stats[idx, cv2.CC_STAT_AREA])
                comp_cx = float(centroids[idx][0])
                if area < 10 or area > max_comp_area:
                    continue
                if w > max_comp_width or h < min_comp_height:
                    continue
                if abs(comp_cx - target_cx) > center_half:
                    continue
                if (y + h) > front_y2:
                    continue
                filtered_u8[labels == idx] = 255
            if int((filtered_u8 > 0).sum()) < 12:
                for idx in range(1, num_labels):
                    w = int(stats[idx, cv2.CC_STAT_WIDTH])
                    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
                    area = int(stats[idx, cv2.CC_STAT_AREA])
                    comp_cx = float(centroids[idx][0])
                    if area < 12 or area > max(220, int(face_w * face_h * 0.040)):
                        continue
                    if w > max(28, int(face_w * 0.30)) or h < max(20, int(face_h * 0.10)):
                        continue
                    if abs(comp_cx - target_cx) > max(18, max(anchor_half_override + 6, int(face_w * 0.26))):
                        continue
                    filtered_u8[labels == idx] = 255
            if int((anchor_lane_u8 > 0).sum()) > 0:
                filtered_u8 = cv2.bitwise_or(filtered_u8, anchor_lane_u8)
            zone_u8 = filtered_u8
            if int((zone_u8 > 0).sum()) < 12:
                return np.zeros((H, W), dtype=np.float32)
        keep_u8 = cv2.morphologyEx(
            zone_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 15) if hair_length == "short" else (5, 9)),
        )
        keep_u8 = cv2.erode(
            keep_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3) if hair_length == "short" else (3, 5)),
            iterations=1,
        )
        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 23) if hair_length == "short" else (7, 13)),
            iterations=1,
        )
        return (keep_u8 > 0).astype(np.float32)

    def _build_short_regen_tail_mask(
        self,
        img_rgb: np.ndarray,
        removal_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str = "short",
    ) -> np.ndarray:
        """
        short hair에서 cleanup 후 남는 하단 잔머리만 좁게 다시 생성하도록 하는 mask.
        side-tail core를 기본으로 하고, dark residual은 core 주변으로만 허용한다.
        """
        H, W = img_rgb.shape[:2]
        if removal_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        _, y1, _, y2 = face_bbox
        face_h = max(int(y2 - y1), 1)

        side_mask = self._build_side_tail_cleanup_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        side_u8 = (side_mask > 0.08).astype(np.uint8) * 255

        core_mask = self._build_short_tail_core_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        core_u8 = (core_mask > 0.08).astype(np.uint8) * 255

        dark_mask = self._build_dark_tail_residual_mask(
            img_rgb=img_rgb,
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        dark_u8 = (dark_mask > 0.08).astype(np.uint8) * 255

        shallow_band = np.zeros((H, W), dtype=np.uint8)
        shallow_top = max(0, int(cutoff_y - face_h * 0.02))
        shallow_bottom = min(H, int(cutoff_y + face_h * 0.92))
        if shallow_top < shallow_bottom:
            shallow_band[shallow_top:shallow_bottom, :] = 255
        side_u8 = cv2.bitwise_and(side_u8, shallow_band)
        core_u8 = cv2.bitwise_and(core_u8, shallow_band)
        dark_u8 = cv2.bitwise_and(dark_u8, shallow_band)

        side_u8 = cv2.morphologyEx(
            side_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 23)),
        )

        regen_u8 = cv2.bitwise_or(side_u8, core_u8)
        regen_u8 = cv2.bitwise_or(regen_u8, dark_u8)
        if int((regen_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        regen_u8 = cv2.morphologyEx(
            regen_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
        )
        return (regen_u8 > 0).astype(np.float32)

    def _build_lower_hair_tail_support_mask(
        self,
        *,
        img_rgb: np.ndarray,
        hair_mask: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        hair_length: str,
    ) -> np.ndarray:
        """
        밝은 옷 위로 내려온 얇은 앞머리 가닥이 SegFace/SAM2에서 빠질 때
        기존 hair mask 하단에 붙은 어두운 세로 성분만 보수적으로 다시 포함한다.
        """
        if hair_length not in ("short", "medium"):
            return np.zeros_like(hair_mask, dtype=np.float32)

        H, W = hair_mask.shape[:2]
        if img_rgb.shape[:2] != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        x_min = max(0, int(x1 - face_w * (1.28 if hair_length == "short" else 1.10)))
        x_max = min(W, int(x2 + face_w * (1.28 if hair_length == "short" else 1.10)))
        y_min = max(0, int(y2 - face_h * 0.03))
        y_max = min(H, int(y2 + face_h * (1.72 if hair_length == "short" else 1.24)))
        if x_min >= x_max or y_min >= y_max:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[y_min:y_max, x_min:x_max] = 255

        hair_u8 = (np.clip(hair_mask.astype(np.float32), 0.0, 1.0) > 0.35).astype(np.uint8) * 255
        anchor_u8 = cv2.dilate(
            hair_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (41, 41) if hair_length == "short" else (29, 29),
            ),
            iterations=1,
        )
        anchor_u8[:max(0, int(y2 - face_h * 0.18)), :] = 0
        anchor_u8 = cv2.bitwise_and(anchor_u8, corridor_u8)
        if int((anchor_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)
        anchor_support_u8 = cv2.dilate(
            anchor_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (35, 61) if hair_length == "short" else (27, 45),
            ),
            iterations=1,
        )
        anchor_support_u8 = cv2.bitwise_and(anchor_support_u8, corridor_u8)
        front_strand_zone_u8 = np.zeros((H, W), dtype=np.uint8)
        front_half = max(12, int(face_w * (0.22 if hair_length == "short" else 0.18)))
        front_x1 = max(0, int(0.5 * (x1 + x2)) - front_half)
        front_x2 = min(W, int(0.5 * (x1 + x2)) + front_half)
        front_y1 = max(0, int(y2 - face_h * 0.02))
        front_y2 = min(H, int(y2 + face_h * (0.82 if hair_length == "short" else 0.58)))
        if front_x1 < front_x2 and front_y1 < front_y2:
            front_strand_zone_u8[front_y1:front_y2, front_x1:front_x2] = 255

        support_zone_u8 = corridor_u8.copy()
        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_hint_u8 = cv2.dilate(
                (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                iterations=1,
            )
            support_zone_u8 = cv2.bitwise_and(
                support_zone_u8,
                cv2.bitwise_or(cloth_hint_u8, anchor_support_u8),
            )
            if int((support_zone_u8 > 0).sum()) < 60:
                support_zone_u8 = corridor_u8.copy()
            front_strand_zone_u8 = cv2.bitwise_and(front_strand_zone_u8, cloth_hint_u8)

        candidate_zone_u8 = cv2.bitwise_or(anchor_support_u8, front_strand_zone_u8)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=7.0, sigmaY=7.0)
        bright_bg = blur > (136.0 if hair_length == "short" else 132.0)
        dark_thresh = 140.0 if hair_length == "short" else 136.0
        contrast_thresh = 7.0 if hair_length == "short" else 6.0

        dark_u8 = (
            (gray < dark_thresh)
            & ((blur - gray) > contrast_thresh)
            & bright_bg
        ).astype(np.uint8) * 255
        dark_u8 = cv2.bitwise_and(dark_u8, support_zone_u8)
        dark_u8 = cv2.bitwise_and(dark_u8, candidate_zone_u8)
        if int((dark_u8 > 0).sum()) < 18:
            return np.zeros((H, W), dtype=np.float32)

        main_seed_u8 = cv2.dilate(
            hair_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (19, 25) if hair_length == "short" else (15, 21),
            ),
            iterations=1,
        )
        main_seed_u8[:max(0, int(y2 - face_h * 0.08)), :] = 0
        main_seed_u8 = cv2.bitwise_and(main_seed_u8, corridor_u8)
        main_seed_u8 = cv2.bitwise_or(main_seed_u8, anchor_u8)

        dark_u8 = cv2.morphologyEx(
            dark_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        dark_u8 = cv2.morphologyEx(
            dark_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 29) if hair_length == "short" else (5, 21),
            ),
        )
        dark_u8 = cv2.dilate(
            dark_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (11, 17) if hair_length == "short" else (9, 13),
            ),
            iterations=1,
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_u8, 8)
        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        max_area = max(
            280,
            min(
                int(H * W * 0.020),
                int(face_w * face_h * (0.42 if hair_length == "short" else 0.28)),
            ),
        )
        min_tail_bottom = int(y2 + face_h * (0.10 if hair_length == "short" else 0.08))
        min_height = max(14, int(face_h * (0.10 if hair_length == "short" else 0.08)))
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 20 or area > max_area:
                continue
            if h < min_height:
                continue
            if (y + h) < min_tail_bottom:
                continue
            comp_u8 = (labels == idx).astype(np.uint8) * 255
            dark_overlap = int((cv2.bitwise_and(comp_u8, dark_u8) > 0).sum())
            if dark_overlap < 8:
                continue
            seed_overlap = int((cv2.bitwise_and(comp_u8, main_seed_u8) > 0).sum())
            anchor_overlap = int((cv2.bitwise_and(comp_u8, anchor_support_u8) > 0).sum())
            front_overlap = int((cv2.bitwise_and(comp_u8, front_strand_zone_u8) > 0).sum())
            is_front_strand = (
                front_overlap >= 10
                and w <= max(26, int(face_w * 0.24))
                and h >= max(26, int(face_h * 0.16))
            )
            if seed_overlap < 12 and anchor_overlap < 10 and not is_front_strand:
                continue
            keep_u8 = cv2.bitwise_or(keep_u8, comp_u8)

        if int((keep_u8 > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)

        bridge_seed_u8 = cv2.dilate(
            main_seed_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 31) if hair_length == "short" else (7, 25),
            ),
            iterations=1,
        )
        bridge_keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 23) if hair_length == "short" else (5, 17),
            ),
            iterations=1,
        )
        bridge_u8 = cv2.bitwise_and(bridge_seed_u8, bridge_keep_u8)
        bridge_u8 = cv2.bitwise_and(bridge_u8, corridor_u8)
        keep_u8 = cv2.bitwise_or(keep_u8, bridge_u8)
        keep_u8 = cv2.morphologyEx(
            keep_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 41) if hair_length == "short" else (7, 31),
            ),
        )

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (13, 23) if hair_length == "short" else (9, 13),
            ),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
        return (keep_u8 > 0).astype(np.float32)

    def _build_center_chest_strand_support_mask(
        self,
        *,
        img_rgb: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        support_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
    ) -> np.ndarray:
        if hair_length not in ("short", "medium"):
            return np.zeros(img_rgb.shape[:2], dtype=np.float32)

        H, W = img_rgb.shape[:2]
        if cloth_mask is None or cloth_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = int(0.5 * (x1 + x2))

        lane_u8 = np.zeros((H, W), dtype=np.uint8)
        lane_half = max(18, int(face_w * (0.20 if hair_length == "short" else 0.17)))
        lane_x1 = max(0, cx - lane_half)
        lane_x2 = min(W, cx + lane_half)
        lane_y1 = max(0, int(cutoff_y - face_h * 0.04))
        lane_y2 = min(H, int(cutoff_y + face_h * (2.05 if hair_length == "short" else 1.42)))
        if lane_x1 >= lane_x2 or lane_y1 >= lane_y2:
            return np.zeros((H, W), dtype=np.float32)
        lane_u8[lane_y1:lane_y2, lane_x1:lane_x2] = 255

        cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )
        zone_u8 = cv2.bitwise_and(lane_u8, cloth_u8)
        if int((zone_u8 > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)

        support_hint_u8 = np.zeros((H, W), dtype=np.uint8)
        if support_mask is not None and support_mask.shape == (H, W):
            support_hint_u8 = (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255
            if int((support_hint_u8 > 0).sum()) > 0:
                support_hint_u8 = cv2.dilate(
                    support_hint_u8,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (11, 31) if hair_length == "short" else (9, 25),
                    ),
                    iterations=1,
                )
                support_hint_u8 = cv2.bitwise_and(support_hint_u8, zone_u8)
                support_hint_u8 = cv2.morphologyEx(
                    support_hint_u8,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (7, 31) if hair_length == "short" else (5, 23),
                    ),
                )

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=9.0, sigmaY=9.0)
        blackhat = cv2.morphologyEx(
            gray.astype(np.uint8),
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (11, 25) if hair_length == "short" else (9, 21),
            ),
        )
        blackhat_u8 = (
            blackhat > (9 if hair_length == "short" else 10)
        ).astype(np.uint8) * 255
        dark_u8 = (
            (gray < (162.0 if hair_length == "short" else 154.0))
            & ((blur - gray) > (2.2 if hair_length == "short" else 2.4))
            & (blur > (110.0 if hair_length == "short" else 104.0))
        ).astype(np.uint8) * 255
        dark_u8 = cv2.bitwise_or(dark_u8, blackhat_u8)
        dark_u8 = cv2.bitwise_and(dark_u8, zone_u8)
        if int((support_hint_u8 > 0).sum()) >= 8:
            dark_u8 = cv2.bitwise_and(dark_u8, support_hint_u8)
        if int((dark_u8 > 0).sum()) < 8:
            return np.zeros((H, W), dtype=np.float32)

        dark_u8 = cv2.morphologyEx(
            dark_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        dark_u8 = cv2.morphologyEx(
            dark_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11)),
        )
        if int((support_hint_u8 > 0).sum()) >= 8:
            dark_u8 = cv2.bitwise_or(dark_u8, support_hint_u8)
            dark_u8 = cv2.morphologyEx(
                dark_u8,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (7, 35) if hair_length == "short" else (5, 27),
                ),
            )

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_u8, 8)
        max_area = max(640, int(face_w * face_h * 0.12))
        max_width = max(26, int(face_w * 0.30))
        min_height = max(22, int(face_h * 0.12))
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            comp_cx = float(centroids[idx][0])
            if area < 8 or area > max_area:
                continue
            if w > max_width or h < min_height:
                continue
            if abs(comp_cx - cx) > max(16, int(face_w * 0.18)):
                continue
            if (y + h) < int(cutoff_y + face_h * 0.14):
                continue
            comp_u8 = (labels == idx).astype(np.uint8) * 255
            if int((support_hint_u8 > 0).sum()) >= 8:
                support_overlap = int((cv2.bitwise_and(comp_u8, support_hint_u8) > 0).sum())
                dark_overlap = int((cv2.bitwise_and(comp_u8, blackhat_u8) > 0).sum())
                if support_overlap < 10 or dark_overlap < 8:
                    continue
            keep_u8 = cv2.bitwise_or(keep_u8, comp_u8)

        if int((keep_u8 > 0).sum()) < 8:
            return np.zeros((H, W), dtype=np.float32)

        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 23) if hair_length == "short" else (7, 17),
            ),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, zone_u8)
        return (keep_u8 > 0).astype(np.float32)

    def _build_lower_tail_post_support_mask(
        self,
        *,
        support_mask: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
    ) -> np.ndarray:
        if hair_length not in ("short", "medium"):
            return np.zeros_like(support_mask, dtype=np.float32)

        H, W = support_mask.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)

        support_u8 = (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        if int((support_u8 > 0).sum()) < 12:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        x_min = max(0, int(x1 - face_w * (1.18 if hair_length == "short" else 1.04)))
        x_max = min(W, int(x2 + face_w * (1.18 if hair_length == "short" else 1.04)))
        y_min = max(0, int(cutoff_y - face_h * 0.03))
        y_max = min(H, int(cutoff_y + face_h * (1.04 if hair_length == "short" else 0.88)))
        if x_min >= x_max or y_min >= y_max:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[y_min:y_max, x_min:x_max] = 255
        support_u8 = cv2.bitwise_and(support_u8, corridor_u8)

        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_near_u8 = cv2.dilate(
                (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            )
            support_u8 = cv2.bitwise_and(support_u8, cloth_near_u8)

        if int((support_u8 > 0).sum()) < 12:
            return np.zeros((H, W), dtype=np.float32)

        support_u8 = cv2.morphologyEx(
            support_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        support_u8 = cv2.dilate(
            support_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 13) if hair_length == "short" else (5, 9),
            ),
            iterations=1,
        )

        filtered_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(support_u8, 8)
        max_area = max(640, int(face_w * face_h * (0.32 if hair_length == "short" else 0.22)))
        min_bottom = int(cutoff_y + face_h * 0.02)
        for idx in range(1, num_labels):
            y = int(stats[idx, cv2.CC_STAT_TOP])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 8 or area > max_area:
                continue
            if (y + h) < min_bottom:
                continue
            filtered_u8[labels == idx] = 255

        if int((filtered_u8 > 0).sum()) < 12:
            return np.zeros((H, W), dtype=np.float32)

        filtered_u8 = cv2.dilate(
            filtered_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (11, 19) if hair_length == "short" else (9, 13),
            ),
            iterations=1,
        )
        return (filtered_u8 > 0).astype(np.float32)

    def _build_lower_tail_removal_extension_mask(
        self,
        *,
        support_mask: np.ndarray,
        removal_mask: np.ndarray,
        cloth_mask: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str,
    ) -> np.ndarray:
        if hair_length not in ("short", "medium"):
            return np.zeros_like(support_mask, dtype=np.float32)

        H, W = support_mask.shape[:2]
        if removal_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)
        cx = float(0.5 * (x1 + x2))

        support_u8 = (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        if int((support_u8 > 0).sum()) < 12:
            return np.zeros((H, W), dtype=np.float32)

        corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        x_min = max(0, int(x1 - face_w * (0.72 if hair_length == "short" else 0.84)))
        x_max = min(W, int(x2 + face_w * (0.72 if hair_length == "short" else 0.84)))
        y_min = max(0, int(cutoff_y - face_h * 0.03))
        y_max = min(H, int(cutoff_y + face_h * (1.20 if hair_length == "short" else 1.08)))
        if x_min >= x_max or y_min >= y_max:
            return np.zeros((H, W), dtype=np.float32)
        corridor_u8[y_min:y_max, x_min:x_max] = 255
        support_u8 = cv2.bitwise_and(support_u8, corridor_u8)

        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_hint_u8 = cv2.dilate(
                (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            )
            support_u8 = cv2.bitwise_and(support_u8, cloth_hint_u8)

        base_hint_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8) * 255
        base_hint_u8[:cutoff_y, :] = 0
        if int((base_hint_u8 > 0).sum()) > 0:
            base_hint_u8 = cv2.dilate(
                base_hint_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (17, 23) if hair_length == "short" else (13, 17),
                ),
                iterations=1,
            )
            support_u8 = cv2.bitwise_and(support_u8, cv2.bitwise_or(base_hint_u8, corridor_u8))

        front_lane_u8 = np.zeros((H, W), dtype=np.uint8)
        center_half = max(12, int(face_w * (0.24 if hair_length == "short" else 0.20)))
        lane_x1 = max(0, int(0.5 * (x1 + x2)) - center_half)
        lane_x2 = min(W, int(0.5 * (x1 + x2)) + center_half)
        lane_y1 = max(0, int(cutoff_y - face_h * 0.04))
        lane_y2 = min(H, int(cutoff_y + face_h * (1.58 if hair_length == "short" else 0.96)))
        if lane_x1 < lane_x2 and lane_y1 < lane_y2:
            front_lane_u8[lane_y1:lane_y2, lane_x1:lane_x2] = 255

        filtered_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(support_u8, 8)
        max_area = max(360, int(face_w * face_h * (0.18 if hair_length == "short" else 0.14)))
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 8 or area > max_area:
                continue
            if (y + h) < int(cutoff_y + face_h * 0.02):
                continue
            comp_u8 = (labels == idx).astype(np.uint8) * 255
            base_overlap = int((cv2.bitwise_and(comp_u8, base_hint_u8) > 0).sum())
            front_overlap = int((cv2.bitwise_and(comp_u8, front_lane_u8) > 0).sum())
            comp_cx = float(centroids[idx][0])
            is_side_component = abs(comp_cx - cx) > max(18, int(face_w * 0.22))
            is_front_strand = (
                front_overlap >= 10
                and w <= max(22, int(face_w * 0.26))
                and h >= max(24, int(face_h * 0.18))
            )
            if base_overlap < 8 and not is_front_strand:
                continue
            if hair_length == "short":
                if w > max(24, int(face_w * 0.24)) and not is_front_strand:
                    continue
                if is_side_component:
                    if base_overlap < max(18, int(area * 0.18)) and front_overlap < 12:
                        continue
                    if area > max(92, int(face_w * face_h * 0.030)):
                        continue
            filtered_u8 = cv2.bitwise_or(filtered_u8, comp_u8)

        if int((filtered_u8 > 0).sum()) < 12:
            return np.zeros((H, W), dtype=np.float32)

        filtered_u8 = cv2.dilate(
            filtered_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 15) if hair_length == "short" else (5, 11),
            ),
            iterations=1,
        )
        filtered_u8 = cv2.bitwise_and(filtered_u8, corridor_u8)
        return (filtered_u8 > 0).astype(np.float32)

    def _build_dark_tail_residual_mask(
        self,
        img_rgb: np.ndarray,
        removal_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        hair_length: str = "short",
    ) -> np.ndarray:
        """
        SegFace가 놓친 회색/검은 잔여 tail blob을 밝기 기반으로 추가 검출한다.
        cleanup 이후 흐리게 남는 하단 머리 덩어리 제거용이다.
        """
        H, W = img_rgb.shape[:2]
        if removal_mask.shape != (H, W):
            return np.zeros((H, W), dtype=np.float32)

        x1, y1, x2, y2 = face_bbox
        face_w = max(int(x2 - x1), 1)
        face_h = max(int(y2 - y1), 1)

        zone = np.clip(removal_mask.astype(np.float32), 0.0, 1.0).copy()
        zone[:cutoff_y, :] = 0.0
        tail_hint = self._build_side_tail_cleanup_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        if float(tail_hint.sum()) > 20.0:
            zone = np.maximum(zone, tail_hint * 1.20)
        front_hint = np.zeros((H, W), dtype=np.float32)
        front_half = max(12, int(face_w * (0.24 if hair_length == "short" else 0.20)))
        front_x1 = max(0, int(0.5 * (x1 + x2)) - front_half)
        front_x2 = min(W, int(0.5 * (x1 + x2)) + front_half)
        front_y1 = max(0, int(cutoff_y - face_h * 0.04))
        front_y2 = min(H, int(cutoff_y + face_h * (0.90 if hair_length == "short" else 0.64)))
        if front_x1 < front_x2 and front_y1 < front_y2:
            front_hint[front_y1:front_y2, front_x1:front_x2] = 1.0
            front_hint = np.clip(front_hint * np.clip(removal_mask.astype(np.float32), 0.0, 1.0), 0.0, 1.0)
            if float(front_hint.sum()) > 20.0:
                zone = np.maximum(zone, front_hint * 1.10)

        zone_thresh = 0.20 if hair_length == "short" else 0.34
        zone_u8 = (zone > zone_thresh).astype(np.uint8) * 255
        deep_start = min(H, int(cutoff_y + face_h * (0.00 if hair_length == "short" else 0.10)))
        x_min = max(0, int(x1 - face_w * (1.35 if hair_length == "short" else 1.20)))
        x_max = min(W, int(x2 + face_w * (1.35 if hair_length == "short" else 1.20)))
        corridor = np.zeros((H, W), dtype=np.uint8)
        if x_min < x_max and deep_start < H:
            corridor[deep_start:, x_min:x_max] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, corridor)
        if int((zone_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=9.0, sigmaY=9.0)
        zone_vals = gray[zone_u8 > 0]
        dark_thresh = 150.0 if hair_length == "short" else 148.0
        if zone_vals.size >= 80:
            dark_thresh = float(
                np.clip(
                    np.percentile(zone_vals, 80.0) + (8.0 if hair_length == "short" else 14.0),
                    110.0 if hair_length == "short" else 96.0,
                    172.0 if hair_length == "short" else 164.0,
                )
            )
        contrast_thresh = 4.0 if hair_length == "short" else 6.5
        dark_u8 = (
            (gray < dark_thresh)
            & ((blur - gray) > contrast_thresh)
        ).astype(np.uint8) * 255
        dark_u8 = cv2.bitwise_and(dark_u8, zone_u8)
        if int((dark_u8 > 0).sum()) < 30:
            return np.zeros((H, W), dtype=np.float32)

        dark_u8 = cv2.morphologyEx(
            dark_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        dark_u8 = cv2.dilate(
            dark_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21) if hair_length == "short" else (7, 7)),
            iterations=1,
        )
        return (dark_u8 > 0).astype(np.float32)

    @staticmethod
    def _cv2_cleanup_dark_tail_blob(
        img_rgb: np.ndarray,
        dark_tail_u8: np.ndarray,
    ) -> np.ndarray:
        """Run a small focused cv2 inpaint pass over deep dark residual tail blobs."""
        if dark_tail_u8.shape[:2] != img_rgb.shape[:2]:
            return img_rgb
        if int((dark_tail_u8 > 0).sum()) < 40:
            return img_rgb

        mask_u8 = cv2.dilate(
            dark_tail_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 25)),
            iterations=1,
        )
        telea = cv2.inpaint(img_rgb, mask_u8, 7, cv2.INPAINT_TELEA)
        ns = cv2.inpaint(img_rgb, mask_u8, 6, cv2.INPAINT_NS)
        fill = cv2.addWeighted(telea, 0.74, ns, 0.26, 0.0)
        alpha = cv2.GaussianBlur(
            (mask_u8 > 0).astype(np.float32),
            (0, 0),
            sigmaX=3.6,
            sigmaY=3.6,
        )[..., np.newaxis]
        out = fill.astype(np.float32) * alpha + img_rgb.astype(np.float32) * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _remove_residual_hair_below_cutoff(
        self,
        img_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        cutoff_y: int,
        removal_mask: Optional[np.ndarray] = None,
        shoulder_protect: Optional[np.ndarray] = None,
        neckline_preserve: Optional[np.ndarray] = None,
        lateral_preserve: Optional[np.ndarray] = None,
        hair_length: str = "short",
        center_anchor_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        short/medium 변환 후 cutoff 아래에 남은 머리카락을 재검출해 정리.
        """
        H, W = img_rgb.shape[:2]
        cutoff_y = int(np.clip(cutoff_y, 0, H - 1))
        _, y1, _, y2 = face_bbox
        face_h = max(int(y2 - y1), 1)
        soft_zone = max(10, int(face_h * 0.22))
        soft_end = min(H - 1, cutoff_y + soft_zone)

        hair_now, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
        residual = hair_now.copy()
        residual[:cutoff_y, :] = 0.0

        residual_u8 = (residual > 0.5).astype(np.uint8) * 255
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        residual_u8 = cv2.morphologyEx(residual_u8, cv2.MORPH_OPEN, open_k)

        if soft_end > cutoff_y:
            ramp = np.ones((H,), dtype=np.float32)
            ramp[:cutoff_y] = 0.0
            ramp[cutoff_y:soft_end + 1] = np.linspace(
                0.0, 1.0, soft_end - cutoff_y + 1, dtype=np.float32
            )
            residual_soft = (residual_u8.astype(np.float32) / 255.0) * ramp[:, np.newaxis]
            residual_u8 = (residual_soft > 0.50).astype(np.uint8) * 255

        dark_tail_u8 = np.zeros((H, W), dtype=np.uint8)
        front_cleanup_u8 = np.zeros((H, W), dtype=np.uint8)
        residual_near_u8 = np.zeros((H, W), dtype=np.uint8)
        if removal_mask is not None and removal_mask.shape == (H, W):
            tail_hint = self._build_side_tail_cleanup_mask(
                removal_mask=removal_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            )
            tail_hint_u8 = (tail_hint > 0.0).astype(np.uint8) * 255
            if int((tail_hint_u8 > 0).sum()) > 0:
                deep_start = min(H, int(cutoff_y + face_h * (0.18 if hair_length == "short" else 0.24)))
                deep_zone = np.zeros((H, W), dtype=np.uint8)
                if deep_start < H:
                    deep_zone[deep_start:, :] = 255
                if int((residual_u8 > 0).sum()) > 0:
                    residual_near_u8 = cv2.dilate(
                        residual_u8,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
                        iterations=1,
                    )
                    tail_hint_u8 = cv2.bitwise_and(
                        tail_hint_u8,
                        cv2.bitwise_or(residual_near_u8, deep_zone),
                    )
                else:
                    tail_hint_u8 = cv2.bitwise_and(tail_hint_u8, deep_zone)
                residual_u8 = cv2.bitwise_or(residual_u8, tail_hint_u8)

            if hair_length == "short":
                tail_core = self._build_short_tail_core_mask(
                    removal_mask=removal_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                )
                tail_core_u8 = (tail_core > 0.0).astype(np.uint8) * 255
                if int((tail_core_u8 > 0).sum()) > 0:
                    tail_core_gate_u8 = np.zeros((H, W), dtype=np.uint8)
                    shallow_limit = min(H, int(cutoff_y + face_h * 0.24))
                    if shallow_limit > cutoff_y:
                        center_band_u8 = np.zeros((H, W), dtype=np.uint8)
                        face_w_local = max(int(face_bbox[2] - face_bbox[0]), 1)
                        center_half = max(16, int(face_w_local * 0.22))
                        center_band_u8[
                            cutoff_y:shallow_limit,
                            max(0, int(0.5 * (face_bbox[0] + face_bbox[2])) - center_half):min(W, int(0.5 * (face_bbox[0] + face_bbox[2])) + center_half),
                        ] = 255
                        shallow_gate_u8 = cv2.bitwise_and(residual_near_u8, center_band_u8)
                        tail_core_gate_u8[cutoff_y:shallow_limit, :] = shallow_gate_u8[cutoff_y:shallow_limit, :]
                    deep_core_start = min(H, int(cutoff_y + face_h * 0.40))
                    if deep_core_start < H:
                        tail_core_gate_u8[deep_core_start:, :] = 255
                    tail_core_u8 = cv2.bitwise_and(tail_core_u8, tail_core_gate_u8)
                if int((tail_core_u8 > 0).sum()) > 0:
                    residual_u8 = cv2.bitwise_or(residual_u8, tail_core_u8)

                front_cleanup = self._build_front_strand_cleanup_mask(
                    removal_mask=removal_mask,
                    face_bbox=face_bbox,
                    cutoff_y=cutoff_y,
                    hair_length=hair_length,
                    anchor_mask=center_anchor_mask,
                )
                front_cleanup_u8 = (front_cleanup > 0.0).astype(np.uint8) * 255
                if int((front_cleanup_u8 > 0).sum()) > 0:
                    residual_u8 = cv2.bitwise_or(residual_u8, front_cleanup_u8)

            dark_tail = self._build_dark_tail_residual_mask(
                img_rgb=img_rgb,
                removal_mask=removal_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            )
            dark_tail_u8 = (dark_tail > 0.0).astype(np.uint8) * 255
            if int((dark_tail_u8 > 0).sum()) > 0:
                residual_u8 = cv2.bitwise_or(residual_u8, dark_tail_u8)

        if int((residual_u8 > 0).sum()) < 60:
            return img_rgb

        dilate_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 13) if hair_length == "short" else (9, 9),
        )
        residual_u8 = cv2.dilate(residual_u8, dilate_k, iterations=1)

        if shoulder_protect is not None and shoulder_protect.shape == (H, W):
            protect_threshold = 0.62 if hair_length == "short" else 0.34
            protect_u8 = (shoulder_protect > protect_threshold).astype(np.uint8) * 255
            if hair_length == "short" and int((protect_u8 > 0).sum()) > 0:
                deep_release_y = min(H, int(cutoff_y + face_h * 0.28))
                if deep_release_y < H:
                    protect_u8[deep_release_y:, :] = 0
                protect_u8 = cv2.erode(
                    protect_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    iterations=1,
                )
            if int((protect_u8 > 0).sum()) > 0:
                residual_u8 = cv2.bitwise_and(residual_u8, cv2.bitwise_not(protect_u8))

        if neckline_preserve is not None and neckline_preserve.shape == (H, W):
            preserve_u8 = (neckline_preserve > (0.34 if hair_length == "short" else 0.26)).astype(np.uint8) * 255
            if int((preserve_u8 > 0).sum()) > 0:
                if hair_length == "short":
                    deep_release_y = min(H, int(cutoff_y + face_h * 0.22))
                    if deep_release_y < H:
                        preserve_u8[deep_release_y:, :] = 0
                preserve_u8 = cv2.dilate(
                    preserve_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                    iterations=1,
                )
                residual_u8 = cv2.bitwise_and(residual_u8, cv2.bitwise_not(preserve_u8))

        if lateral_preserve is not None and lateral_preserve.shape == (H, W):
            lateral_u8 = (lateral_preserve > 0.18).astype(np.uint8) * 255
            if int((lateral_u8 > 0).sum()) > 0:
                lateral_u8 = cv2.dilate(
                    lateral_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 9)),
                    iterations=1,
                )
                residual_u8 = cv2.bitwise_and(residual_u8, cv2.bitwise_not(lateral_u8))

        if int((residual_u8 > 0).sum()) < 60:
            return img_rgb

        cleaned = self._lama_inpaint(img_rgb, residual_u8)
        if int((dark_tail_u8 > 0).sum()) > 0:
            cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, dark_tail_u8)
        if int((front_cleanup_u8 > 0).sum()) > 0:
            cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, front_cleanup_u8)
        return cleaned

    def _final_cutoff_cleanup(
        self,
        img_rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        removal_mask: np.ndarray,
        cutoff_y: int,
        shoulder_protect: Optional[np.ndarray] = None,
        neckline_preserve: Optional[np.ndarray] = None,
        lateral_preserve: Optional[np.ndarray] = None,
        hair_length: str = "short",
        center_anchor_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        최종 결과에서 cutoff 아래 long-hair 제거 마스크 영역을 한 번 더 정리.
        """
        H, W = img_rgb.shape[:2]
        if removal_mask.shape != (H, W):
            return img_rgb

        x1, y1, x2, y2 = face_bbox
        face_h = max(int(y2 - y1), 1)
        face_w = max(int(x2 - x1), 1)
        cx = int(0.5 * (x1 + x2))
        soft_zone = max(12, int(face_h * 0.25))

        force = removal_mask.copy().astype(np.float32)
        cutoff_y = int(np.clip(cutoff_y, 0, H - 1))
        force[:cutoff_y, :] = 0.0
        soft_end = min(H - 1, cutoff_y + soft_zone)
        if soft_end > cutoff_y:
            ramp = np.ones((H,), dtype=np.float32)
            ramp[:cutoff_y] = 0.0
            ramp[cutoff_y:soft_end + 1] = np.linspace(
                0.0, 1.0, soft_end - cutoff_y + 1, dtype=np.float32
            )
            force = force * ramp[:, np.newaxis]

        # 얼굴 주변 corridor 안에서만 cleanup을 허용해 의상/배경 훼손을 줄인다.
        corridor_ratio = 1.55 if hair_length == "short" else 1.35
        x_min = max(0, int(x1 - face_w * corridor_ratio))
        x_max = min(W, int(x2 + face_w * corridor_ratio))
        corridor = np.zeros((H, W), dtype=np.uint8)
        if x_min < x_max:
            corridor[:, x_min:x_max] = 255

        force_thresh = 0.56 if hair_length == "short" else 0.54
        force_u8 = ((force > force_thresh).astype(np.uint8) * 255)
        force_u8 = cv2.bitwise_and(force_u8, corridor)
        dark_tail_u8 = np.zeros((H, W), dtype=np.uint8)
        front_cleanup_u8 = np.zeros((H, W), dtype=np.uint8)

        # 실제 남아있는 hair 픽셀과 교집합을 우선 적용해 의상/배경 훼손 방지
        hair_now, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
        hair_now[:cutoff_y, :] = 0.0
        hair_now_u8 = (hair_now > 0.5).astype(np.uint8) * 255
        if int((hair_now_u8 > 0).sum()) > 0:
            hair_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (11, 11) if hair_length == "short" else (9, 9),
            )
            hair_now_u8 = cv2.dilate(hair_now_u8, hair_k, iterations=1)

        hair_inter_u8 = cv2.bitwise_and(force_u8, hair_now_u8)
        if hair_length == "short":
            # SegFace miss를 보완하기 위해 side-zone에 한해 high-confidence force를 추가 반영
            center_half = max(18, int(face_w * 0.42))
            side_zone = corridor.copy()
            side_zone[:, max(0, cx - center_half):min(W, cx + center_half)] = 0
            fallback_u8 = ((force > 0.78).astype(np.uint8) * 255)
            fallback_u8 = cv2.bitwise_and(fallback_u8, side_zone)
            force_u8 = cv2.bitwise_or(hair_inter_u8, fallback_u8)
            hair_near_u8 = np.zeros((H, W), dtype=np.uint8)
            if int((hair_now_u8 > 0).sum()) > 0:
                hair_near_u8 = cv2.dilate(
                    hair_now_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
                    iterations=1,
                )

            tail_hint = self._build_side_tail_cleanup_mask(
                removal_mask=removal_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            )
            tail_hint_u8 = (tail_hint > 0.0).astype(np.uint8) * 255
            if int((tail_hint_u8 > 0).sum()) > 0:
                deep_start = min(H, int(cutoff_y + face_h * 0.18))
                deep_zone = np.zeros((H, W), dtype=np.uint8)
                if deep_start < H:
                    deep_zone[deep_start:, :] = 255
                forced_tail_u8 = cv2.bitwise_and(
                    tail_hint_u8,
                    cv2.bitwise_or(hair_near_u8, deep_zone),
                )
                force_u8 = cv2.bitwise_or(force_u8, forced_tail_u8)

            tail_core = self._build_short_tail_core_mask(
                removal_mask=removal_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            )
            tail_core_u8 = (tail_core > 0.0).astype(np.uint8) * 255
            if int((tail_core_u8 > 0).sum()) > 0:
                tail_core_gate_u8 = np.zeros((H, W), dtype=np.uint8)
                shallow_limit = min(H, int(cutoff_y + face_h * 0.24))
                if shallow_limit > cutoff_y:
                    center_band_u8 = np.zeros((H, W), dtype=np.uint8)
                    center_half = max(16, int(face_w * 0.22))
                    center_band_u8[
                        cutoff_y:shallow_limit,
                        max(0, cx - center_half):min(W, cx + center_half),
                    ] = 255
                    shallow_gate_u8 = cv2.bitwise_and(hair_near_u8, center_band_u8)
                    tail_core_gate_u8[cutoff_y:shallow_limit, :] = shallow_gate_u8[cutoff_y:shallow_limit, :]
                deep_core_start = min(H, int(cutoff_y + face_h * 0.40))
                if deep_core_start < H:
                    tail_core_gate_u8[deep_core_start:, :] = 255
                tail_core_u8 = cv2.bitwise_and(tail_core_u8, tail_core_gate_u8)
            if int((tail_core_u8 > 0).sum()) > 0:
                force_u8 = cv2.bitwise_or(force_u8, tail_core_u8)

            front_cleanup = self._build_front_strand_cleanup_mask(
                removal_mask=removal_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
                anchor_mask=center_anchor_mask,
            )
            front_cleanup_u8 = (front_cleanup > 0.0).astype(np.uint8) * 255
            if int((front_cleanup_u8 > 0).sum()) > 0:
                force_u8 = cv2.bitwise_or(force_u8, front_cleanup_u8)
        else:
            # medium도 SegFace miss 보완용 fallback force 일부 허용
            fallback_u8 = ((force > 0.74).astype(np.uint8) * 255)
            fallback_u8 = cv2.bitwise_and(fallback_u8, corridor)
            force_u8 = cv2.bitwise_or(hair_inter_u8, fallback_u8)

        dark_tail = self._build_dark_tail_residual_mask(
            img_rgb=img_rgb,
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        dark_tail_u8 = (dark_tail > 0.0).astype(np.uint8) * 255
        if int((dark_tail_u8 > 0).sum()) > 0:
            force_u8 = cv2.bitwise_or(force_u8, cv2.bitwise_and(dark_tail_u8, corridor))

        if shoulder_protect is not None and shoulder_protect.shape == (H, W):
            protect_threshold = 0.62 if hair_length == "short" else 0.34
            protect_u8 = (shoulder_protect > protect_threshold).astype(np.uint8) * 255
            if hair_length == "short" and int((protect_u8 > 0).sum()) > 0:
                deep_release_y = min(H, int(cutoff_y + face_h * 0.28))
                if deep_release_y < H:
                    protect_u8[deep_release_y:, :] = 0
                protect_u8 = cv2.erode(
                    protect_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    iterations=1,
                )
            if int((protect_u8 > 0).sum()) > 0:
                force_u8 = cv2.bitwise_and(force_u8, cv2.bitwise_not(protect_u8))

        if neckline_preserve is not None and neckline_preserve.shape == (H, W):
            preserve_u8 = (neckline_preserve > (0.34 if hair_length == "short" else 0.26)).astype(np.uint8) * 255
            if int((preserve_u8 > 0).sum()) > 0:
                if hair_length == "short":
                    deep_release_y = min(H, int(cutoff_y + face_h * 0.22))
                    if deep_release_y < H:
                        preserve_u8[deep_release_y:, :] = 0
                preserve_u8 = cv2.dilate(
                    preserve_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                    iterations=1,
                )
                force_u8 = cv2.bitwise_and(force_u8, cv2.bitwise_not(preserve_u8))

        if lateral_preserve is not None and lateral_preserve.shape == (H, W):
            lateral_u8 = (lateral_preserve > 0.18).astype(np.uint8) * 255
            if int((lateral_u8 > 0).sum()) > 0:
                lateral_u8 = cv2.dilate(
                    lateral_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 9)),
                    iterations=1,
                )
                force_u8 = cv2.bitwise_and(force_u8, cv2.bitwise_not(lateral_u8))

        min_cleanup_px = 28 if hair_length == "short" else 40
        if int((force_u8 > 0).sum()) < min_cleanup_px:
            return img_rgb

        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 13) if hair_length == "short" else (7, 7),
        )
        force_u8 = cv2.dilate(force_u8, k, iterations=1)

        # LaMa로 잔존 hair 제거
        cleaned = self._lama_inpaint(img_rgb, force_u8)
        if int((dark_tail_u8 > 0).sum()) > 0:
            cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, dark_tail_u8)
        if int((front_cleanup_u8 > 0).sum()) > 0:
            cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, front_cleanup_u8)
        return cleaned

    # ──────────────────────────────────────────────────────────────────────────
    # Compositing
    # ──────────────────────────────────────────────────────────────────────────

    def _composite(
        self,
        orig_bgr: np.ndarray,
        orig_rgb: np.ndarray,
        gen_pil: Image.Image,            # 512×512 RGB
        hair_mask: np.ndarray,           # H×W float32 (original resolution)
        scale: float,
        pad: Tuple[int, int],            # (pad_left, pad_top)
        original_size: Tuple[int, int],  # (W, H)
        protect_mask: Optional[np.ndarray] = None,  # H×W float32: 이 영역은 alpha=0 강제 (얼굴 보호)
        protect_release_mask: Optional[np.ndarray] = None,
        hair_length: str = "long",
    ) -> np.ndarray:
        """
        SD 생성 이미지를 원본에 합성.
        - hair mask 영역: SD 생성 결과
        - 그 외 (+ protect_mask): 원본 (얼굴/배경 유지)
        """
        W, H = original_size
        hair_mask = self._resize_mask_to_shape(hair_mask, (H, W))
        protect_mask = self._resize_mask_to_shape(protect_mask, (H, W))
        protect_release_mask = self._resize_mask_to_shape(protect_release_mask, (H, W))
        pad_l, pad_t = pad
        new_w = int(W * scale)
        new_h = int(H * scale)

        # letterbox 제거 → 원본 비율로 crop
        gen_np = np.array(gen_pil)   # 512×512×3 RGB
        gen_cropped = gen_np[pad_t:pad_t + new_h, pad_l:pad_l + new_w]

        # 원본 해상도로 upscale
        gen_orig = cv2.resize(gen_cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

        # alpha 블렌딩: short/medium는 경계를 더 또렷하게 유지
        sigma = 6.0
        if hair_length == "short":
            sigma = 4.2
        elif hair_length == "medium":
            sigma = 4.8
        alpha = cv2.GaussianBlur(hair_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if hair_length == "short":
            alpha = np.clip((alpha - 0.10) / 0.90, 0.0, 1.0)
        elif hair_length == "medium":
            alpha = np.clip((alpha - 0.07) / 0.93, 0.0, 1.0)
        alpha = np.clip(alpha, 0.0, 1.0)

        # 얼굴/귀/눈 등 보호 영역: alpha를 0으로 강제
        # → Gaussian blur가 얼굴 경계로 번지더라도 원본 픽셀 100% 유지
        if protect_mask is not None:
            # protect_mask도 살짝 dilate해서 경계까지 확실히 보호
            protect_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            protect_dilated = cv2.dilate(protect_mask.astype(np.float32), protect_k)
            if protect_release_mask is not None:
                protect_dilated = np.clip(
                    protect_dilated - np.clip(protect_release_mask.astype(np.float32) * 1.35, 0.0, 1.0),
                    0.0,
                    1.0,
                )
            alpha = alpha * (1.0 - np.clip(protect_dilated, 0.0, 1.0))

        alpha = alpha[..., np.newaxis]   # H×W×1

        orig_f = orig_rgb.astype(np.float32)
        gen_f  = gen_orig.astype(np.float32)
        blend  = gen_f * alpha + orig_f * (1.0 - alpha)
        blend  = np.clip(blend, 0, 255).astype(np.uint8)

        return cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def unload(self) -> None:
        """VRAM 해제"""
        import gc
        self._sd_pipe = None
        self._sam2_factory = None
        if self._mp_face:
            self._mp_face.close()
        if self._mp_face_mesh:
            self._mp_face_mesh.close()
        self._mp_face = None
        self._mp_face_mesh = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[SDPipeline] 모델 언로드 완료")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 테스트
# ─────────────────────────────────────────────────────────────────────────────
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
