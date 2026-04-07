"""
MirrAI SD Inpainting — 설정 & 타입 정의
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SDInpaintConfig, SDInpaintResult dataclass 및 공유 상수 정의.
모든 파이프라인 컴포넌트는 이 모듈에서 상수/타입을 import한다.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ── 프로젝트 루트 (pipeline_sd_components/ 의 부모) ───────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    "단발", "숏컷", "픽시",
])
_MEDIUM_HAIR_KEYWORDS = frozenset([
    "lob", "midi", "medium", "shoulder length", "shoulder-length",
    "collarbone", "clavicle", "mid length", "mid-length",
    "wolf cut", "soft mullet", "mullet", "baby mullet", "mini mullet",
    "two block", "two-block", "comma hair", "comma", "dandy cut", "dandy",
    "regent cut", "regent", "side part", "side-part", "swept-back", "swept back",
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
    enable_upper_clothes_overwrite: bool = True
    upper_clothes_expand_px: int = 22
    hair_overlap_expand_px: int = 18
    controlnet_use_masked_edges: bool = True
    upper_clothes_overwrite_min_px: int = 180
    overwrite_cloth_overlap_ratio_limit: float = 0.68
    overwrite_cloth_guard_residual: float = 0.12
    overwrite_cloth_guard_expand_px: int = 24
    upper_clothes_overwrite_alpha: float = 0.92
    short_upper_clothes_overwrite_alpha: float = 0.72
    medium_upper_clothes_overwrite_alpha: float = 0.82
    long_upper_clothes_overwrite_alpha: float = 0.92
    short_side_column_inner_keepout_ratio: float = 0.42
    short_side_column_neckline_keepout_ratio: float = 0.34
    short_side_column_outer_strip_gap_ratio: float = 0.46
    short_side_column_restore_plain_fill: bool = False
    short_side_column_outer_strip_width_scale: float = 0.96
    short_side_column_allow_plain_cloth_force: bool = False
    final_hair_lane_center_keepout_ratio: float = 0.20
    final_hair_lane_neckline_keepout_ratio: float = 0.30
    final_hair_lane_outer_strip_gap_ratio: float = 0.30
    overwrite_core_fallback_half_ratio: float = 0.46
    overwrite_core_fallback_top_ratio: float = 0.22
    overwrite_core_fallback_bottom_ratio: float = 0.98

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
