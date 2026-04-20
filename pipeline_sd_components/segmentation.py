"""
MirrAI SD Inpainting — 마스크 세그멘테이션
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MediaPipe 얼굴 검출, SegFace 헤어 세그멘테이션, SAM2 마스크 정제, Canny edge 추출.
pipeline_sd_inpainting.py에서 분리된 세그멘테이션 함수 모음.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .config import (
    CLOTH_CLASS_IDX,
    EARRING_CLASS_IDX,
    FACE_CLASS_IDXS,
    GLASS_CLASS_IDX,
    HAIR_CLASS_IDX,
    NECKLACE_CLASS_IDX,
    SD_SIZE,
)

logger = logging.getLogger(__name__)


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
        "raw_cloth_mask": (cloth_orig > 0.5).astype(np.float32),
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


def _prepare_sd_inputs(
    self,
    img_rgb: np.ndarray,     # H×W×3 RGB
    hair_mask: np.ndarray,   # H×W float32
    mask_edge_suppression: float = 1.0,  # 0.0=엣지 보존, 1.0=마스크 내부 엣지 완전 제거
    canny_suppress_mask: Optional[np.ndarray] = None,  # H×W float32 — 이 영역의 canny edge도 제거
    debug_outputs: Optional[Dict[str, np.ndarray]] = None,
    target_size: Optional[int] = None,
) -> Tuple[Image.Image, Image.Image, Image.Image, float, Tuple[int, int]]:
    """
    Letter-box resize → square diffusion canvas.

    Args:
        canny_suppress_mask: short/medium에서 사용. 기존 long-hair 영역의 canny edge를
                             추가로 제거하여 ControlNet이 원본 긴머리 윤곽을 따라가지 않게 함.

    Returns:
        img_512:    PIL RGB square canvas (full image)
        mask_512:   PIL L  square canvas (흰색=inpaint)
        canny_512:  PIL RGB square canvas (ControlNet conditioning)
        scale:      resize 비율
        pad:        (pad_left, pad_top) pixels
    """
    H, W = img_rgb.shape[:2]
    canvas_size = int(target_size or SD_SIZE)
    canvas_size = max(256, int(round(canvas_size / 8)) * 8)
    scale = canvas_size / max(H, W)
    new_w, new_h = int(W * scale), int(H * scale)
    pad_l = (canvas_size - new_w) // 2
    pad_t = (canvas_size - new_h) // 2

    # ── image letterbox
    img_rs = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[pad_t:pad_t + new_h, pad_l:pad_l + new_w] = img_rs

    # ── mask letterbox
    msk_rs = cv2.resize(hair_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    msk_canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    msk_canvas[pad_t:pad_t + new_h, pad_l:pad_l + new_w] = msk_rs

    # ── Canny edge
    # 기본(헤어 생성): 마스크 내부 엣지 강하게 제거
    # 배경 복원(fill): 일부 엣지를 남겨 texture/구조 연속성 확보
    source_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    source_canny_raw = cv2.Canny(source_gray, self.config.canny_low, self.config.canny_high)
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)
    canny_raw_512 = canny.copy()
    suppress = float(np.clip(mask_edge_suppression, 0.0, 1.0))
    hair_hard = (msk_canvas > 0.5).astype(np.float32)

    # canny_suppress_mask가 있으면 해당 영역의 edge도 완전 제거
    # → LaMa 잔여 블러 윤곽이 ControlNet에 전달되지 않음
    if canny_suppress_mask is not None:
        sup_rs = cv2.resize(canny_suppress_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sup_canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
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
    source_hair_hard = cv2.resize(
        hair_hard[pad_t:pad_t + new_h, pad_l:pad_l + new_w],
        (W, H),
        interpolation=cv2.INTER_AREA,
    )
    source_hair_hard = np.clip(source_hair_hard.astype(np.float32), 0.0, 1.0)
    source_canny_suppressed = source_canny_raw.astype(np.float32) * (1.0 - source_hair_hard * suppress)
    canny_rgb = cv2.cvtColor(canny_f.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    if debug_outputs is not None:
        debug_outputs["source_canny_raw"] = source_canny_raw.astype(np.uint8)
        debug_outputs["source_canny_suppressed"] = source_canny_suppressed.astype(np.uint8)
        debug_outputs["source_canny_suppress_mask"] = (np.clip(source_hair_hard, 0.0, 1.0) * 255).astype(np.uint8)
        debug_outputs["control_canny_raw_512"] = canny_raw_512.astype(np.uint8)
        debug_outputs["control_canny_suppressed_512"] = canny_f.astype(np.uint8)
        debug_outputs["control_canny_suppress_mask_512"] = (
            np.clip(hair_hard, 0.0, 1.0) * 255
        ).astype(np.uint8)

    img_512   = Image.fromarray(canvas)
    mask_512  = Image.fromarray((msk_canvas * 255).astype(np.uint8), mode="L")
    canny_512 = Image.fromarray(canny_rgb)

    return img_512, mask_512, canny_512, scale, (pad_l, pad_t)


def bind_segmentation_methods_to_pipeline(cls) -> None:
    """세그멘테이션 메서드를 MirrAISDPipeline에 바인딩."""
    cls._detect_face = _detect_face
    cls._detect_face_mesh = _detect_face_mesh
    cls._build_landmark_hull_mask = staticmethod(_build_landmark_hull_mask)
    cls._build_mediapipe_face_oval_mask = _build_mediapipe_face_oval_mask
    cls._detect_landmark_data = _detect_landmark_data
    cls._build_face_mesh_analysis = staticmethod(_build_face_mesh_analysis)
    cls._render_face_mesh_debug_images = _render_face_mesh_debug_images
    cls._segface_hair_mask = _segface_hair_mask
    cls._build_face_protect_mask = _build_face_protect_mask
    cls._sanitize_face_region_mask = _sanitize_face_region_mask
    cls._build_accessory_protect_mask = _build_accessory_protect_mask
    cls._refine_with_sam2 = _refine_with_sam2
    cls._prepare_sd_inputs = _prepare_sd_inputs
