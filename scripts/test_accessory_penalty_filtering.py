#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_sd_components.config import SDInpaintConfig
from pipeline_sd_components import prompt as prompt_module


def _circle_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    cv2.circle(mask, center, radius, 1.0, -1)
    return mask


def _make_base_image(shape: tuple[int, int, int], face_bbox: tuple[int, int, int, int]) -> np.ndarray:
    image = np.full(shape, 242, dtype=np.uint8)
    x1, y1, x2, y2 = face_bbox
    image[y1:y2, x1:x2] = np.array([212, 182, 164], dtype=np.uint8)
    return image


def _make_face_mask(shape: tuple[int, int], face_bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    x1, y1, x2, y2 = face_bbox
    mask[y1:y2, x1:x2] = 1.0
    return mask


def _make_hair_mask(shape: tuple[int, int], face_bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    x1, y1, x2, _ = face_bbox
    top = max(0, y1 - 24)
    bottom = min(shape[0], y1 + 12)
    left = max(0, x1 + 10)
    right = min(shape[1], x2 - 10)
    mask[top:bottom, left:right] = 1.0
    return mask


def _make_glasses_mask(shape: tuple[int, int], face_bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, _ = face_bbox
    left_eye = (x1 + 22, y1 + 34)
    right_eye = (x2 - 22, y1 + 34)
    mask = np.zeros(shape, dtype=np.float32)
    mask = np.maximum(mask, _circle_mask(shape, left_eye, 12))
    mask = np.maximum(mask, _circle_mask(shape, right_eye, 12))
    cv2.rectangle(mask, (left_eye[0] + 8, left_eye[1] - 2), (right_eye[0] - 8, right_eye[1] + 2), 1.0, -1)
    return mask


def _paint_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    result[mask > 0.5] = np.array(color, dtype=np.uint8)
    return result


def _make_hat_image(shape: tuple[int, int, int], face_bbox: tuple[int, int, int, int]) -> np.ndarray:
    image = _make_base_image(shape, face_bbox)
    x1, y1, x2, _ = face_bbox
    cv2.ellipse(
        image,
        center=((x1 + x2) // 2, y1 - 4),
        axes=(60, 42),
        angle=0,
        startAngle=180,
        endAngle=360,
        color=(20, 20, 20),
        thickness=-1,
    )
    cv2.rectangle(
        image,
        (x1 - 10, y1 - 12),
        (x2 + 10, y1 + 6),
        color=(24, 24, 24),
        thickness=-1,
    )
    return image


class _StubPipeline:
    def __init__(self) -> None:
        self.config = SDInpaintConfig()
        self._last_segface_mask_debug: dict[str, np.ndarray] | None = None
        self._scenario: dict[str, np.ndarray] | None = None

    def set_scenario(self, scenario: dict[str, np.ndarray]) -> None:
        self._scenario = scenario

    def _segface_hair_mask(self, _img_rgb, _face_bbox):
        if self._scenario is None:
            raise RuntimeError("scenario not configured")
        hair_mask = self._scenario["hair_mask"].astype(np.float32)
        face_mask = self._scenario["face_mask"].astype(np.float32)
        cloth_mask = np.zeros_like(hair_mask, dtype=np.float32)
        self._last_segface_mask_debug = {
            "glasses_mask": self._scenario["glasses_mask"].astype(np.float32),
            "earring_mask": self._scenario["earring_mask"].astype(np.float32),
            "necklace_mask": self._scenario["necklace_mask"].astype(np.float32),
        }
        return hair_mask, face_mask, cloth_mask


def main() -> int:
    face_bbox = (50, 60, 130, 160)
    H, W = 200, 180
    shape = (H, W)
    rgb_shape = (H, W, 3)

    face_mask = _make_face_mask(shape, face_bbox)
    hair_mask = _make_hair_mask(shape, face_bbox)
    no_glasses = np.zeros(shape, dtype=np.float32)
    jewelry = np.zeros(shape, dtype=np.float32)
    glasses_mask = _make_glasses_mask(shape, face_bbox)

    plain_source = _paint_mask(_make_base_image(rgb_shape, face_bbox), hair_mask, (28, 28, 28))
    glasses_source = _paint_mask(plain_source, glasses_mask, (36, 36, 36))

    hat_candidate = _paint_mask(_make_hat_image(rgb_shape, face_bbox), hair_mask, (28, 28, 28))
    hat_candidate = _paint_mask(hat_candidate, glasses_mask, (36, 36, 36))
    glasses_candidate = _paint_mask(_make_base_image(rgb_shape, face_bbox), hair_mask, (28, 28, 28))
    glasses_candidate = _paint_mask(glasses_candidate, glasses_mask, (36, 36, 36))

    pipeline = _StubPipeline()

    source_profile_plain = prompt_module._build_accessory_profile(
        pipeline,
        plain_source,
        face_bbox,
        hair_mask=hair_mask,
        face_mask=face_mask,
        glasses_mask=no_glasses,
        earring_mask=jewelry,
        necklace_mask=jewelry,
    )
    pipeline.set_scenario(
        {
            "hair_mask": hair_mask,
            "face_mask": face_mask,
            "glasses_mask": glasses_mask,
            "earring_mask": jewelry,
            "necklace_mask": jewelry,
        }
    )
    hat_details = prompt_module._estimate_accessory_penalty_details(
        pipeline,
        hat_candidate,
        face_bbox,
        source_accessory_profile=source_profile_plain,
    )
    assert hat_details is not None
    assert hat_details["headwear_penalty"] >= 0.34, hat_details
    assert hat_details["glasses_penalty"] >= 0.26, hat_details
    assert hat_details["exclude"] is True, hat_details

    source_profile_glasses = prompt_module._build_accessory_profile(
        pipeline,
        glasses_source,
        face_bbox,
        hair_mask=hair_mask,
        face_mask=face_mask,
        glasses_mask=glasses_mask,
        earring_mask=jewelry,
        necklace_mask=jewelry,
    )
    pipeline.set_scenario(
        {
            "hair_mask": hair_mask,
            "face_mask": face_mask,
            "glasses_mask": glasses_mask,
            "earring_mask": jewelry,
            "necklace_mask": jewelry,
        }
    )
    glasses_details = prompt_module._estimate_accessory_penalty_details(
        pipeline,
        glasses_candidate,
        face_bbox,
        source_accessory_profile=source_profile_glasses,
    )
    assert glasses_details is not None
    assert glasses_details["headwear_penalty"] < 0.10, glasses_details
    assert glasses_details["glasses_penalty"] < 0.10, glasses_details
    assert glasses_details["exclude"] is False, glasses_details

    print(
        json.dumps(
            {
                "hat_candidate": {
                    "exclude": bool(hat_details["exclude"]),
                    "headwear_penalty": round(float(hat_details["headwear_penalty"]), 4),
                    "glasses_penalty": round(float(hat_details["glasses_penalty"]), 4),
                    "total_penalty": round(float(hat_details["total_penalty"]), 4),
                },
                "glasses_candidate": {
                    "exclude": bool(glasses_details["exclude"]),
                    "headwear_penalty": round(float(glasses_details["headwear_penalty"]), 4),
                    "glasses_penalty": round(float(glasses_details["glasses_penalty"]), 4),
                    "total_penalty": round(float(glasses_details["total_penalty"]), 4),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
