#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_sd_components.config import SDInpaintConfig
from pipeline_sd_components.output import build_length_aware_output_crop_box, crop_with_padding


def main() -> int:
    cfg = SDInpaintConfig()
    image_shape = (250, 199)
    face_bbox = (53, 64, 186, 198)

    short_box = build_length_aware_output_crop_box(cfg, image_shape, face_bbox, "short")
    medium_box = build_length_aware_output_crop_box(cfg, image_shape, face_bbox, "medium")
    long_box = build_length_aware_output_crop_box(cfg, image_shape, face_bbox, "long")

    assert short_box is not None, "short crop should stay active for tight portraits"
    assert medium_box is not None, "medium crop should stay active for tight portraits"
    assert long_box is not None, "long crop should stay active for tight portraits"
    assert short_box[1] < 0 or short_box[3] > image_shape[0], short_box
    assert (medium_box[3] - medium_box[1]) > (short_box[3] - short_box[1]), (short_box, medium_box)
    assert (long_box[3] - long_box[1]) > (medium_box[3] - medium_box[1]), (medium_box, long_box)

    height, width = image_shape
    image = np.dstack(
        [
            np.tile(np.arange(width, dtype=np.uint8), (height, 1)),
            np.tile(np.arange(height, dtype=np.uint8).reshape(height, 1), (1, width)),
            np.full((height, width), 127, dtype=np.uint8),
        ]
    )
    short_crop = crop_with_padding(image, short_box)
    assert short_crop.shape[:2] == (short_box[3] - short_box[1], short_box[2] - short_box[0])

    mask = np.full(image_shape, 255, dtype=np.uint8)
    mask_crop = crop_with_padding(
        mask,
        short_box,
        border_mode=cv2.BORDER_CONSTANT,
        constant_value=0,
        blur_padding=False,
    )
    assert mask_crop.shape == short_crop.shape[:2]
    assert int(mask_crop[0, 0]) == 0, "out-of-bounds mask padding should be zero"

    print(
        {
            "short_box": short_box,
            "medium_box": medium_box,
            "long_box": long_box,
            "short_crop_shape": list(short_crop.shape),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
