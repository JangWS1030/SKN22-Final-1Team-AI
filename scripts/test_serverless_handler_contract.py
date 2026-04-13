#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import handler_sd


def _make_image_base64() -> str:
    image = np.full((64, 64, 3), 220, dtype=np.uint8)
    image[:, :, 1] = 180
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class _FakeAnalysisPipeline:
    def _detect_face(self, _img_rgb):
        return (10, 12, 52, 54)

    def _detect_landmark_data(self, _img_rgb, _face_bbox):
        visualization = np.full((64, 64, 3), 255, dtype=np.uint8)
        return {
            "detected": True,
            "debug_data": {
                "ratios": {
                    "cheekbone_to_height": 0.72,
                    "jaw_to_height": 0.60,
                    "temple_to_height": 0.70,
                    "jaw_to_cheekbone": 0.83,
                }
            },
            "debug_images": {
                "mediapipe_face_mesh_contours": visualization,
            },
        }


def main() -> int:
    with patch("handler_sd._get_analysis_pipeline", return_value=_FakeAnalysisPipeline()):
        output = handler_sd.handler(
            {
                "input": {
                    "action": "analyze_face",
                    "image": _make_image_base64(),
                    "include_visualization": True,
                }
            }
        )

    assert output["status"] == "ok", output
    assert output["face_shape"] == "oval", output
    assert "visualization_base64" in output, output

    print(
        json.dumps(
            {
                "status": output["status"],
                "face_shape": output["face_shape"],
                "has_visualization": bool(output.get("visualization_base64")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
