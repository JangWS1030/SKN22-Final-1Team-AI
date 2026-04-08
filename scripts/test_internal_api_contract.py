#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from internal_api_app import API_VERSION, SCHEMA_VERSION, app
from style_recommender import _load_hairstyles


def _make_image_base64() -> str:
    image = np.full((64, 64, 3), 220, dtype=np.uint8)
    image[:, :, 1] = 180
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main() -> int:
    client = TestClient(app)
    sample_image_b64 = _make_image_base64()

    health = client.get("/internal/health", headers={"X-MirrAI-API-Version": API_VERSION})
    assert health.status_code == 200, health.text
    health_data = health.json()
    assert health_data["schema_version"] == SCHEMA_VERSION
    assert health_data["data"]["role"] == "model-ai-analysis-service"

    styles = _load_hairstyles()
    explain = client.post(
        "/internal/explain-style",
        headers={"X-MirrAI-API-Version": API_VERSION},
        json={"style_id": styles[0]["id"]},
    )
    assert explain.status_code == 200, explain.text

    analyze_result = None
    with patch(
        "internal_api_app._analyze_face_core",
        return_value={
            "face_shape": "oval",
            "face_shape_scores": {"oval": 0.91, "round": 0.09},
            "golden_ratio_score": 0.7342,
            "face_ratios": {
                "cheekbone_to_height": 0.72,
                "jaw_to_height": 0.60,
                "temple_to_height": 0.70,
                "jaw_to_cheekbone": 0.83,
            },
            "face_bbox": {"x1": 10, "y1": 12, "x2": 52, "y2": 54},
            "image_url": None,
            "image_url_expires_at": None,
            "schema_version": SCHEMA_VERSION,
        },
    ):
        analyze = client.post(
            "/internal/analyze-face",
            headers={"X-MirrAI-API-Version": API_VERSION},
            json={
                "image_base64": sample_image_b64,
                "include_visualization": False,
            },
        )
        assert analyze.status_code == 200, analyze.text
        analyze_result = analyze.json()

    fake_result = SimpleNamespace(
        rank=0,
        image=np.full((64, 64, 3), 127, dtype=np.uint8),
    )
    fake_recommendations = [
        {
            "rank": 0,
            "style_id": "fake-style-1",
            "style_name": "Fake Style",
            "score": 0.9123,
            "face_shapes": ["oval", "heart"],
            "trend_name": "Fake Trend",
            "description": "Fake description",
        }
    ]

    with (
        patch("internal_api_app._run_recommendation", return_value=(fake_recommendations, None, "fake bob", "natural black")),
        patch("internal_api_app._get_pipeline", return_value=object()),
        patch("internal_api_app._generate_per_recommendation", return_value=[fake_result]),
    ):
        generate = client.post(
            "/internal/generate-simulations",
            headers={"X-MirrAI-API-Version": API_VERSION},
            json={
                "image_base64": sample_image_b64,
                "analysis_data": {
                    "face_shape": "oval",
                    "golden_ratio_score": 0.71,
                    "face_ratios": {
                        "cheekbone_to_height": 0.72,
                        "jaw_to_height": 0.60,
                        "temple_to_height": 0.70,
                        "jaw_to_cheekbone": 0.83,
                    },
                },
                "survey_data": {"length": "short"},
                "top_k": 1,
            },
        )
    assert generate.status_code == 200, generate.text
    generate_data = generate.json()
    assert generate_data["data"]["items"][0]["style_id"] == "fake-style-1"

    print(
        json.dumps(
            {
                "health_status": health_data["status"],
                "explain_status": explain.json()["status"],
                "analyze_face_ran": analyze_result is not None,
                "generate_status": generate_data["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
