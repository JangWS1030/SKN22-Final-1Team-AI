# MirrAI Internal AI Service API

이 문서는 backend가 `MIRRAI_AI_SERVICE_URL` 뒤에 붙여 호출하는 내부 AI 서비스 계약을 정의한다.

기본 URL 규칙:

- 로컬 개발: `MIRRAI_AI_SERVICE_URL=http://localhost:8000`
- 운영: `MIRRAI_AI_SERVICE_URL=https://mirrai.shop`
- backend는 위 base URL 뒤에 `/internal/health`, `/internal/analyze-face`, `/internal/generate-simulations`, `/internal/explain-style`를 붙여 호출한다.

기본 규칙:

- Base URL: `MIRRAI_AI_SERVICE_URL`
- Development base URL: `http://localhost:8000`
- Production base URL: `https://mirrai.shop`
- Path versioning: 사용하지 않음
- Endpoint prefix: `/internal/...`
- Header versioning: 선택적 `X-MirrAI-API-Version: 2026-04-03`
- Response body versioning: `schema_version`, `response_version` 필수
- 인증: `Authorization: Bearer <MIRRAI_INTERNAL_API_TOKEN>` 권장
- 이미지 URL: signed URL
- 이미지 URL TTL 기본값: `3600s`
- 동기 처리: 현재 모든 endpoint는 sync

## 공통 응답 규칙

성공 응답:

```json
{
  "status": "ok",
  "schema_version": "2026-04-03",
  "response_version": "2026-04-03",
  "request_id": "req_123",
  "processing_time_ms": 187,
  "data": {}
}
```

부분 성공 응답:

```json
{
  "status": "partial_success",
  "schema_version": "2026-04-03",
  "response_version": "2026-04-03",
  "request_id": "req_123",
  "processing_time_ms": 8123,
  "data": {}
}
```

에러 응답:

```json
{
  "status": "error",
  "schema_version": "2026-04-03",
  "response_version": "2026-04-03",
  "request_id": "req_123",
  "processing_time_ms": 12,
  "error": {
    "error_code": "VALIDATION_ERROR",
    "message": "Request validation failed.",
    "detail": {
      "errors": []
    },
    "retryable": false
  }
}
```

공통 스키마 규칙:

- required 필드는 문서 예시와 endpoint 설명에서 명시한다.
- optional 필드는 누락 가능하며, nullable 필드는 명시적으로 `null` 허용이다.
- `schema_version`과 `response_version`은 동일하게 `2026-04-03`을 사용한다.
- `request_id`는 클라이언트가 주지 않으면 서버가 생성한다.

## HTTP status code 규칙

- `200`: 정상 완료 또는 partial success
- `400`: body가 비어 있거나 지원하지 않는 요청 구조
- `401`: 인증 실패
- `403`: 만료된 signed URL, 잘못된 asset token
- `404`: style 또는 asset 미존재
- `409`: `X-MirrAI-API-Version` mismatch
- `422`: validation error, face not detected, landmarks not detected
- `429`: 현재 서버 차원 rate limit 미적용, 추후 reserve
- `500`: 내부 로직 오류
- `502`: 외부 provider 연동 실패 시 reserve
- `503`: timeout 또는 서비스 준비 안 됨 reserve

## Retry / idempotent 규칙

- `GET /internal/health`: idempotent
- `POST /internal/analyze-face`: 동일 입력이면 재시도 가능
- `POST /internal/generate-simulations`: 재시도 가능하지만 새 이미지 asset URL은 다시 발급될 수 있음
- `POST /internal/explain-style`: idempotent

권장 timeout:

- health: 5s ~ 10s
- analyze-face: 15s
- generate-simulations: 120s ~ 180s
- explain-style: 10s

권장 retry:

- `401`, `403`, `404`, `409`, `422`: retry 비권장
- `429`, `500`, `502`, `503`: 최대 2회 exponential backoff

## 인증 / 보안

- Primary: `Authorization: Bearer <MIRRAI_INTERNAL_API_TOKEN>`
- Legacy alias: `X-Internal-API-Key`
- secret source: env 또는 vault 주입
- staging / production 키 분리 권장
- key rotation: 30일 주기 권장

## 1. GET /internal/health

- Method: `GET`
- Body: 없음

성공 응답 예시:

```json
{
  "status": "ok",
  "schema_version": "2026-04-03",
  "response_version": "2026-04-03",
  "request_id": "req_health_001",
  "processing_time_ms": 4,
  "data": {
    "status": "online",
    "role": "model-ai-analysis-service",
    "environment": "staging",
    "build_version": "v70",
    "model_version": "v70",
    "uptime_seconds": 4821,
    "schema_version": "2026-04-03"
  }
}
```

실패 응답 예시:

```json
{
  "status": "error",
  "schema_version": "2026-04-03",
  "response_version": "2026-04-03",
  "request_id": "req_health_002",
  "processing_time_ms": 1,
  "error": {
    "error_code": "AUTH_REQUIRED",
    "message": "A valid bearer token is required.",
    "detail": {},
    "retryable": false
  }
}
```

## 2. POST /internal/analyze-face

- Method: `POST`
- Content-Type: `application/json`
- 입력 이미지: `image_url` 또는 `image_base64` 중 하나 필수
- `image_url`: nullable 아님
- `image_base64`: nullable 아님
- 결과 `image_url`: nullable 허용
- 결과 `image_url`: signed URL

요청 예시:

```json
{
  "request_id": "req_analyze_001",
  "image_base64": "<base64>",
  "include_visualization": true
}
```

성공 응답 예시:

```json
{
  "status": "ok",
  "schema_version": "2026-04-03",
  "response_version": "2026-04-03",
  "request_id": "req_analyze_001",
  "processing_time_ms": 236,
  "data": {
    "face_shape": "oval",
    "face_shape_scores": {
      "oval": 0.4211,
      "round": 0.1084,
      "square": 0.1321,
      "heart": 0.1832,
      "oblong": 0.1552
    },
    "golden_ratio_score": 0.7425,
    "face_ratios": {
      "cheekbone_to_height": 0.721334,
      "jaw_to_height": 0.601241,
      "temple_to_height": 0.701197,
      "jaw_to_cheekbone": 0.833394
    },
    "face_bbox": {
      "x1": 205,
      "y1": 74,
      "x2": 598,
      "y2": 602
    },
    "image_url": "https://service/internal/assets/analyze-face-123?expires=1775200000&token=...",
    "image_url_expires_at": "2026-04-03T15:20:00Z",
    "schema_version": "2026-04-03"
  }
}
```

실패 응답 예시:

```json
{
  "status": "error",
  "schema_version": "2026-04-03",
  "response_version": "2026-04-03",
  "request_id": "req_analyze_002",
  "processing_time_ms": 83,
  "error": {
    "error_code": "FACE_NOT_DETECTED",
    "message": "No face was detected in the provided image.",
    "detail": {},
    "retryable": false
  }
}
```

허용값 / 타입:

- `face_shape`: `oval | round | square | heart | oblong`
- `golden_ratio_score`: `float`, range `0.0 ~ 1.0`
- `processing_time_ms`: `int`

제한:

- 최대 download bytes: `25MB`
- 큰 base64 payload는 `422 VALIDATION_ERROR`
- 현재 별도 동시 요청 제한 필드는 없음

## 3. POST /internal/generate-simulations

- Method: `POST`
- Content-Type: `application/json`
- 처리 방식: sync
- 정렬 규칙: `items`는 `rank ASC`
- 1순위 기준: recommendation score 내림차순

요청 예시:

```json
{
  "request_id": "req_sim_001",
  "client_id": "capture_123",
  "image_base64": "<base64>",
  "analysis_data": {
    "face_shape": "oval",
    "golden_ratio_score": 0.7425,
    "face_ratios": {
      "cheekbone_to_height": 0.721334,
      "jaw_to_height": 0.601241,
      "temple_to_height": 0.701197,
      "jaw_to_cheekbone": 0.833394
    }
  },
  "survey_data": {
    "length": "short",
    "mood": ["natural", "trendy"]
  },
  "scoring_weights": {
    "face": 0.4,
    "golden": 0.2,
    "preference": 0.4
  },
  "color_text": "natural black",
  "top_k": 3
}
```

성공 응답 예시:

```json
{
  "status": "ok",
  "schema_version": "2026-04-03",
  "response_version": "2026-04-03",
  "request_id": "req_sim_001",
  "processing_time_ms": 18234,
  "data": {
    "client_id": "capture_123",
    "processing_mode": "sync",
    "schema_version": "2026-04-03",
    "items": [
      {
        "style_id": "prada-bob-1",
        "style_name": "Prada Bob",
        "rank": 0,
        "score": 0.9132,
        "simulation_image_url": "https://service/internal/assets/simulation-prada-bob-1?expires=1775200000&token=...",
        "simulation_image_url_expires_at": "2026-04-03T15:20:00Z",
        "reasoning_snapshot": {
          "face_shape_detected": "oval",
          "golden_ratio_score": 0.7425,
          "matched_face_shapes": ["oval", "heart", "oblong"],
          "recommendation_score": 0.9132,
          "trend_name": "Prada Bob",
          "description": "Jaw-length clean bob with compact silhouette."
        }
      }
    ],
    "partial_failures": []
  }
}
```

부분 성공 예시:

```json
{
  "status": "partial_success",
  "schema_version": "2026-04-03",
  "response_version": "2026-04-03",
  "request_id": "req_sim_002",
  "processing_time_ms": 23114,
  "data": {
    "client_id": "capture_123",
    "processing_mode": "sync",
    "schema_version": "2026-04-03",
    "items": [],
    "partial_failures": [
      {
        "style_id": "trend-1",
        "style_name": "Sample Style",
        "error_code": "SIMULATION_GENERATION_FAILED",
        "message": "RuntimeError: ..."
      }
    ]
  }
}
```

실패 응답 예시:

```json
{
  "status": "error",
  "schema_version": "2026-04-03",
  "response_version": "2026-04-03",
  "request_id": "req_sim_003",
  "processing_time_ms": 18,
  "error": {
    "error_code": "FACE_ANALYSIS_REQUIRED",
    "message": "face_ratios are required to generate recommendation-based simulations.",
    "detail": {},
    "retryable": false
  }
}
```

필드 규칙:

- `items[*].style_id`: recommendation catalog source 기준 고정
- `items[*].simulation_image_url`: signed URL
- `items[*].reasoning_snapshot`: object, required
- `score`: float, nullable 아님
- `rank`: int, nullable 아님

## 4. POST /internal/explain-style

- Method: `POST`
- Content-Type: `application/json`
- 처리 방식: sync
- idempotent: yes

요청 예시:

```json
{
  "request_id": "req_explain_001",
  "style_id": "prada-bob-1",
  "analysis_data": {
    "face_shape": "oval",
    "golden_ratio_score": 0.7425
  },
  "survey_data": {
    "length": "short"
  },
  "simulation_image_url": "https://service/internal/assets/simulation-prada-bob-1?expires=1775200000&token=..."
}
```

성공 응답 예시:

```json
{
  "status": "ok",
  "schema_version": "2026-04-03",
  "response_version": "2026-04-03",
  "request_id": "req_explain_001",
  "processing_time_ms": 19,
  "data": {
    "style_id": "prada-bob-1",
    "style_name": "Prada Bob",
    "llm_explanation": "oval face shape compatibility is explicitly supported by this style. It aligns with the target mood tags: natural, trendy.",
    "simulation_image_url": "https://service/internal/assets/simulation-prada-bob-1?expires=1775200000&token=...",
    "card": {
      "style_id": "prada-bob-1",
      "style_name": "Prada Bob",
      "summary": "Jaw-length clean bob with compact silhouette.",
      "why_it_matches": [
        "oval face shape compatibility is explicitly supported by this style."
      ],
      "styling_points": [
        "Target length: short.",
        "Expected maintenance level: medium."
      ],
      "cautions": [
        "Simulation output should be treated as reference imagery, not a guaranteed salon result."
      ],
      "simulation_image_url": "https://service/internal/assets/simulation-prada-bob-1?expires=1775200000&token=..."
    }
  }
}
```

실패 응답 예시:

```json
{
  "status": "error",
  "schema_version": "2026-04-03",
  "response_version": "2026-04-03",
  "request_id": "req_explain_002",
  "processing_time_ms": 7,
  "error": {
    "error_code": "STYLE_NOT_FOUND",
    "message": "Requested style could not be found in the recommendation catalog.",
    "detail": {
      "style_id": "unknown-style",
      "style_name": null
    },
    "retryable": false
  }
}
```

## 비동기 처리 여부

- 현재 `generate-simulations`, `explain-style` 모두 sync
- `job_id` 없음
- polling endpoint 없음
- 결과 보관: signed image URL TTL 동안 유효

## 운영 메모

- `.env` 또는 runtime env에서 아래 값을 설정한다.
  - backend development: `MIRRAI_AI_SERVICE_URL=http://localhost:8000`
  - backend production: `MIRRAI_AI_SERVICE_URL=https://mirrai.shop`
  - `MIRRAI_SERVICE_MODE=http`
  - `MIRRAI_HTTP_HOST=0.0.0.0`
  - `MIRRAI_HTTP_PORT=8000`
  - `MIRRAI_INTERNAL_API_TOKEN=<token>`
  - `MIRRAI_ASSET_SIGNING_SECRET=<secret>`
  - `MIRRAI_SERVICE_ENV=staging|production|local`

- 현재 RunPod serverless endpoint는 계속 기존 `handler_sd.py`를 사용한다.
- 로컬 HTTP 모드는 `http://localhost:8000`, 운영 HTTP 모드는 `https://mirrai.shop`를 base URL로 사용한다.
