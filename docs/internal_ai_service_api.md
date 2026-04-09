# MirrAI Internal AI Service API

이 문서는 backend가 `MIRRAI_AI_SERVICE_URL` 뒤에 붙여 호출하는 내부 AI 서비스 계약을 정의한다.

기본 URL 규칙:

- 로컬 개발: `MIRRAI_AI_SERVICE_URL=http://localhost:8000`
- 운영: `MIRRAI_AI_SERVICE_URL=https://mirrai.shop`
- backend는 위 base URL 뒤에 `/internal/health`, `/internal/analyze-face`를 붙여 호출한다.

기본 규칙:

- Base URL: `MIRRAI_AI_SERVICE_URL`
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

## HTTP status code 규칙

- `200`: 정상 완료
- `401`: 인증 실패
- `403`: 만료된 signed URL, 잘못된 asset token
- `404`: asset 미존재
- `409`: `X-MirrAI-API-Version` mismatch
- `422`: validation error, face not detected, landmarks not detected
- `500`: 내부 로직 오류

## Retry / idempotent 규칙

- `GET /internal/health`: idempotent
- `POST /internal/analyze-face`: 동일 입력이면 재시도 가능

권장 timeout:

- health: 5s ~ 10s
- analyze-face: 15s

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

## 2. POST /internal/analyze-face

- Method: `POST`
- Content-Type: `application/json`
- 입력 이미지: `image_url` 또는 `image_base64` 중 하나 필수
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

## 3. GET /internal/assets/{asset_id}

- Method: `GET`
- Query: `expires`, `token`
- 용도: `analyze-face` 시각화 signed URL 조회

오류 코드:

- `403 ASSET_URL_EXPIRED`
- `403 INVALID_ASSET_SIGNATURE`
- `404 ASSET_NOT_FOUND`
