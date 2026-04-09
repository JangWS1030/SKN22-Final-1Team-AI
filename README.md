# MirrAI SD Inpainting RunPod Repo

현재 저장소는 실제 운영 중인 SD 인페인팅 서비스 경로만 남긴 정리본입니다.
레거시 생성 파이프라인, 미사용 랜드마크 백엔드, 구형 RunPod 보조 경로는 제거했고, 현재 기준 실행/배포 흐름은 `SD runtime -> Docker -> GitHub Actions -> RunPod`입니다.

## 저장소 범위

- SD 인페인팅 추론 경로 유지
- RunPod 배포/스모크 테스트 경로 유지
- generation 학습/평가 스크립트 유지
- 로컬 테스트, 학습 산출물, 대용량 모델 파일은 업로드 제외

## 핵심 파일

- `handler_sd.py`: RunPod serverless 엔트리포인트 (헬스체크 / 직접 생성 라우팅)
- `internal_api_app.py`: `/internal/...` HTTP facade 엔트리포인트
- `pipeline_sd_inpainting.py`: 실제 SD 추론 파이프라인
- `pipeline_sd_components/`: `pipeline_sd_inpainting.py`에서 분리한 로딩 / 프롬프트 / 후처리 모듈
- `runtime_download.py`: 런타임 모델 캐시 준비
- `download_weights_sd.py`: SD 관련 가중치 다운로드 보조 스크립트
- `entrypoint_sd.sh`: 컨테이너 시작 스크립트
- `Dockerfile.sd.base`: 공통 의존성 베이스 이미지
- `Dockerfile.sd.app`: 현재 CI/CD에서 사용하는 앱 이미지
- `Dockerfile.sd`: 수동 단일 이미지 빌드용

## 현재 파이프라인 구성

- SegFace 기반 hair mask
- SAM2 기반 refinement
- MediaPipe FaceMesh 기반 face protect mask
- Stable Diffusion inpainting
- ControlNet canny
- IP-Adapter face

### short/medium 의상 전면 처리 순서

- `source_garment_prepass_mask`로 torso-front / chest-center 의상 복원 영역을 먼저 확보
- `upper_clothes_overwrite`와 short 전용 torso repaint seed를 prepass / guard release에 선반영
- 이후 본 SD inpainting에서 short/medium silhouette를 생성
- 마지막에 `short_lower_tail_cleanup`, `short_lower_cloth_hard_override`, `final_source_cloth_rescue` 같은 후처리로 잔존 artifact를 정리

즉 현재 short/medium 경로는 "생성 후 의상 복원만 하는 구조"가 아니라, 의상 전면 복원 마스크를 먼저 열어두고 본 생성과 후처리를 이어가는 구조입니다.

## 최근 업데이트 반영

- **입력 이미지 표준화**: `enable_input_standardization` 로직 추가로 인물 중심 스튜디오 비율 최적화 지원
- **단발/중단발 마스크 개선**: 얼굴/목/가슴 영역 세분화를 통한 의상(어깨 선, 밝은 옷 등) 및 피부 보존/복원 로직 대폭 강화
- **source garment prepass 확장**: short 변환에서 torso hair side-column까지 prepass / bridge / ControlNet suppression 경로에 반영
- **RunPod 환경 및 모니터링 대응**: `$RUNPOD_POD_ID` 등 웹훅 환경 변수 자동 정규화, API 응답에 빌드 태그 및 노드 메타 정보 추가
- **디버그 마스크 응답 강화**: 여러 마스크를 분리하여 확인할 수 있도록 핸들러 리턴 구조 개편

## 의존성

- `requirements.txt`: 현재 서비스 런타임 의존성
- `requirements-train.txt`: 학습/평가용 추가 의존성
- `requirements-dev.txt`: 로컬 개발용 추가 의존성

`pipeline_sd_components/` 분리는 코드 구조 변경만 포함하고, 런타임/학습용 서드파티 패키지 추가는 없습니다.

런타임 설치:

```bash
python -m pip install -r requirements.txt
```

학습/평가 환경 설치:

```bash
python -m pip install -r requirements-train.txt
```

## 로컬/업로드 제외 범위

아래 항목은 로컬 전용으로 유지하고 업로드 대상에서 제외합니다.

- `tests/`
- `images/`
- `output/`, `cmd/`
- `dataset_build/`
- `pretrained_models/`
- 생성된 `.docx`

## 남겨둔 문서

- `README_release_runbook.md`: 현재 GitHub Actions + RunPod 릴리스 절차
- `README_runpod_volume.md`: RunPod cold start 완화용 volume 설정 메모
- `docs/pipeline_runtime_config.md`: 현재 파이프라인이 실제로 읽는 runtime config 기준 문서
- `docs/internal_ai_service_api.md`: backend 연동용 내부 AI 서비스 API 계약

## Internal AI Service Base URL

- 개발: `MIRRAI_AI_SERVICE_URL=http://localhost:8000`
- 운영: `MIRRAI_AI_SERVICE_URL=https://mirrai.shop`
- backend는 위 base URL 뒤에 `/internal/health`, `/internal/analyze-face`를 붙여 호출합니다.
- 내부 API는 path versioning 없이 `/internal/...`를 사용하고, 선택적으로 `X-MirrAI-API-Version` 헤더를 받을 수 있습니다.

## RunPod API 엔드포인트

`handler_sd.py`는 단일 RunPod Serverless 핸들러에서 `action` 필드 또는 입력 구조에 따라 2개 기능을 라우팅합니다.

### 공통

- **URL**: `https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync` (동기) 또는 `/run` (비동기)
- **Header**: `Authorization: Bearer {RUNPOD_API_KEY}`
- **Content-Type**: `application/json`

---

### EP0. 헬스체크

서버 상태 및 GPU 정보 확인.

**Request**
```json
{
  "input": {
    "action": "health_check"
  }
}
```

**Response**
```json
{
  "status": "ok",
  "cuda": {
    "available": true,
    "device": "NVIDIA A40"
  }
}
```

---

### EP1. 직접 지정 생성 (기존)

hairstyle/color 텍스트를 직접 지정하여 이미지를 생성합니다.

**Request**
```json
{
  "input": {
    "image":               "<base64 or URL>",
    "hairstyle_text":      "wolf cut, layered bangs",
    "color_text":          "ash brown",
    "white_tshirt_experiment": false,
    "top_k":               3,
    "return_base64":       true,
    "return_intermediates": false,
    "mask_debug_only":     false,
    "bg_fill_mode":        "cv2",
    "lora_path":           null,
    "lora_scale":          1.0
  }
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `image` | string | O | base64 인코딩 이미지 또는 URL |
| `hairstyle_text` | string | O* | 헤어스타일 설명 (영문/한글) |
| `color_text` | string | | 헤어 색상 |
| `white_tshirt_experiment` | bool | | `true`면 메인 generation / garment refine 모두 plain white t-shirt 기준으로 고정 |
| `top_k` | int | | 결과 수 (1~5, 기본 3) |
| `return_base64` | bool | | 결과 이미지 base64 포함 (기본 true) |
| `return_intermediates` | bool | | 디버그 중간 산출물 포함 |
| `bg_fill_mode` | string | | `"cv2"` (Poisson blend) 또는 `"sd"` (diffusion) |

**Response**
```json
{
  "results": [
    {
      "rank": 0,
      "seed": 42,
      "clip_score": 0.312,
      "mask_used": "sam2",
      "image_base64": "...",
      "mask_base64": "...",
      "mask_overlay_base64": "...",
      "face_bbox": {"x1": 100, "y1": 50, "x2": 300, "y2": 350}
    }
  ],
  "intermediates": {},
  "elapsed_seconds": 12.3
}
```

---

## 실행 예시

RunPod handler 로컬 실행:

```bash
python handler_sd.py
```

헬스체크:

```bash
python test_runpod.py --health-check
```

샘플 요청:

```bash
python test_runpod.py \
  --image images/1234.jpg \
  --hairstyle "wolf cut, layered bangs" \
  --color "ash brown" \
  --top-k 1
```

단발/중단발 마스크 비교:

```bash
python test_runpod.py \
  --image images/1234.jpg \
  --hairstyle "short chin-length bob cut, hush cut" \
  --top-k 1 \
  --bg-fill sd \
  --mask-refine-mode segface_priority
```

마스크 비교 모드:

- `sam2`: 기본 경로
- `segface_priority`: SegFace 코어 유지 + SAM2 경계 보정만 약하게 반영
- `segface_only`: SegFace 마스크만 사용

## CI/CD

push 시 현재 기준으로 아래 워크플로가 동작합니다.

- `.github/workflows/build-sd-base.yml`
- `.github/workflows/build-sd-app.yml`
- `.github/workflows/release-runpod.yml`

배포 세부 절차는 `README_release_runbook.md`를 기준으로 봅니다.

## 스크립트 범위

현재 `scripts/`에는 아래 범주의 스크립트가 남아 있습니다.

- RunPod 릴리스/스모크 테스트
- generation 학습 데이터 전처리
- generation 학습 실행
- generation 벤치마크/평가
- 문서 자동 생성 및 결과 정리

## 참고 문서

- `README_release_runbook.md`: GitHub Actions + RunPod 릴리스 절차
- `README_runpod_volume.md`: RunPod cold start 완화용 volume 설정 메모
- `docs/`: generation 학습/평가 및 런타임 설정 관련 문서

기존 StyleGAN / HairCLIP 기반 레거시 경로와 미사용 랜드마크·세그멘테이션 호환 경로는 저장소에서 제거했습니다.

## 생성 프롬프트 입력

- 기본 입력은 요청에 들어온 `hairstyle_text`, `color_text` 그대로 사용
- 백엔드가 `sd_prompt_data`를 함께 보내면 해당 `sd_positive` / `sd_negative` / `sd_guidance`를 우선 사용
- `sd_prompt_data`가 없으면 파이프라인이 `hairstyle_text`와 `color_text`를 기반으로 폴백 SD 프롬프트를 구성
- `white_tshirt_experiment=true`면 source garment hint를 무시하고 메인 generation과 garment/cloth refine 모두 `plain white t-shirt` 기준 prompt를 사용
- `color_text`는 사용자가 명시한 경우에만 실제 색상 타깃으로 적용

## 검증 기준

현재 정리 이후 기본 확인 명령은 아래 두 개입니다.

```bash
python -m py_compile handler_sd.py pipeline_sd_inpainting.py pipeline_sd_components/loading.py pipeline_sd_components/output.py pipeline_sd_components/prompt.py runtime_download.py scripts/runpod_release.py
python test_runpod.py --health-check
```
