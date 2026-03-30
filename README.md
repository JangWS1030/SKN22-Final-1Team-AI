# MirrAI SD Inpainting RunPod Repo

현재 저장소는 실제 운영 중인 SD 인페인팅 서비스 경로만 남긴 정리본입니다.
레거시 생성 파이프라인, 미사용 랜드마크 백엔드, 구형 RunPod 보조 경로는 제거했고, 현재 기준 실행/배포 흐름은 `SD runtime -> Docker -> GitHub Actions -> RunPod`입니다.

## 저장소 범위

- SD 인페인팅 추론 경로 유지
- RunPod 배포/스모크 테스트 경로 유지
- generation 학습/평가 스크립트 유지
- 로컬 테스트, 학습 산출물, 대용량 모델 파일은 업로드 제외

## 핵심 파일

- `handler_sd.py`: RunPod serverless 엔트리포인트 (EP0~EP3 라우팅)
- `pipeline_sd_inpainting.py`: 실제 SD 추론 파이프라인
- `pipeline_sd_components/`: `pipeline_sd_inpainting.py`에서 분리한 로딩 / 프롬프트 / 후처리 모듈
- `style_recommender.py`: 얼굴형 + 취향벡터 → 스타일 추천 엔진 (ChromaDB 코사인 유사도)
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

## 최근 업데이트 반영

- **입력 이미지 표준화**: `enable_input_standardization` 로직 추가로 인물 중심 스튜디오 비율 최적화 지원
- **단발/중단발 마스크 개선**: 얼굴/목/가슴 영역 세분화를 통한 의상(어깨 선, 밝은 옷 등) 및 피부 보존/복원 로직 대폭 강화
- **RunPod 환경 및 모니터링 대응**: `$RUNPOD_POD_ID` 등 웹훅 환경 변수 자동 정규화, API 응답에 빌드 태그 및 노드 메타 정보 추가
- **디버그 마스크 응답 강화**: 여러 마스크를 분리하여 확인할 수 있도록 핸들러 리턴 구조 개편

## 의존성

- `requirements.txt`: 현재 서비스 런타임 의존성
- `requirements-train.txt`: 학습/평가용 추가 의존성
- `requirements-dev.txt`: 로컬 개발용 추가 의존성
- `requirements-trends.txt`: 트렌드 크롤링/RAG 전용 의존성

`pipeline_sd_components/` 분리는 코드 구조 변경만 포함하고, 런타임/학습용 서드파티 패키지 추가는 없습니다.

런타임 설치:

```bash
python -m pip install -r requirements.txt
```

학습/평가 환경 설치:

```bash
python -m pip install -r requirements-train.txt
```

트렌드 파이프라인 설치:

```bash
python -m pip install -r requirements-trends.txt
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
- `docs/rag_pipeline.md`: 통합된 크롤링/RAG 서브시스템 실행 가이드
- `docs/rag_evaluation.md`: stylist-rag와 no-rag 비교 평가 결과
- `docs/pipeline_runtime_config.md`: 현재 파이프라인이 실제로 읽는 runtime config 기준 문서

## RunPod API 엔드포인트

`handler_sd.py`는 단일 RunPod Serverless 핸들러에서 `action` 필드 또는 입력 구조에 따라 4개 기능을 라우팅합니다.

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

### EP2. 추천 기반 생성 (취향벡터 + RAG)

얼굴 분석 데이터 + 사용자 취향 → 스타일 추천 → 추천 스타일별 1장씩 생성합니다.
`face_ratios`가 있으면 자동으로 추천 모드로 진입합니다.

**Request (구조화된 취향)**
```json
{
  "input": {
    "image": "<base64 or URL>",
    "face_ratios": {
      "cheekbone_to_height": 0.72,
      "jaw_to_height": 0.60,
      "temple_to_height": 0.70,
      "jaw_to_cheekbone": 0.83
    },
    "preference": {
      "length": "medium",
      "mood": ["trendy", "natural"],
      "hair_type": "wavy",
      "color_temp": "warm",
      "budget": "medium"
    },
    "color_text": "ash brown",
    "top_k": 5,
    "return_base64": true
  }
}
```

**Request (자연어 취향 + 나이)**
```json
{
  "input": {
    "image": "<base64 or URL>",
    "face_ratios": {
      "cheekbone_to_height": 0.72,
      "jaw_to_height": 0.60,
      "temple_to_height": 0.70,
      "jaw_to_cheekbone": 0.83
    },
    "preference_text": "자연스러운 웨이브 미디엄 길이, 따뜻한 톤",
    "age": 28,
    "top_k": 5,
    "return_base64": true
  }
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `image` | string | O | base64 인코딩 이미지 또는 URL |
| `face_ratios` | object | O | MediaPipe 얼굴 비율 (EP1 얼굴 분석 결과) |
| `preference` | object | △ | 구조화된 취향 벡터 (옵션 A) |
| `preference_text` | string | △ | 자연어 취향 텍스트 (옵션 B) |
| `age` | int | | 나이 (분위기 추론에 사용) |
| `color_text` | string | | 헤어 색상 |
| `top_k` | int | | 추천 수 (1~5, 기본 5) |

`preference` 필드 상세:

| 키 | 값 | 설명 |
|------|------|------|
| `length` | `"short"` / `"medium"` / `"long"` | 선호 길이 |
| `mood` | `["natural", "trendy", "classic", "edgy", "cute"]` | 분위기 (복수 선택) |
| `hair_type` | `"straight"` / `"wavy"` / `"curly"` | 모발 타입 |
| `color_temp` | `"warm"` / `"cool"` / `"neutral"` | 색상 톤 |
| `budget` | `"low"` / `"medium"` / `"high"` | 예산/관리 수준 |

**Response**
```json
{
  "results": [
    {
      "rank": 0,
      "seed": 42,
      "clip_score": 0.298,
      "mask_used": "sam2",
      "image_base64": "...",
      "recommended_style": {
        "style_id": "shaggy-midi",
        "style_name": "Shaggy Midi Cut",
        "recommendation_score": 0.8437
      }
    },
    {
      "rank": 1,
      "seed": 77,
      "clip_score": 0.285,
      "mask_used": "sam2",
      "image_base64": "...",
      "recommended_style": {
        "style_id": "layered-midi-waves",
        "style_name": "Layered Midi Waves",
        "recommendation_score": 0.8122
      }
    }
  ],
  "recommendations": [
    {
      "rank": 0,
      "style_id": "shaggy-midi",
      "style_name": "Shaggy Midi Cut",
      "score": 0.8437,
      "face_shapes": ["round", "oval", "oblong"],
      "description": "Mid-length shag with crown texture...",
      "face_shape_detected": "oval",
      "golden_ratio_score": 0.649
    }
  ],
  "rag_context": "[자료 1]\n제목: ...\n요약: ...",
  "elapsed_seconds": 58.7
}
```

---

### EP3. 트렌드 데이터 최신화

트렌드 크롤링/정제/벡터DB 갱신. 두 가지 모드를 지원합니다.

#### 모드 A: RunPod 내부 파이프라인 실행

```json
{
  "input": {
    "action": "refresh_trends",
    "steps": ["crawl", "refine", "llm_refine", "vectorize", "rebuild_styles"]
  }
}
```

`steps` 생략 시 전체 실행. 사용 가능한 단계:

| 단계 | 설명 |
|------|------|
| `crawl` | 28개 패션 사이트 웹 크롤링 |
| `refine` | 크롤링 데이터 텍스트 정제 |
| `llm_refine` | Gemini 기반 요약/구조화 |
| `vectorize` | ChromaDB 트렌드 벡터DB 갱신 |
| `rebuild_styles` | 스타일 추천 컬렉션 리빌드 |
| `analyze` | 키워드 분석 (선택) |

#### 모드 B: Django에서 빌드한 ChromaDB 수신 (권장)

Django 서버(CPU)에서 크롤링 + 정제 + 벡터화까지 수행한 뒤, ChromaDB 파일을 tar.gz로 압축하여 전송합니다.
RunPod에서는 압축 해제 + 컬렉션 핫 리로드만 수행하므로 GPU 비용이 들지 않습니다.

```json
{
  "input": {
    "action": "refresh_trends",
    "chromadb_tar_base64": "<base64 encoded tar.gz>"
  }
}
```

아카이브 내부 구조:
```
chromadb_trends/    (트렌드 벡터DB)
chromadb_ncs/       (NCS 시술 벡터DB)
chromadb_styles/    (스타일 추천 컬렉션, 선택)
```

Django 측 전송 예시:
```python
import base64, tarfile, io

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    tar.add("data/rag/stores/chromadb_trends", arcname="chromadb_trends")
    tar.add("data/rag/stores/chromadb_ncs", arcname="chromadb_ncs")
    tar.add("data/rag/stores/chromadb_styles", arcname="chromadb_styles")

payload = {
    "input": {
        "action": "refresh_trends",
        "chromadb_tar_base64": base64.b64encode(buf.getvalue()).decode()
    }
}
# requests.post(RUNPOD_URL, json=payload, headers=...)
```

**Response (모드 A)**
```json
{
  "success": true,
  "steps_requested": ["crawl", "refine", "llm_refine", "vectorize", "rebuild_styles"],
  "steps_completed": ["crawl", "refine", "llm_refine", "vectorize", "rebuild_styles"],
  "steps_failed": [],
  "details": {
    "crawl": {"status": "ok", "elapsed_seconds": 120.5},
    "vectorize": {"status": "ok", "document_count": 85, "elapsed_seconds": 3.2},
    "rebuild_styles": {"status": "ok", "style_count": 15, "elapsed_seconds": 1.1}
  },
  "total_elapsed_seconds": 180.3,
  "elapsed_seconds": 180.5
}
```

**Response (모드 B)**
```json
{
  "success": true,
  "mode": "receive_archive",
  "replaced_collections": ["chromadb_trends", "chromadb_ncs", "chromadb_styles"],
  "archive_size_mb": 4.2,
  "elapsed_seconds": 2.1
}
```

---

### 추천 벡터 가중치 구조

스타일 추천은 3개 벡터를 가중 결합하여 코사인 유사도로 Top-K를 선정합니다.

| 입력 벡터 | 구성 요소 | 가중치 |
|-----------|-----------|--------|
| 얼굴 비율 벡터 | 비율 측정값 + 얼굴형 분류 결과 (oval/round/square/heart/oblong) | 40% |
| 황금비율 근접도 | 얼굴 세로/가로 비율의 황금비(1.618) 편차 점수 | 20% |
| user_preference_vector | 길이 / 분위기 / 모발 타입 / 컬러 톤 / 예산 | 40% |

벡터 차원 구성 (총 23차원):
```
face_shape   [5]  oval, round, square, heart, oblong
golden_ratio [1]  0~1 (1=완벽한 황금비)
length       [3]  short, medium, long
mood         [5]  natural, trendy, classic, edgy, cute
hair_type    [3]  straight, wavy, curly
color_temp   [3]  warm, cool, neutral
budget       [3]  low, medium, high
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

스타일 추천 테스트 (로컬):

```bash
python style_recommender.py
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

## 통합된 트렌드 크롤링/RAG

상위 폴더의 `crawling_git`는 현재 저장소 기준으로 흡수했습니다.

- 코드: `rag_pipeline/`
- 데이터: `data/rag/`
- 원본 PDF: `data/rag/sources/ncs/`
- 벡터 DB:
  `data/rag/stores/chromadb_trends` (트렌드)
  `data/rag/stores/chromadb_ncs` (NCS 시술)
  `data/rag/stores/chromadb_styles` (스타일 추천)
- 통합 질의:
  `python -m rag_pipeline.main stylist-rag --query "요즘 유행하는 단발 추천하고 시술 포인트도 알려줘"`
- 평가:
  `python -m rag_pipeline.main stylist-eval --top-k 3`

기존 RunPod SD 서비스 경로와 충돌하지 않도록 완전히 분리된 네임스페이스로 넣었습니다. 자세한 사용법은 `docs/rag_pipeline.md`를 보면 됩니다.

## 생성 프롬프트와 트렌드 데이터

런타임 헤어 생성은 `data/llm_refined_trends.json`를 기준으로 요청한 `hairstyle_text`를 해석합니다.

- 한국어/영문 스타일명을 트렌드 레코드와 매칭
- 매칭된 `hairstyle_text` 키워드로 SAM2 힌트, 길이 분류, SD 프롬프트를 보강
- `color_text`는 사용자가 명시한 경우에만 실제 색상 타깃으로 적용

## 검증 기준

현재 정리 이후 기본 확인 명령은 아래 두 개입니다.

```bash
python -m py_compile handler_sd.py pipeline_sd_inpainting.py pipeline_sd_components/loading.py pipeline_sd_components/prompt.py pipeline_sd_components/postprocess.py runtime_download.py scripts/runpod_release.py
python test_runpod.py --health-check
```
