# MirrAI SD Inpainting RunPod Repo

현재 저장소는 레거시 생성 경로를 제거한 뒤, 실제로 쓰는 SD 인페인팅 서비스 경로만 남긴 통합본입니다.

## 현재 기준 실행 경로

- `handler_sd.py`: RunPod serverless 진입점
- `pipeline_sd_inpainting.py`: 실제 추론 파이프라인
- `runtime_download.py`: Hugging Face 모델 캐시 워밍
- `entrypoint_sd.sh`: 컨테이너 시작 스크립트
- `Dockerfile.sd.base`: 공통 의존성 베이스 이미지
- `Dockerfile.sd.app`: 현재 CI/CD가 사용하는 앱 이미지
- `Dockerfile.sd`: 수동 단일 이미지 빌드용

## 의존성 파일

- `requirements.txt`: 현재 서비스 런타임 전체 의존성
- `requirements-train.txt`: 학습용 추가 의존성
- `requirements-dev.txt`: 로컬 개발용 추가 의존성

설치 예시:

```bash
python -m pip install -r requirements.txt
```

학습 환경:

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
- `docs/trend_pipeline.md`: 통합된 크롤링/RAG 서브시스템 실행 가이드

## 실행 예시

RunPod handler 로컬 실행:

```bash
python handler_sd.py
```

헬스체크:

```bash
python tests/test_runpod.py --health-check
```

샘플 요청:

```bash
python tests/test_runpod.py \
  --image images/1234.jpeg \
  --hairstyle "wolf cut, layered bangs" \
  --color "ash brown" \
  --top-k 1
```

## 모델 메모

현재 파이프라인은 다음 구성을 사용합니다.

- SegFace 기반 hair mask
- SAM2 기반 refinement
- MediaPipe FaceMesh 기반 face protect mask
- Stable Diffusion inpainting
- ControlNet canny
- IP-Adapter face

기존 StyleGAN / HairCLIP 기반 레거시 경로와 미사용 랜드마크·세그멘테이션 호환 경로는 저장소에서 제거했습니다.

## 통합된 트렌드 크롤링/RAG

상위 폴더의 `crawling_git`는 현재 저장소 기준으로 흡수했습니다.

- 코드: `trend_pipeline/`
- 데이터: `data/trend_pipeline/raw`, `data/trend_pipeline/processed`
- 벡터 DB: `data/trend_pipeline/chromadb` (gitignore)

기존 RunPod SD 서비스 경로와 충돌하지 않도록 완전히 분리된 네임스페이스로 넣었습니다. 자세한 사용법은 `docs/trend_pipeline.md`를 보면 됩니다.

## 생성 프롬프트와 트렌드 데이터

런타임 헤어 생성은 이제 `data/llm_refined_trends.json`를 기준으로 요청한 `hairstyle_text`를 해석합니다.

- 한국어/영문 스타일명을 트렌드 레코드와 매칭
- 매칭된 `hairstyle_text` 키워드로 SAM2 힌트, 길이 분류, SD 프롬프트를 보강
- `color_text`는 사용자가 명시한 경우에만 실제 색상 타깃으로 적용
