# MirrAI SD Inpainting RunPod Repo

현재 저장소는 실제 운영 중인 SD 인페인팅 서비스 경로만 남긴 정리본입니다.
레거시 생성 파이프라인, 미사용 랜드마크 백엔드, 구형 RunPod 보조 경로는 제거했고, 현재 기준 실행/배포 흐름은 `SD runtime -> Docker -> GitHub Actions -> RunPod`입니다.

## 저장소 범위

- SD 인페인팅 추론 경로 유지
- RunPod 배포/스모크 테스트 경로 유지
- generation 학습/평가 스크립트 유지
- 로컬 테스트, 학습 산출물, 대용량 모델 파일은 업로드 제외

## 핵심 파일

- `handler_sd.py`: RunPod serverless 엔트리포인트
- `pipeline_sd_inpainting.py`: 실제 SD 추론 파이프라인
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

## 의존성

- `requirements.txt`: 현재 서비스 런타임 의존성
- `requirements-train.txt`: 학습/평가용 추가 의존성
- `requirements-dev.txt`: 로컬 개발용 추가 의존성
- `requirements-trends.txt`: 트렌드 크롤링/RAG 전용 의존성

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
- `docs/trend_pipeline.md`: 통합된 크롤링/RAG 서브시스템 실행 가이드
- `docs/pipeline_runtime_config.md`: 현재 파이프라인이 실제로 읽는 runtime config 기준 문서

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
  --image images/1234.jpeg \
  --hairstyle "wolf cut, layered bangs" \
  --color "ash brown" \
  --top-k 1
```

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

- 코드: `trend_pipeline/`
- 데이터: `data/trend_pipeline/raw`, `data/trend_pipeline/processed`
- 벡터 DB: `data/trend_pipeline/chromadb` (gitignore)

기존 RunPod SD 서비스 경로와 충돌하지 않도록 완전히 분리된 네임스페이스로 넣었습니다. 자세한 사용법은 `docs/trend_pipeline.md`를 보면 됩니다.

## 생성 프롬프트와 트렌드 데이터

런타임 헤어 생성은 `data/llm_refined_trends.json`를 기준으로 요청한 `hairstyle_text`를 해석합니다.

- 한국어/영문 스타일명을 트렌드 레코드와 매칭
- 매칭된 `hairstyle_text` 키워드로 SAM2 힌트, 길이 분류, SD 프롬프트를 보강
- `color_text`는 사용자가 명시한 경우에만 실제 색상 타깃으로 적용

## 검증 기준

현재 정리 이후 기본 확인 명령은 아래 두 개입니다.

```bash
python -m py_compile handler_sd.py pipeline_sd_inpainting.py runtime_download.py scripts/runpod_release.py
python test_runpod.py --health-check
```
