# 수동 릴리스 런북

이 문서는 이 저장소에서 사람이 직접 아래 순서를 수행하는 방법을 정리한 운영 문서입니다.

- 테스트
- 수정
- 커밋 / 푸시
- GitHub Actions 빌드
- Docker Hub 푸시 확인
- RunPod Serverless 이미지 교체
- 최종 테스트

이 문서는 이번 작업에서 실제로 사용한 절차와 주의점을 기준으로 작성했습니다.

## 1. 사전 준비

### 로컬 도구

- `git`
- `gh` (GitHub CLI)
- `docker`
- `python3`
- `requests`, `python-dotenv`, `Pillow`

확인 명령:

```bash
gh auth status
docker version
python3 -c 'import requests,dotenv,PIL; print("deps_ok")'
```

### 로컬 인증 상태

- GitHub: `gh auth login`
- Docker Hub: Docker Desktop 또는 `docker login`
- RunPod: `.env`에 아래 값 필요

```bash
RUNPOD_API_KEY=...
RUNPOD_ENDPOINT_ID=...
```

이 저장소의 테스트 스크립트와 배포 스크립트는 `.env`를 읽습니다.

## 2. 현재 배포본 기준선 확인

배포 전에 현재 서버가 살아 있는지 먼저 확인합니다.

헬스체크:

```bash
set -a
source .env
set +a
python3 test_runpod.py --health-check
```

샘플 생성 테스트:

```bash
set -a
source .env
set +a
python3 test_runpod.py \
  --image images/1234.jpeg \
  --hairstyle "short chin-length bob cut, hush cut" \
  --color "ash beige" \
  --top-k 1 \
  --filename-prefix baseline_check
```

확인 포인트:

- 헬스체크 응답에 `"status": "ok"`
- 샘플 이미지가 `output/` 아래 저장됨
- 응답 시간이 비정상적으로 길지 않음

## 3. 코드 / 워크플로 수정

필요한 코드를 수정합니다.

이번 작업에서 실제로 수정한 파일:

- `.github/workflows/build-sd-base.yml`
- `.github/workflows/build-sd-app.yml`
- `scripts/update_runpod_serverless.py`

### GitHub Actions 캐시 주의점

`docker/build-push-action`에 `cache-from: type=gha` / `cache-to: type=gha`를 넣을 때는 `docker/setup-buildx-action`에서 `driver: docker`를 쓰면 안 됩니다.

실패 증상:

```text
ERROR: failed to build: Cache export is not supported for the docker driver.
```

해결:

- `docker/setup-buildx-action`의 `with: driver: docker` 제거
- 기본 `docker-container` 드라이버 사용

### RunPod 템플릿 이미지 교체 스크립트

이번 작업에서 추가한 스크립트:

```bash
python3 scripts/update_runpod_serverless.py --image byoungj/sd:v61 --wait
```

이 스크립트가 하는 일:

1. 현재 엔드포인트 조회
2. 연결된 `templateId` 조회
3. 템플릿의 `imageName`을 새 태그로 교체
4. 엔드포인트 버전 증가 및 워커 롤아웃 대기

## 4. 로컬 검증

수정 후 최소 검증:

```bash
python3 -m py_compile scripts/update_runpod_serverless.py test_runpod.py
```

RunPod 배포 스크립트 dry-run:

```bash
set -a
source .env
set +a
python3 scripts/update_runpod_serverless.py --image byoungj/sd:v58 --dry-run
```

## 5. 커밋 / 푸시

```bash
git status
git add .github/workflows/build-sd-base.yml \
        .github/workflows/build-sd-app.yml \
        scripts/update_runpod_serverless.py
git commit -m "ci: add build cache and runpod deploy script"
git push origin master
```

필요하면 후속 수정 커밋도 같은 방식으로 추가 푸시합니다.

이번 작업에선 캐시 설정 실패를 바로 잡기 위해 추가 커밋을 한 번 더 만들었습니다.

## 6. GitHub Actions 빌드 확인

워크플로 목록 확인:

```bash
gh workflow list --repo PracLee/hair_swap_model
```

최근 런 확인:

```bash
gh run list --repo PracLee/hair_swap_model --limit 10
```

특정 워크플로만 확인:

```bash
gh run list --repo PracLee/hair_swap_model --workflow "Build and Push SD Base Image" --limit 3
gh run list --repo PracLee/hair_swap_model --workflow "Build and Push SD App Image" --limit 3
```

실시간 감시:

```bash
gh run watch <run-id> --repo PracLee/hair_swap_model --exit-status
```

실패 로그:

```bash
gh run view <run-id> --repo PracLee/hair_swap_model --log-failed
```

## 7. Docker Hub 푸시 확인

GitHub Actions가 끝났는지와 별개로, 레지스트리에 태그가 실제로 올라왔는지 직접 확인할 수 있습니다.

예시:

```bash
python3 - <<'PY'
import requests
for tag in ["v61", "latest", "base-latest"]:
    url = f"https://hub.docker.com/v2/namespaces/byoungj/repositories/sd/tags/{tag}"
    r = requests.get(url, timeout=30)
    print(tag, r.status_code, r.json().get("last_updated") if r.ok else None)
PY
```

이번 작업 기준 실제 확인 포인트:

- `byoungj/sd:v61`
- `byoungj/sd:latest`
- `byoungj/sd:base-latest`

주의:

- GitHub Actions UI가 아직 `in_progress`여도 Docker Hub 태그가 먼저 보일 수 있습니다.
- 이 경우 태그 timestamp와 RunPod 반영 상태를 함께 보고 판단합니다.

## 8. RunPod Serverless 템플릿 이미지 교체

### 스크립트 사용

가장 안전한 방법은 추가된 스크립트를 쓰는 것입니다.

```bash
set -a
source .env
set +a
python3 scripts/update_runpod_serverless.py \
  --image byoungj/sd:v61 \
  --wait \
  --timeout 1800 \
  --poll-interval 15
```

### 사람이 API로 직접 할 때의 순서

1. 엔드포인트 조회

```bash
GET https://rest.runpod.io/v1/endpoints/{endpointId}
```

2. 응답에서 `templateId` 확인

3. 템플릿 조회

```bash
GET https://rest.runpod.io/v1/templates/{templateId}?includeEndpointBoundTemplates=true
```

4. 템플릿 이미지 변경

```bash
POST https://rest.runpod.io/v1/templates/{templateId}/update
Content-Type: application/json

{
  "imageName": "byoungj/sd:v61"
}
```

5. 엔드포인트 버전 증가 및 워커 교체 확인

```bash
GET https://rest.runpod.io/v1/endpoints/{endpointId}?includeWorkers=true
```

## 9. RunPod 롤아웃 확인

템플릿 업데이트 직후에는 워커 이미지가 섞여 보일 수 있습니다.

예시:

```json
{
  "version": 5,
  "workerImages": [
    "byoungj/sd:v58",
    "byoungj/sd:v58",
    "byoungj/sd:v61"
  ]
}
```

이 상태는 정상입니다. 기존 워커가 drain 되고 새 워커가 붙는 과정입니다.

확인 방법:

```bash
set -a
source .env
set +a
python3 - <<'PY'
import os, requests, json
endpoint = os.environ["RUNPOD_ENDPOINT_ID"]
key = os.environ["RUNPOD_API_KEY"]
r = requests.get(
    f"https://rest.runpod.io/v1/endpoints/{endpoint}?includeWorkers=true",
    headers={"Authorization": f"Bearer {key}"},
    timeout=60,
)
r.raise_for_status()
obj = r.json()
print(json.dumps({
    "version": obj.get("version"),
    "workerImages": [w.get("imageName") for w in obj.get("workers", [])],
}, indent=2))
PY
```

추가 확인:

- 헬스체크를 여러 번 보내서 `workerId`가 새 이미지 워커에 매핑되는지 확인

## 10. 최종 테스트

### 헬스체크

```bash
set -a
source .env
set +a
python3 test_runpod.py --health-check
```

### 샘플 생성

```bash
set -a
source .env
set +a
python3 test_runpod.py \
  --image images/1234.jpeg \
  --hairstyle "short chin-length bob cut, hush cut" \
  --color "ash beige" \
  --top-k 1 \
  --filename-prefix postdeploy_check
```

확인 포인트:

- 요청이 `COMPLETED`
- 결과 이미지 저장 성공
- 가능하면 새 이미지 워커가 요청을 처리했는지 확인

## 11. Docker MCP / Docker Hub MCP를 사람이 수동 설정하는 방법

이 부분은 필수는 아니지만, Codex나 MCP 클라이언트에서 Docker Hub를 같이 쓰고 싶을 때 필요합니다.

### Docker MCP 활성 서버 확인

```bash
docker mcp server ls
```

### Docker Hub MCP 활성화

```bash
docker mcp server enable dockerhub
```

### Docker Hub 사용자명 설정

`~/.docker/mcp/config.yaml`

```yaml
dockerhub:
  username: byoungj
```

### Docker Hub PAT secret 저장

예시:

```bash
cat dockerhub_pat.txt | docker mcp secret set dockerhub.pat_token
```

확인:

```bash
docker mcp server ls
```

기대 상태:

- `dockerhub`
- `SECRETS: done`
- `CONFIG: done`

## 12. 이번 작업에서 실제 있었던 문제와 대응

### 문제 1. GHA 캐시를 넣자마자 앱 빌드 실패

원인:

- Buildx 드라이버가 `docker`

증상:

```text
Cache export is not supported for the docker driver.
```

대응:

- `driver: docker` 제거
- 기본 `docker-container` 드라이버 사용

### 문제 2. RunPod 템플릿은 바뀌었는데 워커 이미지가 섞여 보임

원인:

- 롤링 교체 중

대응:

- `includeWorkers=true`로 상태 확인
- 헬스체크를 반복해 새 워커가 실제 요청을 처리하는지 확인

### 문제 3. GitHub Actions 앱 런이 오래 `in_progress`로 남음

원인:

- GitHub UI 종료 지연 또는 post-step 정리 지연 가능

대응:

- Docker Hub 태그가 실제로 올라왔는지 직접 확인
- RunPod가 새 이미지로 반응하는지 우선 검증

## 13. 권장 운영 순서 요약

```bash
# 1) 현재 상태 확인
python3 test_runpod.py --health-check

# 2) 코드 수정 / 검증
python3 -m py_compile scripts/update_runpod_serverless.py test_runpod.py

# 3) 커밋 / 푸시
git add ...
git commit -m "..."
git push origin master

# 4) GitHub Actions 감시
gh run list --repo PracLee/hair_swap_model --limit 10
gh run watch <app-run-id> --repo PracLee/hair_swap_model --exit-status
gh run watch <base-run-id> --repo PracLee/hair_swap_model --exit-status

# 5) Docker Hub 태그 확인
# byoungj/sd:vNN, latest 확인

# 6) RunPod 이미지 교체
python3 scripts/update_runpod_serverless.py --image byoungj/sd:vNN --wait

# 7) 최종 테스트
python3 test_runpod.py --health-check
python3 test_runpod.py --image images/1234.jpeg --top-k 1 --filename-prefix postdeploy_check
```

