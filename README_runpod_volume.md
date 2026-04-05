# RunPod Cold Start (모델 다운로드 시간) 해결 가이드

현재 `Dockerfile.sd` 방식을 보면, 빌드 시간과 Docker 이미지 크기(~6GB)를 줄이기 위해 거대한 AI 모델 파일들(약 10~15GB)을 이미지에 넣지 않고 RunPod 서버가 처음 켜질 때(Cold Start) `huggingface_hub`를 통해 실시간으로 다운로드받도록 설정되어 있습니다.
이 과정 때문에 서버를 켤 때마다 20~30분씩 다운로드 시간이 발생하는 것입니다.

이 문제를 우회하고 **Cold Start를 1~2분 내로 끝내려면**, RunPod의 **Network Volume** 기능을 사용해서 모델 캐시 파일을 영구적으로 저장하고 재사용해야 합니다.

## 중요 경고

- 배포나 endpoint 재생성 과정에서 `networkVolumeId`를 비우거나 제거하지 마세요.
- template의 환경 변수에서 `HF_HOME`, `TORCH_HOME`를 지우지 마세요.
- startup preload를 유지하려면 `MIRRAI_PRELOAD_ON_STARTUP=1`도 같이 보존해야 합니다.
- 위 3개 중 하나라도 빠지면 새 worker가 뜰 때마다 Hugging Face 모델을 다시 내려받아 20분 이상 콜드스타트가 발생할 수 있습니다.

현재 운영 기준으로 유지해야 하는 값:

- Network Volume: `h6lcfsxdt0` (`mirrai-model-cache`)
- `HF_HOME=/runpod-volume/huggingface`
- `TORCH_HOME=/runpod-volume/torch`
- `MIRRAI_PRELOAD_ON_STARTUP=1`

## 🛠 해결 방법: Network Volume 마운트하기

RunPod에서 Pod을 생성할 때 아래 단계를 따라주세요.

### 1단계: Network Volume 생성
1. RunPod 대시보드 좌측 메뉴에서 **Storage** 클릭
2. **+ Network Volume** 버튼 클릭
3. 이름은 `mirrai-models` 등으로 짓고, 데이터 센터 위치를 팟을 띄울 곳과 동일하게 맞춘 뒤 생성 (용량은 20GB 정도 추천)

### 2단계: Serverless Endpoint 생성 시 마운트 설정
1. RunPod의 **Serverless** 탭에서 **New Endpoint**를 클릭합니다.
2. 템플릿(본인의 `byoungj/sd-v1:latest` 등)을 선택한 뒤, 화면 하단의 **Advanced** 메뉴를 펼칩니다.
3. 그 안의 **Network Volumes** 설정에서 아까 만든 `mirrai-models` 볼륨을 선택합니다.
   (*참고: 최근 RunPod Serverless에서는 Network Volume을 선택하면 컨테이너 내부의 `/runpod-volume` 이라는 경로로 **자동 고정 마운트**됩니다.*)
4. 동일한 Advanced 메뉴 내의 **Environment Variables (환경 변수)** 영역에 다음 2개 변수를 **반드시** 추가해주세요 (모델 저장 경로를 `/runpod-volume`쪽으로 인식하게 하는 명령입니다):
   - Key: `HF_HOME` / Value: `/runpod-volume/huggingface`
   - Key: `TORCH_HOME` / Value: `/runpod-volume/torch`
5. preload를 startup 시점에 수행하려면 아래 변수도 함께 넣습니다:
   - Key: `MIRRAI_PRELOAD_ON_STARTUP` / Value: `1`
6. 저장하고 Endpoint 파드를 생성(Deploy)합니다.

### 배포 후 확인 항목

- endpoint에 `networkVolumeId`가 실제로 남아 있는지 확인합니다.
- template env에 `HF_HOME`, `TORCH_HOME`, `MIRRAI_PRELOAD_ON_STARTUP`가 그대로 남아 있는지 확인합니다.
- 위 값이 빠졌다면 worker 교체 전에 먼저 복구해야 합니다.

### 👉 원리
위와 같이 설정하면 첫 번째 호출 시 약 20분 걸려서 모델을 다운로드받지만, 그 파일들이 RunPod의 공용 외장하드(`/runpod-volume`)에 차곡차곡 영구 저장됩니다.
이후에 파드가 슬립(Scale to Zero)되었다가 다시 깨어날 때, 해당 볼륨을 즉시 연결하여 저장된 모델을 인식하므로 1~2초만에 다운로드 구간을 통과하게 됩니다!
