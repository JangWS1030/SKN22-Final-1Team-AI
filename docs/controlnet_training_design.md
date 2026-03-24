# ControlNet Hair Training Design

## 1. 목적

현재 저장소의 생성 경로는 `SD 1.5 Inpainting + ControlNet(Canny) + IP-Adapter(face)` 기반이다.

- 입력: `source image + hairstyle_text + color_text`
- 마스킹: `SegFace custom(hair) + SegFace base(face/cloth) + SAM2`
- 생성: `runwayml/stable-diffusion-inpainting + lllyasviel/control_v11p_sd15_canny`
- 얼굴 정체성 보존: `IP-Adapter face crop`

즉, 이번 학습 설계의 목표는 다음과 같다.

1. 헤어 영역 생성 품질을 높인다.
2. 얼굴 identity와 배경/의상 보존을 강화한다.
3. 짧은/중간/긴 길이, 컬감, 업두, 앞머리 등 헤어 속성 제어성을 높인다.
4. 한국/글로벌 스타일 텍스트를 현재 `data/trend_hairstyles.json`의 스타일 taxonomy와 연결한다.

이 설계는 `현재 추론 구조를 유지한 채` 학습 데이터를 정규화하고, LoRA/adapter 위주로 안정적으로 미세조정하는 것을 기본 전제로 한다.

현재 구현 기준으로 segmentation 역할은 분리되어 있다.

- custom SegFace: hair mask 전용
- base SegFace: face / cloth protect mask 전용
- SAM2: hair 경계 refinement

즉, 학습 데이터 설계에서도 hair supervision과 face/cloth protection supervision을 분리해서 생각하는 것이 맞다.

## 2. 핵심 판단

`siik/FaceSketches-HairStyle40-refined`는 메인 학습셋으로 쓰지 않는다.

- 이유 1: `204`장 규모로 너무 작다.
- 이유 2: class당 `2~10`장으로 불균형하다.
- 이유 3: 현재 파이프라인에 필요한 `mask / identity pair / edit pair`가 없다.
- 이유 4: 이 데이터는 "reference / OOD 점검용"으로 두는 것이 더 맞다.

따라서 역할은 아래처럼 분리한다.

- `train-main`: `CelebA-Dialog HQ`
- `train-aux-control`: `Multi-Modal-CelebA-HQ`
- `train-mask`: `CelebAHairMask-HQ`, `LaPa`
- `train-style-domain`: `K-Hairstyle`
- `eval-reference / OOD`: `siik/FaceSketches-HairStyle40-refined`

## 3. 데이터셋 역할 정의

### 3.1 Main supervised set

`CelebA-Dialog HQ`

- 이유: `identity label + caption/edit request + face mask` 조합이 가장 현재 목표와 맞다.
- 용도:
  - self-reconstruction
  - same-identity edit pair
  - 텍스트 조건 학습
  - identity-preserving evaluation

### 3.2 Auxiliary control set

`Multi-Modal-CelebA-HQ`

- 이유: `mask + sketch + text + transparent bg`를 같이 제공한다.
- 용도:
  - ControlNet auxiliary pretraining
  - hair shape prior 보강
  - 향후 `sketch-control` 실험용 데이터 소스

### 3.3 Hair mask supervision

`CelebAHairMask-HQ`, `LaPa`

- 이유: 헤어 경계/잔머리/귀/의상 경계 분리가 중요하다.
- 용도:
  - custom hair head supervision 향상
  - 생성 후 hair region consistency 평가
  - training mask refinement

추가 메모:

- 현재 파이프라인에서는 face/cloth 마스크를 별도 protect branch처럼 쓰고 있으므로, hair 데이터셋만으로는 충분하지 않다.
- `CelebAMask-HQ`, `LaPa` 계열은 hair뿐 아니라 face/cloth class도 함께 제공하므로 현재 구조와 특히 잘 맞는다.

### 3.4 Style domain boost

`K-Hairstyle`

- 이유: 한국 헤어스타일 도메인 coverage가 크다.
- 주의: face blur가 포함될 수 있으므로 메인 identity preservation supervision으로 직접 쓰지 않는다.
- 용도:
  - hairstyle text normalization
  - style classifier / retrieval bank
  - hair-only crop auxiliary training

### 3.5 OOD / qualitative evaluation

`siik/FaceSketches-HairStyle40-refined`

- 용도:
  - style retrieval validation
  - unseen hairstyle qualitative eval
  - prompt/style taxonomy sanity check

## 4. 통합 샘플 타입

모든 데이터셋을 아래 3가지 샘플 타입으로 정규화한다.

### 4.1 `recon`

같은 이미지를 hair-masked input으로 넣고 원본을 복원한다.

- 장점: before/after 편집쌍이 없어도 학습 가능
- 효과: hair texture, boundary, face/background preservation 강화
- 사용 데이터: `CelebA-Dialog HQ`, `CelebAHairMask-HQ`, `LaPa`, `FFHQ`, 일부 `K-Hairstyle`

### 4.2 `edit_same_id`

같은 identity의 다른 이미지 두 장을 짝지어 source/target으로 학습한다.

- 장점: "얼굴은 유지하고 헤어만 바꾸는" 목표에 가장 가깝다
- 제약: pose/angle 차이가 너무 크면 학습 품질이 떨어짐
- 사용 데이터: `CelebA-Dialog HQ`

### 4.3 `style_ref`

reference 이미지를 직접 target supervision으로 쓰지 않고, style text / style embedding / retrieval bank용으로 쓴다.

- 장점: unpaired reference도 활용 가능
- 용도: style classifier, prompt enrichment, OOD eval
- 사용 데이터: `K-Hairstyle`, `siik/FaceSketches-HairStyle40-refined`, 원본 `FaceSketches-HairStyle40`

## 5. 통합 manifest 스키마

학습용 표준 포맷은 `jsonl` 한 줄당 한 샘플을 권장한다.

필수 필드:

| field | type | 설명 |
| --- | --- | --- |
| `sample_id` | string | 전역 유니크 ID |
| `sample_type` | string | `recon`, `edit_same_id`, `style_ref` |
| `split` | string | `train`, `val`, `test`, `ood` |
| `source_name` | string | 원본 데이터셋 이름 |
| `identity_id` | string/null | identity가 없으면 `null` |
| `source_image_path` | string | source 원본 이미지 |
| `target_image_path` | string/null | pixel target 이미지 |
| `mask_path` | string | source 기준 hair edit mask |
| `control_image_path` | string | ControlNet condition image (`canny`) |
| `face_crop_path` | string | IP-Adapter face crop |
| `caption_target` | string | 학습용 최종 prompt |
| `caption_negative` | string | negative prompt |
| `hairstyle_id` | string/null | 내부 normalized style id |
| `hairstyle_text` | string | 스타일 설명 |
| `color_text` | string | 색상 설명 |
| `hair_length` | string | `short`, `medium`, `long`, `updo` |
| `hair_texture` | string | `straight`, `wavy`, `curly`, `coily`, `mixed` |
| `bangs` | string | `none`, `curtain`, `full`, `side`, `baby`, `unknown` |
| `quality_score` | float | 0~1 |
| `loss_profile` | string | 어떤 loss weighting을 적용할지 |

선택 필드:

| field | type | 설명 |
| --- | --- | --- |
| `reference_image_path` | string/null | style reference 이미지 |
| `target_mask_path` | string/null | target 기준 hair mask |
| `source_landmarks_path` | string/null | source landmarks |
| `target_landmarks_path` | string/null | target landmarks |
| `pose_distance` | float/null | source-target pose 차이 |
| `style_domain` | string/null | `kr`, `global`, `editorial` 등 |
| `raw_labels` | object | 원본 데이터셋 라벨 보존 |
| `notes` | string/null | 예외사항 기록 |

## 6. 폴더 구조

권장 구조:

```text
dataset_build/
├─ raw/
│  ├─ celeba_dialog_hq/
│  ├─ multimodal_celebahq/
│  ├─ celebahairmask_hq/
│  ├─ lapa/
│  ├─ k_hairstyle/
│  └─ facesketches_refined/
├─ processed/
│  ├─ images/
│  ├─ masks/
│  ├─ controls/
│  │  └─ canny/
│  ├─ face_crops/
│  ├─ landmarks/
│  ├─ captions/
│  └─ pairs/
├─ manifests/
│  ├─ train.jsonl
│  ├─ val.jsonl
│  ├─ test.jsonl
│  └─ ood.jsonl
└─ reports/
   ├─ filtering_summary.json
   └─ pair_mining_summary.json
```

## 7. 전처리 파이프라인

### 7.1 공통 ingest

모든 데이터셋에 대해 아래를 생성한다.

1. 원본 이미지 표준화: RGB, `min_short_side >= 512`
2. face detect / landmarks 추출
3. hair mask 생성: `SegFace/BiSeNet + SAM2 refinement`
4. canny control 생성
5. face crop 생성
6. normalized caption 생성
7. quality metrics 산출

### 7.2 quality filter

기본 필터:

- face count = 1
- `shortest_side >= 512`
- face coverage `0.12 ~ 0.45` 권장
- off-center 과도한 샘플 제거
- hair mask ratio `0.05 ~ 0.45`
- blur / heavy occlusion / text overlay 제거

`siik/FaceSketches-HairStyle40-refined`처럼 이미 경고 플래그가 있는 데이터는 해당 값을 그대로 `raw_labels`에 보존한다.

### 7.3 style taxonomy mapping

원본 데이터셋 클래스/속성을 내부 style taxonomy로 매핑한다.

기준 소스:

- `data/trend_hairstyles.json`
- dataset raw label
- caption/edit request

예시:

- `BobHair`, `Italian Bob`, `Curtain Bob` -> `curtain-bob` 또는 `sleek-lob` 계열
- `Perm`, `wave`, `soft curl` -> `air-perm-layers`, `layered-midi-waves`
- `PixieCut` -> `volumized-pixie`
- `Mullet` -> `modern-mullet-soft`

중요 원칙:

- 원본 라벨은 버리지 않는다.
- normalized label은 학습 prompt와 retrieval에만 사용한다.
- 다대일 매핑이 가능하도록 `raw_labels`를 항상 보존한다.

## 8. pair mining 설계

### 8.1 `recon` 샘플 생성

모든 main-quality 샘플에 대해 생성한다.

- `source_image = target_image`
- `mask = source hair mask` (dilate 포함)
- `control = canny(source_image)`
- `caption_target = normalized hairstyle + color + texture`

이 샘플은 가장 많이 확보한다.

권장 비중:

- 전체 train batch의 `60~70%`

### 8.2 `edit_same_id` 샘플 생성

`CelebA-Dialog HQ`에서 같은 identity 내에서 아래 조건으로 pair를 만든다.

- hairstyle / bangs / color / length 중 최소 1개 이상 차이
- pose distance 임계치 이하
- landmark alignment 품질 양호
- source-target 얼굴 bbox scale 차이 제한

pair 구성:

- `source_image = identity A의 원본`
- `target_image = 같은 identity A의 다른 헤어 이미지`
- `mask = source hair mask`
- `control = canny(source_image)`
- `caption_target = target 헤어 속성`
- `target_mask = target hair mask`

권장 비중:

- 전체 train batch의 `20~30%`

### 8.3 `style_ref` 샘플 생성

직접 pixel target loss는 약하게 두거나 아예 두지 않는다.

활용 방식:

- prompt expansion
- style classifier
- retrieval bank
- validation prompt set

권장 비중:

- 전체 train batch의 `10%` 내외

## 9. 학습 단계

### Stage A. reconstruction-first LoRA

목표:

- hair region 복원력 강화
- 마스크 경계 안정화
- 배경/의상 보존 강화

학습 대상:

- UNet LoRA
- ControlNet LoRA

freeze:

- VAE
- text encoder 전체 또는 후반부만 선택적으로 unfreeze
- IP-Adapter 가중치
- parser / SAM2

데이터:

- `recon` 중심

### Stage B. same-identity edit alignment

목표:

- "얼굴은 그대로, 헤어만 변경" 학습

데이터:

- `edit_same_id` 비중 확대

추가 조건:

- source-target landmark alignment score 기반 샘플링
- hair region loss 강화
- face region preservation loss 강화

### Stage C. style specialization

목표:

- 한국 헤어스타일/세부 길이/질감 prompt 반응 향상

데이터:

- `K-Hairstyle`
- `style_ref`
- `Multi-Modal-CelebA-HQ`

주의:

- 이 단계는 hair-only bias를 주는 용도다.
- face identity를 직접 맞추는 supervised target으로 사용하지 않는다.

## 10. loss 설계

기본 loss:

1. diffusion denoising loss
2. hair region weighted reconstruction loss
3. non-hair preservation loss

추가 loss 권장:

1. face identity preservation loss
2. hair mask consistency loss
3. edge consistency loss
4. text-image similarity auxiliary loss

권장 weighting 방향:

- hair mask 내부 loss weight를 background보다 높게
- face region은 생성 대상이 아니므로 preservation weight를 높게
- short/medium 변환은 long-hair residual suppression term을 높게

## 11. 평가 설계

정량 평가:

- hair mask IoU / Dice
- face identity similarity
- LPIPS on hair region
- CLIP similarity between prompt and generated hair crop
- background preservation score

정성 평가:

- `siik/FaceSketches-HairStyle40-refined` 기반 style prompt suite
- `reference_only`, `holdout` 계열 별도 qualitative sheet
- 한국형 스타일 prompt set (`hush cut`, `air perm`, `c-curl lob`) 별도 관리

평가 split 원칙:

- identity leakage 방지 위해 identity 기반 분리
- `style_ref`는 train에 쓰더라도 `siik refined`는 test/ood 중심으로 유지

## 12. 구현 우선순위

### Priority 1

- unified manifest builder
- hair mask/canny/face crop preprocessing
- `recon` dataset loader
- Stage A LoRA training

### Priority 2

- same-identity pair mining
- `edit_same_id` loader
- identity / face preservation evaluation

### Priority 3

- style retrieval bank
- `K-Hairstyle` taxonomy integration
- sketch auxiliary branch

## 13. 비권장 사항

- `FaceSketches-HairStyle40-refined`를 메인 train set으로 바로 넣는 것
- 전 데이터셋을 동일 가중치로 섞는 것
- full UNet / full ControlNet을 처음부터 전체 미세조정하는 것
- blur face가 포함된 샘플을 identity supervision으로 직접 쓰는 것
- style taxonomy 정규화 없이 raw class만 prompt에 넣는 것

## 14. 이번 저장소 기준 바로 다음 단계

이 저장소에서 실제 구현 순서는 아래가 가장 안전하다.

1. `dataset_build/manifests/train.jsonl` 포맷 확정
2. `CelebA-Dialog HQ` 기준 `recon` 샘플 생성기 구현
3. 현재 파이프라인의 `SegFace/SAM2/Canny/face crop` 로직을 재사용하는 preprocessing 스크립트 추가
4. Stage A용 ControlNet/UNet LoRA 학습 스크립트 추가
5. 이후 `edit_same_id` pair mining 추가

즉, 첫 구현 목표는 "헤어 복원 재구성 학습"이고, 그 다음에 "같은 사람 헤어 편집 학습"으로 올라가는 2단계 접근이 적절하다.
