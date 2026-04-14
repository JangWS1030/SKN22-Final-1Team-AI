# PR Title

RunPod 설문 기반 성별 분기 프롬프트 보정 및 남성 bob 계열 drift 차단

# Summary

이번 PR은 RunPod 추천/시뮬레이션 서비스가 구조화된 설문 데이터를 우선적으로 해석하도록 수정하고, `male` / `female` 프롬프트 생성을 분리해 남성 입력이 `bob`, `lob`, `mini bob`, `c-curl bob` 같은 여성 코딩 용어로 drift 하는 문제를 해결합니다.

핵심 목표는 아래 4가지입니다.

- `survey_data` 기반 canonical field를 legacy text보다 우선 사용
- `survey_profile.gender_branch`를 1차 분기 기준으로 사용
- 남성 branch에서 남성형 헤어 vocabulary로만 내부 프롬프트를 구성
- 기존 legacy caller가 깨지지 않도록 fallback 유지

# Problem

현재 서버는 정규화된 설문 데이터를 source of truth로 보내고 있지만, RunPod 쪽은 여전히 `hairstyle_text` 중심으로 동작하는 경로가 남아 있었습니다.

그 결과 아래와 같은 male structured 입력이 들어와도 내부 프롬프트가 여성형 bob vocabulary로 흘러갈 수 있었습니다.

- `target_length=short`
- `target_vibe=natural | chic`
- `scalp_type=curly`
- `style_axes.two_block=soft`
- `style_axes.front_styling=down`
- `style_axes.parting=non_parted`

예상 해석:

- `male short soft two-block with down styling, non-parted front, curly texture`

기존 drift 예:

- `sleek mini bob`
- `lob`
- `feminine short bob`

# What Changed

## 1. Request parsing

[handler_sd.py](/Users/leebyoungjae/Desktop/Workspaces/final_ai/handler_sd.py)

- `survey_data` / `survey_profile` / canonical field 파싱 추가
- `survey_profile.gender_branch`를 우선 gender branch로 사용
- structured field와 legacy field가 같이 오면 structured를 우선 적용
- legacy fallback 입력은 계속 허용
  - `hairstyle_text`
  - `color_text`
  - `preference`
  - `preference_text`
- 응답 메타데이터 `request_resolution` 추가
  - resolved gender branch
  - resolved canonical preferences
  - blocked vocabulary
  - fallback mode
  - structured payload used 여부

## 2. Prompt branching

[pipeline_sd_components/prompt.py](/Users/leebyoungjae/Desktop/Workspaces/final_ai/pipeline_sd_components/prompt.py)

- structured prompt context 정규화 로직 추가
- canonical `target_length` 기반 hair length 해석 추가
- 남성/여성 structured prompt builder 분리
- male branch vocabulary를 아래처럼 고정
  - short crop
  - clean short
  - natural two-block
  - soft two-block
  - down style
  - non-parted front
  - curly texture
  - soft down perm finish
  - clean side line
- male branch negative/block vocabulary 추가
  - bob
  - lob
  - mini bob
  - c-curl bob
  - feminine bob silhouette
  - female-coded framing terms
- 기존 short male branch에서 공통 positive suffix로 다시 `strict short bob silhouette`가 붙던 문제 제거
- `sd_prompt_data`가 같이 와도 structured payload가 있으면 structured mapping이 우선되도록 수정

## 3. Pipeline propagation

[pipeline_sd_inpainting.py](/Users/leebyoungjae/Desktop/Workspaces/final_ai/pipeline_sd_inpainting.py)

- `prompt_context`를 pipeline run 경로로 전달
- resolved gender branch / canonical preference / final prompt / negative prompt를 debug data와 style metadata에 저장
- request resolution 로그 추가

## 4. Short-tail refinement safeguard

[pipeline_sd_components/refinement.py](/Users/leebyoungjae/Desktop/Workspaces/final_ai/pipeline_sd_components/refinement.py)

- short-tail cleanup용 SD refine prompt도 branch-aware 하도록 수정
- male short cleanup 단계에서 bob 기준 보정 prompt를 쓰지 않도록 변경

## 5. Verification script

[scripts/test_structured_prompt_branching.py](/Users/leebyoungjae/Desktop/Workspaces/final_ai/scripts/test_structured_prompt_branching.py)

- male structured sample이 bob/lob vocabulary로 가지 않는지 검증
- female bob flow가 유지되는지 검증
- legacy fallback이 유지되는지 검증

# Male / Female Mapping

## Male

structured field를 우선 해석해서 남성형 표현으로 내부 스타일을 조합합니다.

예:

- `target_length=short`
- `two_block=soft`
- `front_styling=down`
- `parting=non_parted`
- `scalp_type=curly`
- `target_vibe=natural`

내부 스타일 해석:

- `male short soft two-block haircut`
- `clean side line`
- `down style`
- `non-parted front`
- `curly texture with a soft down perm finish`
- `natural mood`

## Female

기존 female bob / short flow는 유지하되, structured `target_length=bob`가 오면 female bob 흐름이 분명하게 유지되도록 했습니다.

# Before / After

## Input

male structured sample:

- `target_length=short`
- `target_vibe=natural`
- `scalp_type=curly`
- `gender_branch=male`
- `style_axes.two_block=soft`
- `style_axes.front_styling=down`
- `style_axes.parting=non_parted`

## Before

- `short bob`
- `sleek mini bob`
- `lob`
- `feminine short bob`

## After

normalized male style:

```text
male short soft two-block haircut, clean side line, down style, non-parted front, curly texture with a soft down perm finish, natural mood, clean ear contour
```

blocked male vocabulary:

```text
bob, lob, mini bob, c-curl bob, feminine bob silhouette, female-coded framing terms
```

# Backward Compatibility

- `survey_profile`이 없으면 기존 legacy text 흐름으로 폴백합니다.
- `preference` / `preference_text`도 legacy style text로 계속 반영됩니다.
- structured payload가 없으면 기존 `hairstyle_text`, `color_text`, `subject_gender`, `sd_prompt_data` 중심 동작을 유지합니다.
- female bob flow도 유지됩니다.

# Validation

로컬 검증:

```bash
python -m py_compile handler_sd.py pipeline_sd_inpainting.py pipeline_sd_components/prompt.py pipeline_sd_components/refinement.py scripts/test_structured_prompt_branching.py
python scripts/test_structured_prompt_branching.py
```

검증 항목:

- male structured sample에서 `bob/lob/mini bob/c-curl bob`가 positive prompt에 들어가지 않음
- male negative prompt에 blocked vocabulary가 포함됨
- female structured bob flow 정상 유지
- legacy `preference_text` fallback 정상 유지

# Deployment Notes

- commit: `b3c030b`
- GitHub Actions build: success
- build tag: `v159`
- RunPod template image: `byoungj/sd:v159`
- 요청에 따라 RunPod health check 기반 최종 테스트는 생략했고, 실운영 테스트는 별도 진행 예정입니다.

# Risk

- male branch의 표현을 더 강하게 제한했기 때문에, 남성 사용자가 명시적으로 bob 계열을 원하는 예외 케이스는 structured 질문 설계나 별도 explicit override 정책이 필요할 수 있습니다.
- 현재는 structured payload가 있으면 structured를 우선 사용하므로, legacy text에 남아 있는 여성형 단어는 기본적으로 무시되는 방향입니다.
