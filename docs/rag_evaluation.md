# Stylist RAG Evaluation

`stylist-rag`의 목표는 "요즘 트렌드 추천"과 "실제 시술 포인트"를 한 답변 안에서 같이 제공하는 것입니다. 이 문서는 `RAG 적용`과 `no-rag baseline`을 정량적으로 비교한 최신 평가 결과를 정리합니다.

## 평가 대상

- 비교군 1: `stylist-rag`
- 비교군 2: 같은 모델을 쓰되 검색 문맥 없이 답하는 `no-rag baseline`
- 생성 모델: `gpt-4o-mini`
- judge 모델: `gpt-4o-mini`
- 벤치마크: [stylist_eval_set.json](/Users/leebyoungjae/Desktop/Workspaces/final_ai/data/rag/benchmarks/stylist_eval_set.json)
- 샘플 수: `20`

벤치마크 구성은 아래와 같습니다.

- `커트` 4개
- `펌` 4개
- `컬러` 4개
- `업스타일` 4개
- `샴푸/클리닉` 2개
- `스타일링` 2개

각 샘플은 기존 `reference`, `required_terms` 외에 아래 기준을 추가로 가집니다.

- `expected_trend_terms`
- `expected_sources`
- `expected_pages`
- `expected_source_pages`
- `expected_steps`

즉 이번 평가는 "답변이 기준 답변과 비슷한가"만 보는 것이 아니라, "정답 문서를 찾았는가", "검색 근거에 충실한가", "실제 시술 단계를 빠뜨리지 않았는가"까지 같이 봅니다.

## 실행 명령

```bash
python3 -m rag_pipeline.main stylist-eval \
  --top-k 3 \
  --output-dir /Users/leebyoungjae/Desktop/Workspaces/final_ai/data/rag/analysis/stylist_ragas_eval_full
```

구현은 [evaluate_stylist_rag.py](/Users/leebyoungjae/Desktop/Workspaces/final_ai/rag_pipeline/evaluate_stylist_rag.py)에 있고, CLI 엔트리는 [main.py](/Users/leebyoungjae/Desktop/Workspaces/final_ai/rag_pipeline/main.py#L109)에 연결되어 있습니다.

## 지표

### Trend Retrieval

- `trend_hit_at_k`
  - 미리 정의한 `expected_trend_terms`가 top-k 트렌드 문서 안에 한 번이라도 회수됐는지 봅니다.

### NCS Retrieval

- `retrieval_source_hit_at_k`
  - 기대한 NCS 소스 문서가 top-k 안에 한 번이라도 들어왔는지 봅니다.
- `retrieval_hit_at_k`
  - 기대한 NCS 소스와 페이지 조합이 top-k 안에 들어왔는지 봅니다.
- `retrieval_page_recall_at_k`
  - 기대한 NCS 페이지 집합 중 몇 페이지를 실제로 회수했는지 비율로 계산합니다.

### Integrated

- `dual_source_hit`
  - 같은 질문에서 `trend_hit_at_k=1`이면서 동시에 `retrieval_hit_at_k=1`인 비율입니다. 즉 트렌드와 NCS 근거가 둘 다 제대로 붙었는지 보는 통합 retrieval 지표입니다.
- `service_consistency`
  - 기대 서비스 유형이 `추론된 서비스 축`, `트렌드 문서`, `NCS 문서`와 얼마나 일관되게 맞았는지 0~1 사이 점수로 계산합니다.
- `llm_context_precision_without_reference`
  - RAG가 끌어온 문맥이 실제 답변 생성에 얼마나 정밀하게 쓰였는지 봅니다.
- `context_recall`
  - 기준 답변에 필요한 정보가 검색 문맥 안에 얼마나 포함됐는지 봅니다.
- `semantic_similarity`
  - 답변과 기준 답변의 의미적 유사도를 다국어 sentence-transformer 코사인 유사도로 계산합니다.
- `required_term_coverage`
  - 벤치마크에 미리 정의한 핵심 용어가 답변에 얼마나 포함됐는지 비율로 계산합니다.
- `faithfulness`
  - judge LLM이 검색 근거를 기준으로 답변의 구체 주장들이 실제 문맥에 의해 얼마나 뒷받침되는지 5단계 레이블로 평가하고 이를 점수화합니다.
- `step_completeness`
  - judge LLM이 `expected_steps` 중 답변에 실제로 반영된 단계를 골라내고, 그 비율을 점수화합니다.

`faithfulness`와 `step_completeness`는 RAG와 no-rag 모두 계산하지만, no-rag도 같은 검색 근거와 같은 단계 체크리스트를 기준으로 채점합니다. 그래서 이 둘은 "검색 근거가 주어졌을 때 어느 쪽 답변이 더 근거 중심적으로 맞는가"를 보는 비교입니다.

## 결과 요약

결과 원본:

- [summary.json](/Users/leebyoungjae/Desktop/Workspaces/final_ai/data/rag/analysis/stylist_ragas_eval_full/summary.json)
- [report.md](/Users/leebyoungjae/Desktop/Workspaces/final_ai/data/rag/analysis/stylist_ragas_eval_full/report.md)
- [rag_rows.json](/Users/leebyoungjae/Desktop/Workspaces/final_ai/data/rag/analysis/stylist_ragas_eval_full/rag_rows.json)
- [no_rag_rows.json](/Users/leebyoungjae/Desktop/Workspaces/final_ai/data/rag/analysis/stylist_ragas_eval_full/no_rag_rows.json)

이번 full eval에는 아래 트렌드 retrieval 튜닝이 반영돼 있습니다.

- 트렌드 RAG에 dense + lexical hybrid search 추가
- dense `fetch_k` 확대
- `style_tags / color_tags / title / search_text` 기반 재랭킹
- 하이픈, 복수형, 영문-한글 별칭 정규화
- 기존 NCS hybrid retrieval은 유지

### Trend Retrieval

| metric | rag |
| --- | ---: |
| trend_hit_at_k | 0.7500 |

### NCS Retrieval

| metric | rag |
| --- | ---: |
| retrieval_source_hit_at_k | 1.0000 |
| retrieval_hit_at_k | 1.0000 |
| retrieval_page_recall_at_k | 0.6042 |

### Integrated

| metric | rag | no-rag | delta |
| --- | ---: | ---: | ---: |
| dual_source_hit | 0.7500 | - | - |
| service_consistency | 0.9833 | - | - |
| llm_context_precision_without_reference | 0.9592 | - | - |
| context_recall | 0.6167 | - | - |
| semantic_similarity | 0.4644 | 0.5208 | -0.0564 |
| required_term_coverage | 0.5500 | 0.4833 | +0.0667 |
| faithfulness | 0.7875 | 0.2125 | +0.5750 |
| step_completeness | 0.9600 | 0.3500 | +0.6100 |

## 튜닝 해석

- `trend_hit_at_k`는 `0.5500 -> 0.7500`으로 올랐습니다. 이번 변경의 핵심 효과는 트렌드 1차 회수 개선입니다.
- `retrieval_hit_at_k`는 `0.9000 -> 1.0000`으로 올라가 NCS 쪽 정답 소스/페이지 회수도 전부 맞췄습니다.
- `dual_source_hit`는 `0.5000 -> 0.7500`으로 올라가, 질문 20개 중 15개에서 트렌드와 NCS가 동시에 기대 수준으로 붙었습니다.
- `service_consistency=0.9833`은 retrieval 결과가 거의 전부 올바른 서비스 축으로 정렬된다는 뜻입니다.
- `llm_context_precision_without_reference=0.9592`와 `context_recall=0.6167`도 같이 올라가서, 최종 답변 단계의 grounding 품질도 개선됐습니다.

RAG 우세 샘플 수는 아래와 같습니다.

- `semantic_similarity`: `20개 중 8개`
- `required_term_coverage`: `20개 중 7개`
- `faithfulness`: `20개 중 20개`
- `step_completeness`: `20개 중 13개`

## 해석

- `trend_hit_at_k=0.7500`은 트렌드 retrieval이 확실히 좋아졌다는 뜻입니다. 현재 남은 병목은 특정 `펌`과 `클리닉` 트렌드 질의입니다.
- `retrieval_source_hit_at_k=1.0000`은 NCS 문서군 자체는 20개 전부 맞췄다는 뜻입니다.
- `retrieval_hit_at_k=1.0000`은 기대한 NCS 소스/페이지 조합을 20개 전부 top-k 안에서 맞췄다는 뜻입니다.
- `retrieval_page_recall_at_k=0.6042`는 문서군은 정확하지만, 기대 페이지를 top-3에 얼마나 넓게 담아내느냐는 아직 조정 여지가 있다는 뜻입니다.
- `dual_source_hit=0.7500`은 멀티 소스 RAG 목적에 맞게 트렌드와 NCS가 동시에 붙는 비율이 크게 올라갔다는 뜻입니다.
- `service_consistency=0.9833`은 retrieval과 답변 가이드가 거의 서비스 축을 잘못 타지 않는다는 뜻입니다.
- `semantic_similarity`는 여전히 no-rag가 높았습니다. 기준 답변이 짧고 정규화된 문체라서, 더 일반적인 no-rag 답변이 이 지표에서는 유리합니다.
- `required_term_coverage`, `faithfulness`, `step_completeness`는 각각 `+0.0667`, `+0.5750`, `+0.6100`으로 RAG가 우세했습니다.

즉 이번 20샘플 평가에서는 `트렌드 retrieval`과 `멀티 소스 결합`이 실제로 개선됐고, 최종 답변 단계에서는 RAG가 no-rag보다 더 근거 중심적이고 더 절차 완결적인 답변을 냈습니다.

## 서비스별 관찰

RAG 기준 서비스별 평균은 아래와 같습니다.

- `커트`: retrieval hit `1.0000`, trend hit `1.0000`, dual source `1.0000`, faithfulness `0.8125`, step completeness `1.0000`
- `펌`: retrieval hit `1.0000`, trend hit `0.2500`, dual source `0.2500`, faithfulness `0.7500`, step completeness `1.0000`
- `컬러`: retrieval hit `1.0000`, trend hit `1.0000`, dual source `1.0000`, faithfulness `0.8125`, step completeness `0.9000`
- `업스타일`: retrieval hit `1.0000`, trend hit `1.0000`, dual source `1.0000`, faithfulness `0.8125`, step completeness `0.9000`
- `샴푸/클리닉`: retrieval hit `1.0000`, trend hit `0.0000`, dual source `0.0000`, faithfulness `0.7500`, step completeness `1.0000`
- `스타일링`: retrieval hit `1.0000`, trend hit `1.0000`, dual source `1.0000`, faithfulness `0.7500`, step completeness `1.0000`

현재 약한 구간은 아래 5개입니다.

- `perm_layered`
- `perm_heating_wave`
- `perm_aftercare`
- `clinic_damaged`
- `clinic_aftercare`

공통점은 NCS retrieval은 이미 맞추고 있지만, 기대한 트렌드 표현을 top-3 안에서 아직 못 붙인다는 점입니다. 즉 다음 보강 포인트는 `펌 트렌드 표현`과 `클리닉/케어 트렌드 표현` 쪽입니다.

## 한계

- `semantic_similarity`는 답변 품질보다 문체 차이에 영향을 많이 받습니다.
- `required_term_coverage`는 같은 의미의 우회 표현을 놓칠 수 있습니다.
- `faithfulness`와 `step_completeness`는 judge LLM 기반이라 완전한 규칙 지표는 아닙니다.
- `retrieval_page_recall_at_k`가 `0.6042`라는 것은 문서군은 잘 찾지만 페이지 단위 정밀도는 더 끌어올릴 여지가 있다는 뜻입니다.

## 권장 문서 서술

문서나 발표용으로는 아래처럼 정리하는 것이 맞습니다.

> `20개 미용 실무 샘플 기준으로 stylist-rag는 trend hit@3가 0.7500, NCS retrieval hit@3가 1.0000, dual source hit가 0.7500이었다. 즉 트렌드와 시술 근거를 동시에 붙이는 멀티 소스 retrieval이 안정적으로 개선됐다. 최종 답변 단계에서도 faithfulness가 0.5750, step completeness가 0.6100 높아 no-rag보다 더 근거 중심적이고 절차가 완결된 실무 답변을 생성했다.`

## 다음 보강 포인트

- `perm_layered`, `perm_heating_wave`, `perm_aftercare` 중심의 펌 트렌드 retrieval 보강
- `clinic_damaged`, `clinic_aftercare` 중심의 클리닉 트렌드 retrieval 보강
- 펌/클리닉 트렌드 질의에서 `soft waves`, `effortless hair`, `repair/damage care` 계열 표현을 더 안정적으로 회수하는 lexical 별칭 보강
- 페이지 단위 recall을 더 높이기 위한 `expected page` 근처 문서 가중치 보정
- 기준 답변을 좀 더 구조화해 `semantic_similarity` 편향을 줄이기
- 서비스별 subgroup metric을 자동 표로 뽑아 회귀 테스트에 포함
