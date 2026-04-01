# Google Docs 수정 지시서

아래 내용을 Claude에 그대로 전달해서 Google Docs 문서를 수정하게 하면 된다.

## 작업 목표

LLM 모델링 및 평가 문서에 `5-1. 프롬프트 최적화` 소제목과 프롬프트 코드 예시를 추가한다.

## 수정 위치

- `5. 성능 개선을 위해 적용한 기술` 섹션의 마지막 문단 바로 뒤
- `6. 개선 전후 성능 비교` 섹션 바로 앞

즉, `5번 섹션 끝`과 `6번 섹션 시작` 사이에 아래 내용을 새로 삽입한다.

## 삽입할 제목

`5-1. 프롬프트 최적화`

## 삽입할 본문

본 프로젝트는 검색된 트렌드 자료와 NCS 시술 자료를 하나의 프롬프트로 통합하는 방식으로 RAG 응답을 생성하였다. 이 과정에서 질문, 서비스 힌트, 스타일 키워드, 트렌드 자료, 시술 자료를 구조화하여 LLM에 전달함으로써 문맥 손실을 줄이고, 답변이 검색 근거에 기반하도록 유도하였다. 특히 트렌드 자료와 시술 자료를 분리된 블록으로 제공하여 최신 스타일 추천과 실제 시술 절차가 혼합되지 않도록 설계하였다.

다음은 실제 시스템에서 사용한 프롬프트 구성 코드의 핵심 예시이다.

```python
def build_stylist_user_prompt(query: str, bundle: dict) -> str:
    return (
        f"[질문]\n{query}\n\n"
        f"[답변 형식 가이드]\n{build_response_guide(bundle['service_types'])}\n\n"
        f"[추론된 시술 힌트]\n{', '.join(bundle['service_types']) or '없음'}\n\n"
        f"[추론된 스타일 키워드]\n{', '.join(bundle['style_keywords']) or '없음'}\n\n"
        f"[NCS 검색어]\n{bundle['ncs_query']}\n\n"
        f"[트렌드 자료]\n{_build_trend_context(bundle['trend_docs']) or '없음'}\n\n"
        f"[시술 자료]\n{_build_ncs_context(bundle['ncs_docs']) or '없음'}"
    )
```

위와 같이 질문과 검색 결과를 역할별로 구분해 프롬프트를 구성함으로써, 모델이 최신 헤어 트렌드와 실제 시술 절차를 함께 고려하면서도 각 정보의 출처를 구분하여 답변하도록 설계하였다. 이를 통해 근거 기반 답변의 일관성을 높이고, 시술 단계의 누락이나 불필요한 추정을 줄일 수 있었다.

## Claude에게 같이 줄 설명

아래 요구사항을 같이 전달하면 된다.

1. 기존 문단 구조와 번호 체계를 유지할 것
2. `5. 성능 개선을 위해 적용한 기술`과 `6. 개선 전후 성능 비교` 사이에만 새 내용을 삽입할 것
3. 코드 블록은 반드시 `python` 코드 블록으로 넣을 것
4. 기존 점수 표나 수치는 수정하지 말 것

## 코드 출처

- 시스템 프롬프트: `/Users/leebyoungjae/Desktop/Workspaces/final_ai/rag_pipeline/stylist_rag_query.py`
- 코드 위치: `build_stylist_user_prompt`
