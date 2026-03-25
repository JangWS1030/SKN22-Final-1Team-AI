# RAG Pipeline Integration

`final_ai`를 메인 저장소로 유지한 채, 상위 폴더의 `crawling_git` 기능을 충돌 없이 흡수한 구조입니다.

## 배치 원칙

- 생성/배포 서비스 경로는 그대로 유지합니다.
- 크롤링/RAG 코드는 `rag_pipeline/` 패키지로 분리했습니다.
- 원본 기사, 정제 결과, NCS PDF, 벤치마크, 벡터 스토어는 `data/rag/` 아래로 통합했습니다.
- ChromaDB와 분석 산출물은 생성물이라 git에서 제외했습니다.

## 디렉터리

```text
final_ai/
├── rag_pipeline/
│   ├── main.py
│   ├── pipeline.py
│   ├── ncs_pdf_ingest.py
│   ├── ncs_llm_refiner.py
│   ├── ncs_vectorize_chromadb.py
│   ├── ncs_rag_query.py
│   ├── stylist_rag_query.py
│   ├── universal_crawler.py
│   ├── data_refiner.py
│   ├── llm_refiner.py
│   ├── vectorize_chromadb.py
│   ├── rag_query.py
│   └── analyze_trends.py
└── data/rag/
    ├── sources/
    │   └── ncs/
    ├── raw/
    │   └── trends/
    ├── processed/
    │   ├── trends/
    │   └── ncs/
    ├── benchmarks/
    ├── analysis/
    └── stores/
        ├── chromadb_trends/
        └── chromadb_ncs/
```

## 설치

```bash
python -m pip install -r requirements-trends.txt
python -m playwright install chromium
```

`.env`에는 아래 키를 사용합니다.

```env
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

## 실행 예시

크롤링:

```bash
python -m rag_pipeline.main crawl
```

크롤링 결과 정제:

```bash
python -m rag_pipeline.main refine
```

LLM 정제:

```bash
python -m rag_pipeline.main llm-refine
```

벡터 DB 생성:

```bash
python -m rag_pipeline.main vectorize
```

트렌드 RAG 질의:

```bash
python -m rag_pipeline.main rag --query "올봄 유행하는 단발 추천해줘"
```

NCS PDF 전처리:

```bash
python -m rag_pipeline.main ncs-extract --front-matter-pages 13
```

NCS PDF Gemini 정제:

```bash
python -m rag_pipeline.main ncs-llm-refine --delay-seconds 0.6
```

NCS 벡터 DB 생성:

```bash
python -m rag_pipeline.main ncs-vectorize
```

NCS RAG 질의:

```bash
python -m rag_pipeline.main ncs-rag --query "손상모 클리닉은 어떻게 진행해?"
```

미용사용 통합 RAG 질의:

```bash
python -m rag_pipeline.main stylist-rag --query "요즘 유행하는 단발 추천하고 그 스타일 시술 포인트도 정리해줘"
```

Stylist RAG 평가:

```bash
python -m rag_pipeline.main stylist-eval --top-k 3
```

전체 파이프라인:

```bash
python -m rag_pipeline.main full --with-llm --with-vectorize
```

키워드 분석:

```bash
python -m rag_pipeline.main analyze
```

## 참고

- 기존 `crawling_git/src/pipeline.py`는 누락된 모듈을 import하고 있어 그대로 가져오지 않았습니다.
- 대신 현재 저장소 구조에 맞는 [pipeline.py](/Users/leebyoungjae/Desktop/Workspaces/final_ai/rag_pipeline/pipeline.py)로 재구성했습니다.
- 기존에 흩어져 있던 기사 원본, 정제 결과, NCS PDF, 벤치마크, 분석 결과를 `data/rag/` 하나로 묶었습니다.
- ChromaDB는 물리적으로 `data/rag/stores/chromadb_trends`, `data/rag/stores/chromadb_ncs`로 분리했습니다.
- `stylist-rag`는 트렌드 RAG와 NCS RAG를 같이 조회해서 "요즘 스타일 + 실제 시술 포인트"를 한 답변으로 합성합니다.
- `stylist-rag`는 서비스 유형을 추론해서 `커트/펌/컬러/업스타일/샴푸·클리닉`별 답변 템플릿으로 출력합니다.
- `rag`와 `stylist-rag`의 트렌드 retrieval은 현재 dense + lexical hybrid search를 사용하고, `fetch_k`를 넓힌 뒤 `style_tags / color_tags / title / search_text` 기준으로 재랭킹합니다.
- 트렌드 retrieval에는 하이픈, 복수형, 영문-한글 별칭 정규화가 들어가 있어 `bob/단발`, `ash-brown/애쉬 브라운`, `french bob/프렌치 밥` 같은 표현 차이를 흡수합니다.
- `ncs-rag`와 `stylist-rag`의 NCS retrieval은 현재 dense + lexical hybrid search를 사용하고, dense fetch 후보를 넓힌 뒤 동일 소스 인접 페이지를 보강합니다.
- 규칙 기반 query expansion도 실험했지만 retrieval 회수가 악화돼 현재 기본 경로에서는 비활성 상태로 유지합니다.
- `stylist-eval`은 현재 20개 벤치마크 샘플 기준으로 `stylist-rag`와 no-rag baseline을 비교해 `Trend / NCS / Integrated` 3개 표 형태로 결과를 저장합니다.
- 벤치마크는 `expected_trend_terms`, `expected_source_pages`, `expected_steps`를 함께 가져가서 `trend_hit@k`, `retrieval_hit@k`, `dual_source_hit`, `service_consistency`까지 계산합니다.
- 평가 방법과 최신 결과 해석은 [rag_evaluation.md](/Users/leebyoungjae/Desktop/Workspaces/final_ai/docs/rag_evaluation.md)를 기준으로 봅니다.
