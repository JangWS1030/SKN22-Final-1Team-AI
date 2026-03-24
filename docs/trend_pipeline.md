# Trend Pipeline Integration

`final_ai`를 메인 저장소로 유지한 채, 상위 폴더의 `crawling_git` 기능을 충돌 없이 흡수한 구조입니다.

## 배치 원칙

- 생성/배포 서비스 경로는 그대로 유지합니다.
- 크롤링/RAG는 `trend_pipeline/` 패키지로 분리했습니다.
- 원본/가공 데이터는 `data/trend_pipeline/` 아래로 분리해 기존 `data/` 파일과 충돌하지 않게 했습니다.
- ChromaDB와 분석 산출물은 생성물이라 git에서 제외했습니다.

## 디렉터리

```text
final_ai/
├── trend_pipeline/
│   ├── main.py
│   ├── pipeline.py
│   ├── universal_crawler.py
│   ├── data_refiner.py
│   ├── llm_refiner.py
│   ├── vectorize_chromadb.py
│   ├── rag_query.py
│   └── analyze_trends.py
└── data/trend_pipeline/
    ├── raw/
    ├── processed/
    ├── chromadb/
    └── analysis/
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
python -m trend_pipeline.main crawl
```

크롤링 결과 정제:

```bash
python -m trend_pipeline.main refine
```

LLM 정제:

```bash
python -m trend_pipeline.main llm-refine
```

벡터 DB 생성:

```bash
python -m trend_pipeline.main vectorize
```

RAG 질의:

```bash
python -m trend_pipeline.main rag --query "올봄 유행하는 단발 추천해줘"
```

전체 파이프라인:

```bash
python -m trend_pipeline.main full --with-llm --with-vectorize
```

키워드 분석:

```bash
python -m trend_pipeline.main analyze
```

## 참고

- 기존 `crawling_git/src/pipeline.py`는 누락된 모듈을 import하고 있어 그대로 가져오지 않았습니다.
- 대신 현재 저장소 구조에 맞는 `trend_pipeline/pipeline.py`로 재구성했습니다.
- 이전에 수집해 둔 JSON 결과물은 `data/trend_pipeline/raw` 및 `data/trend_pipeline/processed`로 옮겨 두었습니다.
