from __future__ import annotations

import json

import chromadb
from chromadb.utils import embedding_functions

from .paths import CHROMA_TRENDS_DIR, TREND_PROCESSED_DIR, ensure_directories


INPUT_FILE = TREND_PROCESSED_DIR / "final_rag_trends.json"
COLLECTION_NAME = "hair_trends"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_data() -> list[dict]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"입력 파일 형식이 올바르지 않습니다: {INPUT_FILE}")
    return data


def build_collection() -> chromadb.api.models.Collection.Collection:
    ensure_directories()
    data = load_data()
    print(f"총 {len(data)}건의 트렌드 데이터를 로드했습니다.")

    client = chromadb.PersistentClient(path=str(CHROMA_TRENDS_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"기존 '{COLLECTION_NAME}' 컬렉션을 삭제했습니다.")
    except ValueError:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "Hair trend RAG data integrated into final_ai"},
    )

    batch_size = 500
    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for offset, item in enumerate(batch):
            idx = start + offset
            ids.append(f"trend_{idx:04d}")
            documents.append(item.get("search_text", ""))
            metadatas.append(
                {
                    "canonical_name": item.get("canonical_name", ""),
                    "display_title": item.get("display_title", ""),
                    "category": item.get("category", ""),
                    "style_tags": ", ".join(item.get("style_tags", [])),
                    "color_tags": ", ".join(item.get("color_tags", [])),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "year": item.get("year", ""),
                }
            )

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"  [{start + len(batch)}/{len(data)}] 삽입 완료")

    print("\n====== 벡터화 완료! ======")
    print(f"컬렉션: {COLLECTION_NAME} ({collection.count()}건)")
    print(f"저장 경로: {CHROMA_TRENDS_DIR}")
    return collection


def query_test(collection, query_text: str, n_results: int = 5) -> None:
    results = collection.query(query_texts=[query_text], n_results=n_results)
    print(f'\n🔍 검색어: "{query_text}"')
    print("-" * 60)
    for index, (doc_id, metadata, distance) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        del doc_id
        print(f"  [{index}] {metadata['display_title']}")
        print(f"      카테고리: {metadata['category']} | 소스: {metadata['source']}")
        print(f"      거리: {distance:.4f}")
        print()


def main() -> None:
    collection = build_collection()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_TRENDS_DIR))
    collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    query_test(collection, "2026 spring blonde hair trend")
    query_test(collection, "올봄 유행하는 단발 헤어스타일")
    query_test(collection, "celebrity bob haircut")


if __name__ == "__main__":
    main()
