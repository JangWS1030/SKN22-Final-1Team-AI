from __future__ import annotations

import json

import chromadb
from chromadb.utils import embedding_functions

from .paths import CHROMA_NCS_DIR, NCS_PROCESSED_DIR, ensure_directories


INPUT_FILE = NCS_PROCESSED_DIR / "ncs_rag_ready.json"
COLLECTION_NAME = "hair_ncs_manuals"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_data() -> list[dict]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"입력 파일 형식이 올바르지 않습니다: {INPUT_FILE}")
    return data


def build_ncs_collection() -> chromadb.api.models.Collection.Collection:
    ensure_directories()
    data = load_data()
    print(f"총 {len(data)}건의 NCS 데이터를 로드했습니다.")

    client = chromadb.PersistentClient(path=str(CHROMA_NCS_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"기존 '{COLLECTION_NAME}' 컬렉션을 삭제했습니다.")
    except ValueError:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "NCS hairstyle manual RAG data"},
    )

    batch_size = 200
    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for offset, item in enumerate(batch):
            idx = start + offset
            source_id = str(item.get("source_id", "")).strip()
            ids.append(f"ncs_{idx:05d}")
            documents.append(item.get("search_text", ""))
            metadatas.append(
                {
                    "source_id": source_id,
                    "canonical_name": item.get("canonical_name", ""),
                    "display_title": item.get("display_title", ""),
                    "category": item.get("category", ""),
                    "service_type": ", ".join(item.get("service_type", [])),
                    "target_conditions": ", ".join(item.get("target_conditions", [])),
                    "tools": ", ".join(item.get("tools", [])),
                    "steps": " | ".join(item.get("steps", [])),
                    "cautions": " | ".join(item.get("cautions", [])),
                    "summary": item.get("summary", ""),
                    "stylist_answer": item.get("stylist_answer", ""),
                    "source_document_name": item.get("source_document_name", ""),
                    "source_page": str(item.get("source_page", "")),
                    "source": item.get("source", ""),
                }
            )

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"  [{start + len(batch)}/{len(data)}] 삽입 완료")

    print("\n====== NCS 벡터화 완료! ======")
    print(f"컬렉션: {COLLECTION_NAME} ({collection.count()}건)")
    print(f"저장 경로: {CHROMA_NCS_DIR}")
    return collection


def query_test(collection, query_text: str, n_results: int = 5) -> None:
    results = collection.query(query_texts=[query_text], n_results=n_results)
    print(f'\n🔍 검색어: "{query_text}"')
    print("-" * 60)
    for index, metadata in enumerate(results["metadatas"][0], start=1):
        distance = results["distances"][0][index - 1]
        print(f"  [{index}] {metadata['display_title']}")
        print(f"      카테고리: {metadata['category']} | 서비스: {metadata['service_type']}")
        print(f"      문서: {metadata['source_document_name']} p.{metadata['source_page']}")
        print(f"      거리: {distance:.4f}")
        print()


def main() -> None:
    collection = build_ncs_collection()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_NCS_DIR))
    collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    query_test(collection, "다운펌처럼 눌러야 하는 짧은 머리 커트 방법")
    query_test(collection, "손상모 클리닉 어떻게 진행해?")
    query_test(collection, "펌 후 홈케어 안내")


if __name__ == "__main__":
    main()
