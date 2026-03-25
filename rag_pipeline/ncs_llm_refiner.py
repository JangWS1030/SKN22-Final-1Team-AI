from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

from google import genai
from pydantic import BaseModel, Field

from .paths import NCS_PROCESSED_DIR, ensure_directories


class NcsChunkInfo(BaseModel):
    source_id: str = Field(description="입력 청크의 source_id를 그대로 반환")
    is_valid: bool = Field(
        description="미용사 챗봇에 직접 도움이 되는 시술/관리/안전/상담 지식이면 true, 서식/행정/평가/시험/메타 안내면 false"
    )
    canonical_name: str = Field(description="청크의 핵심 주제를 짧게 정규화한 이름")
    category: Literal["consultation", "preparation", "procedure", "finishing", "aftercare", "theory", "safety", "drop"] = Field(
        description="미용사 응답 용도에 맞춘 청크 분류"
    )
    service_type: list[str] = Field(description="컷, 펌, 컬러, 샴푸/클리닉, 업스타일, 가발 등 서비스 유형 태그")
    target_conditions: list[str] = Field(description="대상 모발 상태, 고객 조건, 사용 상황 태그")
    tools: list[str] = Field(description="본문에 실제로 언급된 도구/제품/기기")
    steps: list[str] = Field(description="본문의 핵심 절차를 2~6개 짧은 단계로 정리한 목록")
    cautions: list[str] = Field(description="시술 시 주의할 점")
    summary: str = Field(description="핵심만 요약한 2~3문장 한국어 요약")
    stylist_answer: str = Field(description="미용사가 챗봇 답변으로 바로 활용할 수 있는 자연스러운 한국어 설명")
    search_text: str = Field(description="벡터 검색을 위한 합성 검색 텍스트")


class NcsChunkBatchInfo(BaseModel):
    items: list[NcsChunkInfo] = Field(description="입력 청크별 구조화 결과")


class NcsLlmRefiner:
    def __init__(
        self,
        *,
        model_name: str | None = None,
        input_file: Path | None = None,
        output_file: Path | None = None,
        checkpoint_file: Path | None = None,
    ) -> None:
        ensure_directories()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.model_name = model_name or os.environ.get("NCS_REFINER_MODEL", "gemini-2.5-flash")
        self.input_file = input_file or (NCS_PROCESSED_DIR / "ncs_manual_chunks.json")
        self.output_file = output_file or (NCS_PROCESSED_DIR / "ncs_rag_ready.json")
        self.checkpoint_file = checkpoint_file or (NCS_PROCESSED_DIR / "ncs_rag_ready_checkpoint.json")

    def refine_with_llm(
        self,
        *,
        delay_seconds: float = 0.6,
        save_every: int = 20,
        limit: int | None = None,
        batch_size: int = 5,
    ) -> list[dict]:
        if not self.api_key or self.client is None:
            print("⚠️ GEMINI_API_KEY가 없어 NCS LLM 정제를 건너뜁니다.", flush=True)
            return []

        source_data = self._load_source_data()
        if limit is not None:
            source_data = source_data[:limit]

        checkpoint = self._load_checkpoint()
        print(
            f"====== NCS Gemini 정제 시작: 총 {len(source_data)}건 / 기존 체크포인트 {len(checkpoint)}건 / 모델 {self.model_name} ======",
            flush=True,
        )

        pending_items = [item for item in source_data if str(item["id"]) not in checkpoint]
        processed_since_save = 0
        total = len(source_data)

        for batch_start in range(0, len(pending_items), batch_size):
            batch = pending_items[batch_start : batch_start + batch_size]
            first_item = batch[0]
            last_item = batch[-1]
            first_index = source_data.index(first_item) + 1
            last_index = source_data.index(last_item) + 1
            print(
                f"[{first_index}-{last_index}/{total}] {first_item['display_title']} .. {last_item['display_title']}",
                flush=True,
            )

            results = self._refine_batch(batch, delay_seconds=delay_seconds)
            for result in results:
                source_id = str(result["source_id"])
                checkpoint[source_id] = result
                processed_since_save += 1

                status = "채택" if result.get("is_valid") and result.get("category") != "drop" else "드롭"
                print(f"   {source_id} | {status} | category={result.get('category')} | canonical={result.get('canonical_name')}", flush=True)

            if processed_since_save >= save_every:
                self._save_outputs(source_data, checkpoint)
                processed_since_save = 0

        self._save_outputs(source_data, checkpoint)
        valid_items = self._build_final_items(source_data, checkpoint)
        print(
            f"====== NCS Gemini 정제 완료: 입력 {len(source_data)}건 -> 최종 {len(valid_items)}건 ======",
            flush=True,
        )
        print(f"결과물 저장 경로: {self.output_file}", flush=True)
        print(f"체크포인트 저장 경로: {self.checkpoint_file}", flush=True)
        return valid_items

    def _load_source_data(self) -> list[dict]:
        if not self.input_file.exists():
            raise FileNotFoundError(f"입력 파일이 없습니다: {self.input_file}")

        with self.input_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(f"입력 파일 형식이 올바르지 않습니다: {self.input_file}")
        return data

    def _load_checkpoint(self) -> dict[str, dict]:
        if not self.checkpoint_file.exists():
            return {}

        with self.checkpoint_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            return {}
        return {str(item["source_id"]): item for item in data if isinstance(item, dict) and "source_id" in item}

    def _save_outputs(self, source_data: list[dict], checkpoint: dict[str, dict]) -> None:
        checkpoint_items = self._ordered_checkpoint_items(source_data, checkpoint)
        with self.checkpoint_file.open("w", encoding="utf-8") as file:
            json.dump(checkpoint_items, file, ensure_ascii=False, indent=2)

        final_items = self._build_final_items(source_data, checkpoint)
        with self.output_file.open("w", encoding="utf-8") as file:
            json.dump(final_items, file, ensure_ascii=False, indent=2)

    def _ordered_checkpoint_items(self, source_data: list[dict], checkpoint: dict[str, dict]) -> list[dict]:
        ordered: list[dict] = []
        for item in source_data:
            source_id = str(item["id"])
            if source_id in checkpoint:
                ordered.append(checkpoint[source_id])
        return ordered

    def _build_final_items(self, source_data: list[dict], checkpoint: dict[str, dict]) -> list[dict]:
        valid_items: list[dict] = []
        for item in source_data:
            source_id = str(item["id"])
            refined = checkpoint.get(source_id)
            if not refined:
                continue
            if not refined.get("is_valid") or refined.get("category") == "drop":
                continue
            valid_items.append(refined)
        return valid_items

    def _refine_batch(self, items: list[dict], *, delay_seconds: float) -> list[dict]:
        prompt = self._build_batch_prompt(items)
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=NcsChunkBatchInfo,
                        temperature=0.1,
                    ),
                )

                if getattr(response, "parsed", None) is not None:
                    parsed = response.parsed
                    parsed_items = parsed.items
                    results = [item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in parsed_items]
                else:
                    results = json.loads(response.text)["items"]

                result_map = {str(item["source_id"]): item for item in results}
                normalized_results: list[dict] = []
                missing_items: list[dict] = []
                for source_item in items:
                    source_id = str(source_item["id"])
                    result = result_map.get(source_id)
                    if result is None:
                        missing_items.append(source_item)
                        continue
                    normalized_results.append(self._merge_result(source_item, result))

                if missing_items:
                    print(f"   ! 배치 응답 누락 {len(missing_items)}건, 개별 재시도", flush=True)
                    normalized_results.extend(self._refine_single_items(missing_items, delay_seconds=delay_seconds))

                return normalized_results
            except Exception as exc:
                last_error = exc
                wait_seconds = 2 ** attempt
                print(f"   ! [배치 재시도 {attempt + 1}/3] {exc}", flush=True)
                time.sleep(wait_seconds)

        print("   ! 배치 실패, 개별 재시도로 전환", flush=True)
        return self._refine_single_items(items, delay_seconds=delay_seconds, last_error=last_error)

    def _refine_single_items(
        self,
        items: list[dict],
        *,
        delay_seconds: float,
        last_error: Exception | None = None,
    ) -> list[dict]:
        results: list[dict] = []
        for item in items:
            results.append(self._refine_item(item, delay_seconds=delay_seconds, inherited_error=last_error))
        return results

    def _refine_item(self, item: dict, *, delay_seconds: float, inherited_error: Exception | None = None) -> dict:
        prompt = f"""
당신은 헤어 시술 교육 자료를 미용사 챗봇용 RAG 데이터로 구조화하는 분석가입니다.

목표:
- 미용사가 고객에게 시술 방법, 준비, 마무리, 홈케어를 설명할 때 바로 활용 가능한 지식만 남깁니다.
- 교육행정/시험/평가/동의서/카드 서식/문헌정보/출판정보는 드롭합니다.
- 원문에 근거해 한국어로만 정리합니다.

판정 기준:
1. 직접적인 시술 절차, 준비물, 도구 선택, 관리법, 주의사항이면 유지합니다.
2. 지나치게 추상적인 학습행정 문구, 시험문제, 평가표, 개인정보 서식이면 드롭합니다.
3. steps, cautions, tools는 원문에 근거한 것만 넣고 없으면 빈 배열로 둡니다.
4. summary와 stylist_answer는 과장 없이 실무적으로 씁니다.
5. search_text에는 서비스 유형, 단계, 대상 조건, 도구, 핵심 절차를 함께 녹입니다.

[반드시 지킬 것]
- source_id는 반드시 "{item.get("id", "")}"로 그대로 반환합니다.

[메타데이터]
- 문서명: {item.get("document_name", "")}
- 페이지: {item.get("page", "")}
- 1차 분류: {item.get("content_type", "")}
- 기존 스타일 태그: {", ".join(item.get("style_tags", []))}

[원문]
{item.get("chunk_text", "")}
"""

        last_error: Exception | None = inherited_error
        for attempt in range(3):
            try:
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=NcsChunkInfo,
                        temperature=0.1,
                    ),
                )

                if getattr(response, "parsed", None) is not None:
                    parsed = response.parsed
                    result = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
                else:
                    result = json.loads(response.text)

                return self._merge_result(item, result)
            except Exception as exc:
                last_error = exc
                wait_seconds = 2 ** attempt
                print(f"   ! [재시도 {attempt + 1}/3] {exc}", flush=True)
                time.sleep(wait_seconds)

        return {
            "source_id": item.get("id", ""),
            "source_display_title": item.get("display_title", ""),
            "source_document_name": item.get("document_name", ""),
            "source_page": item.get("page", 0),
            "source_alias_names": item.get("alias_names", []),
            "source_content_type": item.get("content_type", ""),
            "source_style_tags": item.get("style_tags", []),
            "chunk_text": item.get("chunk_text", ""),
            "canonical_name": item.get("canonical_name", ""),
            "display_title": item.get("display_title", ""),
            "category": "drop",
            "service_type": [],
            "target_conditions": [],
            "tools": [],
            "steps": [],
            "cautions": [f"Gemini 정제 실패: {last_error}"] if last_error else [],
            "summary": "",
            "stylist_answer": "",
            "search_text": "",
            "is_valid": False,
            "source": "NCS PDF + Gemini",
            "year": "",
        }

    def _build_batch_prompt(self, items: list[dict]) -> str:
        blocks: list[str] = []
        for item in items:
            blocks.append(
                f"""[청크]
source_id: {item.get("id", "")}
문서명: {item.get("document_name", "")}
페이지: {item.get("page", "")}
1차 분류: {item.get("content_type", "")}
기존 스타일 태그: {", ".join(item.get("style_tags", []))}
원문:
{item.get("chunk_text", "")}
"""
            )

        joined_blocks = "\n\n".join(blocks)
        return f"""
당신은 헤어 시술 교육 자료를 미용사 챗봇용 RAG 데이터로 구조화하는 분석가입니다.

목표:
- 미용사가 고객에게 시술 방법, 준비, 마무리, 홈케어를 설명할 때 바로 활용 가능한 지식만 남깁니다.
- 교육행정/시험/평가/동의서/카드 서식/문헌정보/출판정보는 드롭합니다.
- 원문에 근거해 한국어로만 정리합니다.

판정 기준:
1. 직접적인 시술 절차, 준비물, 도구 선택, 관리법, 주의사항이면 유지합니다.
2. 지나치게 추상적인 학습행정 문구, 시험문제, 평가표, 개인정보 서식이면 드롭합니다.
3. 역사 서술, 장신구/관모/쓰개 종류 설명, 시대별 분류처럼 시술 설명에 직접 연결되지 않는 배경지식은 드롭합니다.
4. theory는 실제 시술 판단에 도움이 되는 모발/도구/기법 원리만 유지합니다.
5. steps, cautions, tools는 원문에 근거한 것만 넣고 없으면 빈 배열로 둡니다.
6. summary와 stylist_answer는 과장 없이 실무적으로 씁니다.
7. search_text에는 서비스 유형, 단계, 대상 조건, 도구, 핵심 절차를 함께 녹입니다.
8. 각 항목마다 source_id를 입력값과 정확히 동일하게 반환합니다.

아래 여러 개의 청크를 각각 독립적으로 분석해 JSON의 items 배열로 반환하세요.

{joined_blocks}
"""

    def _merge_result(self, item: dict, result: dict) -> dict:
        return {
            "source_id": item.get("id", ""),
            "source_display_title": item.get("display_title", ""),
            "source_document_name": item.get("document_name", ""),
            "source_page": item.get("page", 0),
            "source_alias_names": item.get("alias_names", []),
            "source_content_type": item.get("content_type", ""),
            "source_style_tags": item.get("style_tags", []),
            "chunk_text": item.get("chunk_text", ""),
            "canonical_name": result.get("canonical_name", ""),
            "display_title": item.get("display_title", ""),
            "category": result.get("category", ""),
            "service_type": result.get("service_type", []),
            "target_conditions": result.get("target_conditions", []),
            "tools": result.get("tools", []),
            "steps": result.get("steps", []),
            "cautions": result.get("cautions", []),
            "summary": result.get("summary", ""),
            "stylist_answer": result.get("stylist_answer", ""),
            "search_text": result.get("search_text", ""),
            "is_valid": bool(result.get("is_valid", False)),
            "source": "NCS PDF + Gemini",
            "year": "",
        }


def main() -> None:
    NcsLlmRefiner().refine_with_llm()


if __name__ == "__main__":
    main()
