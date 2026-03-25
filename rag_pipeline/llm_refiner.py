from __future__ import annotations

import json
import os
import time
from typing import Literal

from google import genai
from pydantic import BaseModel, Field

from .paths import TREND_PROCESSED_DIR, ensure_directories


class TrendInfo(BaseModel):
    is_valid: bool = Field(description="내용이 쓸모없는 단순 광고, 빈약한 잡담이면 false, 데이터로서 가치가 있으면 true")
    canonical_name: str = Field(description="정규화된 트렌드명 (예: faux bob, soft mullet, hydro bob). 영문 소문자 권장")
    category: Literal["style_trend", "color_trend", "celebrity_example", "styling_guide", "drop"] = Field(
        description=(
            "style_trend: 실제 헤어스타일의 유행 / "
            "color_trend: 시즌 헤어컬러 트렌드 / "
            "celebrity_example: 특정 셀럽의 헤어스타일 예시 / "
            "styling_guide: 관리법이나 연출 가이드 / "
            "drop: 광고 또는 무관한 콘텐츠"
        )
    )
    style_tags: list[str] = Field(description="추출된 주요 스타일 키워드 배열")
    color_tags: list[str] = Field(description="추출된 주요 컬러 키워드 배열")
    summary: str = Field(description="핵심 내용만 간추린 2~3문장 요약")
    search_text: str = Field(description="RAG 검색을 위한 합성 텍스트")


class LLMRefiner:
    def __init__(self, model_name: str | None = None) -> None:
        ensure_directories()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.model_name = model_name or os.environ.get("TREND_REFINER_MODEL", "gemini-2.5-flash")
        self.input_file = TREND_PROCESSED_DIR / "refined_trends.json"
        self.output_file = TREND_PROCESSED_DIR / "final_rag_trends.json"

    def refine_with_llm(self, delay_seconds: float = 1.0) -> list[dict]:
        if not self.api_key or self.client is None:
            print("⚠️ GEMINI_API_KEY가 없어 LLM 정제를 건너뜁니다.")
            return []

        if not self.input_file.exists():
            print(f"입력 파일이 없습니다: {self.input_file}")
            return []

        with self.input_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            print(f"입력 파일 형식이 올바르지 않습니다: {self.input_file}")
            return []

        print(f"====== 최상위 LLM RAG 데이터 정제 작업 시작 (총 {len(data)}건) ======")
        valid_items: list[dict] = []

        for index, item in enumerate(data, start=1):
            title = str(item.get("trend_name", ""))[:30]
            print(f"[{index}/{len(data)}] {title}...")

            prompt = f"""
당신은 최고의 헤어 트렌드 분석가이자 RAG 엔지니어입니다.
주어진 뷰티 매거진 텍스트를 분석하여, RAG 검색용 벡터 데이터베이스에 적합한 완벽한 스키마 구조로 파싱하세요.

[분리 및 필터링 기준]
1. 완벽한 트렌드 DB 남기기 (category: style_trend, color_trend, celebrity_example)
2. 헤어 가이드 분리하기 (category: styling_guide)
3. 배제하기 (category: drop, is_valid: false)

입력 텍스트:
제목: {item.get('trend_name')}
본문 내용: {item.get('description')}
"""

            try:
                time.sleep(delay_seconds)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=TrendInfo,
                        temperature=0.2,
                    ),
                )

                if getattr(response, "parsed", None) is not None:
                    parsed = response.parsed
                    result = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
                else:
                    result = json.loads(response.text)

                if result.get("is_valid") and result.get("category") != "drop":
                    valid_items.append(
                        {
                            "canonical_name": result.get("canonical_name", ""),
                            "display_title": item.get("trend_name", ""),
                            "category": result.get("category", ""),
                            "style_tags": result.get("style_tags", []),
                            "color_tags": result.get("color_tags", []),
                            "summary": result.get("summary", ""),
                            "search_text": result.get("search_text", ""),
                            "source": item.get("source", "Unknown"),
                            "year": str(item.get("year", "")),
                        }
                    )
                    print(f"   ✓ [채택] 카테고리: {result.get('category')} | 정규화: {result.get('canonical_name', '')}")
                else:
                    print("   X [드롭] 얕은 기사 또는 제품 광고 처리됨")
            except Exception as exc:
                print(f"   ! [에러] 처리 중 예외: {exc}")

        print("\n====== 중복 제거 작업 ======")
        deduplicated: list[dict] = []
        seen: set[str] = set()
        for item in valid_items:
            uniq_key = f"{item['canonical_name'].lower().strip()}_{item['category']}"
            if uniq_key in seen:
                continue
            seen.add(uniq_key)
            deduplicated.append(item)

        with self.output_file.open("w", encoding="utf-8") as file:
            json.dump(deduplicated, file, ensure_ascii=False, indent=2)

        print(
            f"\n====== 최종 정제 완료! 원본 {len(data)}건 -> 1차 정제 {len(valid_items)}건 -> 중복제거 최종 {len(deduplicated)}건 ======"
        )
        print(f"결과물 저장 경로: {self.output_file}")
        return deduplicated


if __name__ == "__main__":
    LLMRefiner().refine_with_llm()
