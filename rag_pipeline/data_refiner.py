from __future__ import annotations

import json
import re
from pathlib import Path

from .paths import TREND_PROCESSED_DIR, TREND_RAW_DIR, ensure_directories


class DataRefiner:
    def __init__(self) -> None:
        ensure_directories()
        self.raw_data_dir = TREND_RAW_DIR
        self.output_file = TREND_PROCESSED_DIR / "refined_trends.json"

        self.junk_patterns = [
            r"Currently, only residents from GDPR countries.*?Privacy Policy for more information\.",
            r"©2026Condé Nast.*?Ad Choices\s*CN Fashion & Beauty",
            r"All products featured on.*?through these links\.",
            r"Sign up for.*?newsletter.*?wellness\.",
            r"The Vogue Runway app has expanded!.*?\bcontributors\.",
            r"Become a.*?Member—the ultimate resource.*?\bprofessionals\.",
            r"Have a beauty or wellness trend you’re curious about\?.*?@vogue\.com\.",
            r"More from Vogue.*?See More Stories",
            r"Related Video.*?(?=\n|$)",
            r"Shop the look.*?(?=\n|$)",
            r"Shop our favorite.*?(?=\n|$)",
            r"Available at Amazon.*?(?=\n|$)",
            r"Available at Sephora.*?(?=\n|$)",
            r"Available at Nordstrom.*?(?=\n|$)",
        ]

        self.hair_styles = [
            "bob", "pixie", "layered", "curtain bangs", "bangs", "updo",
            "bun", "braid", "ponytail", "lob", "shag", "mullet", "extensions",
            "chignon", "waves", "curls", "knot", "half-up",
            "단발", "숏컷", "레이어드", "시스루뱅", "앞머리", "업스타일",
            "번헤어", "브레이드", "포니테일", "허쉬컷", "울프컷", "빌드컷",
            "웨이브", "컬", "펌", "히피펌", "레이어컷", "가르마",
            "보브컷", "태슬컷", "샤기컷", "헤어스타일", "머리스타일",
            "투블럭", "리젠트", "댄디컷", "포마드", "바버컷", "크롭컷",
            "애즈펌", "다운펌", "가일컷", "리프컷",
        ]

        self.hair_colors = [
            "blonde", "brunette", "red", "copper", "balayage", "highlights",
            "ombre", "silver", "gray", "black", "brown", "caramel", "chocolate",
            "strawberry", "auburn", "platinum",
            "염색", "탈색", "블론드", "브루넷", "발레아쥬", "하이라이트",
            "옴브레", "애쉬", "밀크티컬러", "핑크브라운", "초코브라운",
            "카라멜", "구리빛", "레드브라운", "로즈골드", "베이지브라운",
            "흑발", "다크브라운", "톤다운", "톤업", "컬러링", "새치염색",
            "그레이컬러", "실버컬러", "머쉬룸블론드",
        ]

        self.banned_keywords = [
            "1990s", "1980s", "1970s", "1960s", "1950s", "vintage", "old hollywood",
            "history", "retro", "nuptials", "wedding", "bridal", "bride", "royal", "princess",
            "how to use", "best shampoo", "best conditioner", "hair dryer", "curling iron",
            "flat iron", "serum", "scalp scrub", "hair growth", "hair loss", "thinning hair",
            "dandruff", "vitamin c", "pillowcase", "leggings", "showerhead", "shop now",
            "buy now", "amazon", "sephora", "nordstrom", "ulta", "price:",
            "스킨케어", "피부관리", "피부과", "여드름", "주름", "모공",
            "파운데이션", "립스틱", "아이섀도", "마스카라", "컨실러", "블러셔",
            "선크림", "자외선", "톤크림", "쿠션팩트", "비비크림",
            "향수", "퍼퓸", "디퓨저", "바디로션", "바디워시", "핸드크림",
            "네일", "매니큐어", "페디큐어", "젤네일",
            "구매하기", "최저가", "할인", "쿠폰", "무료배송",
            "다이어트", "운동", "필라테스", "요가", "헬스",
            "패션위크", "컬렉션 리뷰", "런웨이 리뷰", "스트리트패션",
            "가방", "신발", "시계", "주얼리", "악세서리", "선글라스",
            "인테리어", "레시피", "맛집", "여행", "호텔",
        ]

        self.banned_patterns = [
            r"\b19\d{2}\b",
            r"\'?[89]0s",
            r"spray \d+ to \d+",
            r"how to use",
        ]

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        for pattern in self.junk_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        text = re.sub(r"\n+", "\n", text)
        return text.strip()

    def extract_attributes(self, text: str, title: str) -> tuple[str, str]:
        combined_text = f"{title} {text}".lower()
        found_styles = [style for style in self.hair_styles if style in combined_text]
        found_colors = [color for color in self.hair_colors if color in combined_text]
        return ", ".join(sorted(set(found_styles))), ", ".join(sorted(set(found_colors)))

    def _iter_source_files(self) -> list[Path]:
        return sorted(path for path in self.raw_data_dir.glob("*.json") if path.is_file())

    def refine(self) -> list[dict]:
        print("====== 데이터 정제 작업 시작 ======")
        all_items: list[dict] = []
        target_files = self._iter_source_files()

        if not target_files:
            print(f"원본 데이터가 없습니다: {self.raw_data_dir}")
            return []

        for filepath in target_files:
            try:
                with filepath.open("r", encoding="utf-8") as file:
                    data = json.load(file)
            except Exception as exc:
                print(f"[{filepath.name}] 처리 중 에러: {exc}")
                continue

            if not isinstance(data, list):
                print(f"[{filepath.name}] 리스트 형식이 아니어서 스킵합니다.")
                continue

            for raw_item in data:
                if not isinstance(raw_item, dict):
                    continue

                item = dict(raw_item)
                cleaned_desc = self.clean_text(str(item.get("description", "")))
                if len(cleaned_desc) < 30:
                    continue

                current_style = str(item.get("hairstyle_text", "") or "")
                current_color = str(item.get("color_text", "") or "")
                ext_style, ext_color = self.extract_attributes(cleaned_desc, str(item.get("trend_name", "")))

                combined_text_for_filter = f"{item.get('trend_name', '')} {cleaned_desc}".lower()
                if any(banned_word in combined_text_for_filter for banned_word in self.banned_keywords):
                    continue
                if any(re.search(pattern, combined_text_for_filter, re.IGNORECASE) for pattern in self.banned_patterns):
                    continue
                if not current_style and not ext_style and not current_color and not ext_color:
                    continue

                if not current_style:
                    item["hairstyle_text"] = ext_style
                if not current_color:
                    item["color_text"] = ext_color

                item["description"] = cleaned_desc
                all_items.append(item)

            print(f"[{filepath.name}] {len(data)}건 중 유효 데이터 추출 완료.")

        unique_items: list[dict] = []
        seen: set[str] = set()
        for item in all_items:
            key = (item.get("trend_name", "") + item.get("description", "")[:50]).strip()
            if key in seen:
                continue
            seen.add(key)
            unique_items.append(item)

        with self.output_file.open("w", encoding="utf-8") as file:
            json.dump(unique_items, file, ensure_ascii=False, indent=2)

        print(f"====== 정제 완료! 총 {len(unique_items)}건의 트렌드 데이터가 저장되었습니다. ======")
        print(f"저장 경로: {self.output_file}")
        return unique_items


if __name__ == "__main__":
    DataRefiner().refine()
