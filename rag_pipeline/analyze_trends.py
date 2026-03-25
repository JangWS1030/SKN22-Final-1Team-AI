from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .paths import ANALYSIS_DIR, TREND_PROCESSED_DIR, TREND_RAW_DIR, ensure_directories


class KeywordAnalyzer:
    def __init__(self) -> None:
        from konlpy.tag import Okt
        import matplotlib.pyplot as plt

        ensure_directories()
        self.okt = Okt()
        self.plt = plt
        self.data_dirs = [TREND_RAW_DIR, TREND_PROCESSED_DIR]
        self.output_dir = ANALYSIS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plt.rc("font", family="AppleGothic")
        self.plt.rcParams["axes.unicode_minus"] = False

        self.stopwords = [
            "수", "것", "이", "그", "저", "있", "하", "같", "에", "에서", "으로", "로",
            "곳", "분", "머리", "헤어", "스타일", "스타일링", "추천", "많이", "정말", "너무",
            "진짜", "요즘", "오늘", "지금", "유행", "트렌드", "년", "월", "일", "시", "분",
            "디자이너", "원장", "미용실", "고객", "시술", "진행", "생각", "느낌", "이미지",
            "얼굴", "사람", "우리", "나", "저희", "이번", "그냥", "항상", "조금",
        ]

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        return re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)

    def extract_nouns(self, text: str) -> list[str]:
        cleaned_text = self.clean_text(text)
        nouns = self.okt.nouns(cleaned_text)
        return [word for word in nouns if len(word) >= 2 and word not in self.stopwords]

    def _iter_json_files(self) -> list[Path]:
        paths: list[Path] = []
        for directory in self.data_dirs:
            if directory.exists():
                paths.extend(sorted(directory.glob("*.json")))
        return paths

    def load_all_data(self) -> tuple[str, list[str]]:
        all_text_parts: list[str] = []
        all_hashtags: list[str] = []

        print("데이터 로딩 중...")
        for file_path in self._iter_json_files():
            try:
                with file_path.open("r", encoding="utf-8") as file:
                    data = json.load(file)
            except Exception as exc:
                print(f"파일 읽기 에러 ({file_path.name}): {exc}")
                continue

            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                continue

            for item in data:
                if not isinstance(item, dict):
                    continue

                text_parts = [
                    item.get("trend_name", ""),
                    item.get("display_title", ""),
                    item.get("description", ""),
                    item.get("summary", ""),
                    item.get("search_text", ""),
                    item.get("content", ""),
                ]
                joined_text = " ".join(str(part) for part in text_parts if part)
                if joined_text:
                    all_text_parts.append(joined_text)

                hashtags = item.get("hashtags", [])
                if isinstance(hashtags, list):
                    all_hashtags.extend(str(tag).replace("#", "") for tag in hashtags if tag)

        return " ".join(all_text_parts), all_hashtags

    def _resolve_font_path(self) -> str | None:
        candidates = [
            Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
            Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def analyze_and_visualize(self) -> None:
        from wordcloud import WordCloud

        all_text, all_hashtags = self.load_all_data()
        if not all_text and not all_hashtags:
            print("분석할 데이터가 없습니다.")
            return

        print(f"총 {len(all_text)} 글자의 텍스트, {len(all_hashtags)}개의 해시태그 로드 완료.")
        print("명사 추출 및 빈도 계산 중 (시간이 소요될 수 있습니다)...")

        nouns = self.extract_nouns(all_text)
        noun_counts = Counter(nouns)
        top_30_nouns = noun_counts.most_common(30)

        refined_hashtags = [tag for tag in all_hashtags if tag not in self.stopwords and len(tag) > 1]
        hashtag_counts = Counter(refined_hashtags)
        top_30_hashtags = hashtag_counts.most_common(30)

        print("\n==== [TOP 20 많이 언급된 키워드 (본문)] ====")
        for word, count in top_30_nouns[:20]:
            print(f"- {word}: {count}회")

        print("\n==== [TOP 20 많이 쓰인 해시태그] ====")
        for tag, count in top_30_hashtags[:20]:
            print(f"- #{tag}: {count}회")

        result_dict = {"top_nouns": dict(top_30_nouns), "top_hashtags": dict(top_30_hashtags)}
        with (self.output_dir / "keyword_frequency.json").open("w", encoding="utf-8") as file:
            json.dump(result_dict, file, ensure_ascii=False, indent=2)

        print("\n워드클라우드 생성 중...")
        common_kwargs = {
            "width": 800,
            "height": 800,
            "background_color": "white",
        }
        font_path = self._resolve_font_path()
        if font_path:
            common_kwargs["font_path"] = font_path

        wordcloud_noun = WordCloud(colormap="viridis", **common_kwargs).generate_from_frequencies(noun_counts)
        wordcloud_hashtag = WordCloud(colormap="plasma", **common_kwargs).generate_from_frequencies(hashtag_counts)

        self.plt.figure(figsize=(16, 8))
        self.plt.subplot(1, 2, 1)
        self.plt.imshow(wordcloud_noun, interpolation="bilinear")
        self.plt.title("Top Keywords from Contents", fontsize=20)
        self.plt.axis("off")

        self.plt.subplot(1, 2, 2)
        self.plt.imshow(wordcloud_hashtag, interpolation="bilinear")
        self.plt.title("Top Hashtags", fontsize=20)
        self.plt.axis("off")

        figure_path = self.output_dir / "trend_wordcloud.png"
        self.plt.tight_layout()
        self.plt.savefig(figure_path)
        print(f"시각화 이미지가 저장되었습니다: {figure_path}")


if __name__ == "__main__":
    KeywordAnalyzer().analyze_and_visualize()
