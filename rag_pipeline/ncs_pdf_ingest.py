from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .paths import NCS_PROCESSED_DIR, NCS_SOURCE_DIR, ensure_directories


NCS_OUTPUT_FILE = NCS_PROCESSED_DIR / "ncs_manual_chunks.json"


@dataclass(frozen=True)
class PdfGroup:
    representative: Path
    aliases: list[str]


class NcsPdfIngestor:
    def __init__(
        self,
        *,
        source_dir: Path | None = None,
        output_file: Path | None = None,
        front_matter_pages: int = 13,
        min_chars: int = 120,
    ) -> None:
        ensure_directories()
        self.source_dir = source_dir or NCS_SOURCE_DIR
        self.output_file = output_file or NCS_OUTPUT_FILE
        self.front_matter_pages = front_matter_pages
        self.min_chars = min_chars

        self.page_skip_patterns = [
            re.compile(r"차\s*례"),
            re.compile(r"NCS-학습모듈의 위치"),
            re.compile(r"NCS학습모듈 개발이력"),
            re.compile(r"학습모듈의 개요"),
            re.compile(r"교수\s*[·.\-]?\s*학습 방법"),
            re.compile(r"활용 서식"),
            re.compile(r"발행일"),
            re.compile(r"개발기관"),
            re.compile(r"집필진"),
            re.compile(r"고객관리 차트"),
            re.compile(r"고객카드"),
            re.compile(r"개인정보"),
            re.compile(r"동의서"),
            re.compile(r"회원정보"),
            re.compile(r"평가지"),
            re.compile(r"진단 영역"),
            re.compile(r"시험문제"),
            re.compile(r"사지선다형"),
            re.compile(r"단답형"),
            re.compile(r"학습자 확인"),
            re.compile(r"총점"),
            re.compile(r"평가 방법"),
            re.compile(r"평가\s*준거"),
            re.compile(r"평가\s*항목"),
            re.compile(r"수행\s*준거"),
            re.compile(r"형성\s*평가"),
            re.compile(r"확인\s*문제"),
            re.compile(r"참고\s*자료"),
            re.compile(r"참고문헌"),
            re.compile(r"평가자 체크리스트"),
            re.compile(r"작업장 평가"),
            re.compile(r"실습평가"),
            re.compile(r"포트폴리오"),
            re.compile(r"피드백"),
        ]
        self.stop_section_patterns = [
            re.compile(r"^\s*교수\s*[·.\-]?\s*학습 방법\s*$"),
            re.compile(r"^\s*평가 방법\s*$"),
            re.compile(r"^\s*평가(?:\s*준거|\s*항목)?\s*$"),
            re.compile(r"^\s*수행\s*준거\s*$"),
            re.compile(r"^\s*형성\s*평가\s*$"),
            re.compile(r"^\s*확인\s*문제\s*$"),
            re.compile(r"^\s*참고\s*자료\s*$"),
            re.compile(r"^\s*참고문헌\s*$"),
            re.compile(r"^\s*활용\s*서식\s*$"),
            re.compile(r"^\s*평가자 체크리스트\s*$"),
            re.compile(r"^\s*작업장 평가\s*$"),
            re.compile(r"^\s*실습평가\s*$"),
            re.compile(r"^\s*포트폴리오\s*$"),
            re.compile(r"^\s*피드백\s*$"),
        ]
        self.drop_line_patterns = [
            re.compile(r"^\s*$"),
            re.compile(r"^\s*\d+\s*$"),
            re.compile(r"^\s*출처[:\s].*$"),
            re.compile(r"^\s*\[그림.*$"),
            re.compile(r"^\s*<표.*$"),
        ]
        self.style_keywords = [
            "커트",
            "컷",
            "펌",
            "컬러",
            "염색",
            "블로우드라이",
            "업스타일",
            "가발",
            "샴푸",
            "클리닉",
            "두피",
            "손상모",
            "헤어스타일",
        ]
        self.content_type_patterns = [
            ("procedure", re.compile(r"수행\s*순서|시술\s*순서|실행하기|마무리하기")),
            ("safety", re.compile(r"안전\s*[·･]?\s*유의 사항")),
            ("knowledge", re.compile(r"필요 지식|용어 정리|이론")),
            ("tip", re.compile(r"수행 tip|tip")),
        ]

    def extract(self) -> list[dict]:
        print("====== NCS PDF 전처리 시작 ======")
        if not self.source_dir.exists():
            print(f"PDF 소스 디렉터리가 없습니다: {self.source_dir}")
            return []

        groups = self._group_duplicate_pdfs()
        all_chunks: list[dict] = []

        for group in groups:
            print(f"[{group.representative.name}] 별칭 {len(group.aliases)}개 처리 중...")
            all_chunks.extend(self._extract_pdf_chunks(group))

        deduped = self._deduplicate_chunks(all_chunks)
        with self.output_file.open("w", encoding="utf-8") as file:
            json.dump(deduped, file, ensure_ascii=False, indent=2)

        print(f"====== NCS PDF 전처리 완료: {len(deduped)}개 청크 ======")
        print(f"저장 경로: {self.output_file}")
        return deduped

    def _group_duplicate_pdfs(self) -> list[PdfGroup]:
        hash_to_paths: dict[str, list[Path]] = defaultdict(list)
        for path in sorted(self.source_dir.glob("*.pdf")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_to_paths[digest].append(path)

        groups: list[PdfGroup] = []
        for paths in sorted(hash_to_paths.values(), key=lambda items: items[0].name):
            aliases = [path.stem for path in sorted(paths)]
            representative = sorted(paths, key=lambda path: (len(path.stem), path.stem))[0]
            groups.append(PdfGroup(representative=representative, aliases=aliases))
        return groups

    def _extract_pdf_chunks(self, group: PdfGroup) -> list[dict]:
        reader = PdfReader(str(group.representative))
        chunks: list[dict] = []

        for page_number, page in enumerate(reader.pages, start=1):
            if page_number <= self.front_matter_pages:
                continue

            lines = self._clean_lines(page.extract_text() or "")
            if not lines:
                continue
            if self._is_skippable_page(lines):
                continue

            filtered_lines = self._filter_lines(lines)
            text = "\n".join(filtered_lines).strip()
            if len(text) < self.min_chars:
                continue

            stem = group.representative.stem
            aliases = ", ".join(group.aliases)
            page_id = f"{self._slugify(stem)}_p{page_number:03d}"
            summary = self._build_summary(text)
            content_type = self._infer_content_type(text)
            style_tags = self._extract_style_tags(" ".join(group.aliases), text)
            search_text = f"{stem} {aliases} 페이지 {page_number} {text}"

            chunks.append(
                {
                    "id": page_id,
                    "canonical_name": self._slugify(stem),
                    "display_title": f"{stem} p.{page_number}",
                    "category": "ncs_manual",
                    "style_tags": style_tags,
                    "color_tags": [],
                    "summary": summary,
                    "search_text": search_text,
                    "source": "NCS PDF",
                    "year": "",
                    "document_name": stem,
                    "alias_names": group.aliases,
                    "page": page_number,
                    "content_type": content_type,
                    "chunk_text": text,
                }
            )

        return chunks

    def _clean_lines(self, text: str) -> list[str]:
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.replace("\x00", " ")
            line = re.sub(r"\s+", " ", line).strip()
            line = re.split(r"출\s*처\s*:", line, maxsplit=1)[0].strip()
            if not line:
                continue
            lines.append(line)
        return lines

    def _is_skippable_page(self, lines: list[str]) -> bool:
        joined = " ".join(lines)
        if any(pattern.search(joined) for pattern in self.page_skip_patterns):
            return True
        if joined.count("교육부(") >= 3 or joined.count("한국직업능력개발원") >= 3:
            return True
        if "학위논문" in joined and joined.count("대학교") >= 2:
            return True
        if all(keyword in joined for keyword in ["발행일", "개발기관", "집필진"]):
            return True
        return False

    def _filter_lines(self, lines: list[str]) -> list[str]:
        filtered: list[str] = []
        for line in lines:
            if any(pattern.match(line) for pattern in self.stop_section_patterns):
                break
            if any(pattern.match(line) for pattern in self.drop_line_patterns):
                continue
            filtered.append(line)
        return filtered

    def _build_summary(self, text: str, limit: int = 220) -> str:
        compact = text.replace("\n", " ")
        return compact[:limit].strip()

    def _infer_content_type(self, text: str) -> str:
        for name, pattern in self.content_type_patterns:
            if pattern.search(text):
                return name
        return "manual"

    def _extract_style_tags(self, aliases_text: str, text: str) -> list[str]:
        combined = f"{aliases_text} {text}"
        return [keyword for keyword in self.style_keywords if keyword in combined]

    def _deduplicate_chunks(self, chunks: list[dict]) -> list[dict]:
        deduped: list[dict] = []
        seen: set[str] = set()
        for item in chunks:
            key = hashlib.sha256(item["chunk_text"].encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _slugify(self, value: str) -> str:
        value = unicodedata.normalize("NFC", value).lower().replace("+", "_").replace(" ", "_")
        value = re.sub(r"[^0-9a-zA-Z가-힣_]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        return value or "ncs"


def main() -> None:
    NcsPdfIngestor().extract()


if __name__ == "__main__":
    main()
