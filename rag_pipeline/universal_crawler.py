from __future__ import annotations

import json
import time
from datetime import datetime

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from .paths import TREND_RAW_DIR, ensure_directories


class UniversalCrawler:
    def __init__(self) -> None:
        ensure_directories()
        self.data_dir = TREND_RAW_DIR
        self.targets = [
            {"name": "allure", "url": "https://www.allure.com/hair-ideas", "base": "https://www.allure.com", "keywords": ["/story/", "/gallery/"]},
            {"name": "byrdie", "url": "https://www.byrdie.com/hair-styling-4628405", "base": "https://www.byrdie.com", "keywords": ["hair"]},
            {"name": "marieclaire", "url": "https://www.marieclaire.com/beauty/hair/", "base": "https://www.marieclaire.com", "keywords": ["/beauty/", "hair"]},
            {"name": "harpersbazaar", "url": "https://www.harpersbazaar.com/beauty/hair/", "base": "https://www.harpersbazaar.com", "keywords": ["/beauty/hair/a"]},
            {"name": "instyle", "url": "https://www.instyle.com/hair", "base": "https://www.instyle.com", "keywords": ["hair"]},
            {"name": "glamour", "url": "https://www.glamour.com/beauty/hair", "base": "https://www.glamour.com", "keywords": ["/story/", "/gallery/"]},
            {"name": "vogue", "url": "https://www.vogue.com/beauty/hair", "base": "https://www.vogue.com", "keywords": ["/article/"]},
            {"name": "whowhatwear", "url": "https://www.whowhatwear.com/beauty/hair", "base": "https://www.whowhatwear.com", "keywords": ["/beauty/hair/"]},
            {"name": "elle", "url": "https://www.elle.com/beauty/hair/", "base": "https://www.elle.com", "keywords": ["/beauty/hair/a", "/beauty/"]},
            {"name": "trendspotter_women", "url": "https://www.thetrendspotter.net/category/womens-hairstyles/", "base": "https://www.thetrendspotter.net", "keywords": ["hair"]},
            {"name": "gq", "url": "https://www.gq.com/about/hair", "base": "https://www.gq.com", "keywords": ["/story/", "/gallery/"]},
            {"name": "trendspotter_men", "url": "https://www.thetrendspotter.net/category/mens-hairstyles/", "base": "https://www.thetrendspotter.net", "keywords": ["hair"]},
            {"name": "americansalon", "url": "https://www.americansalon.com/hair-0", "base": "https://www.americansalon.com", "keywords": ["/hair/"]},
            {"name": "beautylaunchpad_cut", "url": "https://www.beautylaunchpad.com/cut", "base": "https://www.beautylaunchpad.com", "keywords": ["/cut/"]},
            {"name": "beautylaunchpad_color", "url": "https://www.beautylaunchpad.com/color", "base": "https://www.beautylaunchpad.com", "keywords": ["/color/"]},
            {"name": "beautylaunchpad_styles", "url": "https://www.beautylaunchpad.com/styles", "base": "https://www.beautylaunchpad.com", "keywords": ["/styles/"]},
            {"name": "hypehair", "url": "https://hypehair.com/category/hair/", "base": "https://hypehair.com", "keywords": ["hair", "/20"]},
            {"name": "hji", "url": "https://hji.co.uk/trends", "base": "https://hji.co.uk", "keywords": ["/trends/"]},
            {"name": "esteticamagazine", "url": "https://www.esteticamagazine.com/category/trends/hair-collection/", "base": "https://www.esteticamagazine.com", "keywords": ["/trends", "/hair", "collection"]},
            {"name": "wkorea", "url": "https://www.wkorea.com/beauty/", "base": "https://www.wkorea.com", "keywords": ["/beauty/"]},
            {"name": "elle_korea", "url": "https://www.elle.co.kr/beauty", "base": "https://www.elle.co.kr", "keywords": ["/beauty/"]},
            {"name": "vogue_korea", "url": "https://www.vogue.co.kr/beauty/", "base": "https://www.vogue.co.kr", "keywords": ["/beauty/"]},
            {"name": "harpersbazaar_korea", "url": "https://www.harpersbazaar.co.kr/beauty", "base": "https://www.harpersbazaar.co.kr", "keywords": ["/beauty/"]},
            {"name": "marieclaire_korea", "url": "https://www.marieclairekorea.com/category/beauty/beauty_trend/", "base": "https://www.marieclairekorea.com", "keywords": ["/beauty/", "/category/"]},
            {"name": "gq_korea", "url": "https://www.gqkorea.co.kr/style/grooming/", "base": "https://www.gqkorea.co.kr", "keywords": ["/style/", "/grooming/"]},
            {"name": "cosmopolitan_korea", "url": "https://www.cosmopolitan.co.kr/beauty", "base": "https://www.cosmopolitan.co.kr", "keywords": ["/beauty/"]},
            {"name": "allure_korea", "url": "https://www.allurekorea.com/beauty/hair/", "base": "https://www.allurekorea.com", "keywords": ["/beauty/", "/hair/"]},
        ]

    def _is_article_link(self, href: str | None, keywords: list[str], base_url: str) -> bool:
        del base_url
        if not href:
            return False

        excludes = ["/about/", "/contact", "/privacy", "author", "tag", "/category/", "?page=", "newsletter", "subscribe"]
        href_lower = href.lower()
        if any(exc in href_lower for exc in excludes):
            return False

        if keywords and keywords[0] != "/" and not any(kw in href for kw in keywords):
            return False

        if len(href.split("/")) < 4 and len(href) < 30:
            return False

        return True

    def _extract_body_text(self, soup: BeautifulSoup) -> list[str]:
        article = soup.find("article")
        if article:
            paragraphs = [p.get_text(strip=True) for p in article.find_all("p") if p.get_text(strip=True)]
            if paragraphs:
                return paragraphs

        body_selectors = [
            {"class_": lambda value: value and any(key in str(value).lower() for key in ["article-body", "post-content", "entry-content", "article_body", "article-content", "content-body", "story-body"])},
            {"class_": lambda value: value and any(key in str(value).lower() for key in ["detail", "view_cont", "article_cont", "news_body", "article_view"])},
        ]
        for selector in body_selectors:
            container = soup.find("div", **selector)
            if not container:
                continue
            paragraphs = [p.get_text(strip=True) for p in container.find_all("p") if p.get_text(strip=True)]
            if paragraphs:
                return paragraphs

        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 30]
        return paragraphs

    def parse_article(self, html: str, source_name: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        items: list[dict] = []

        for tag in soup.find_all(["nav", "footer", "header", "aside", "script", "style", "iframe"]):
            tag.decompose()

        title_elem = soup.find("h1")
        main_title = title_elem.get_text(strip=True) if title_elem else ""
        if not main_title:
            og_title = soup.find("meta", property="og:title")
            if og_title:
                main_title = og_title.get("content", "")
        if not main_title:
            title = soup.find("title")
            if title:
                main_title = title.get_text(strip=True)

        year = str(datetime.now().year)
        headings = soup.find_all(["h2", "h3"])

        if len(headings) >= 2:
            for heading in headings:
                trend_name = heading.get_text(strip=True)
                if len(trend_name) > 100 or len(trend_name) < 3:
                    continue

                desc_paragraphs: list[str] = []
                for element in heading.find_all_next():
                    if element.name in ["h2", "h3", "h1"]:
                        break
                    if element.name != "p":
                        continue
                    text = element.get_text(strip=True)
                    if text and len(text) > 10:
                        desc_paragraphs.append(text)

                if not desc_paragraphs:
                    continue

                items.append(
                    {
                        "trend_name": trend_name,
                        "year": year,
                        "hairstyle_text": "",
                        "color_text": "",
                        "description": "\n".join(desc_paragraphs),
                        "source": source_name,
                    }
                )

        if not items and main_title:
            paragraphs = self._extract_body_text(soup)
            body = "\n".join(paragraphs)
            if len(body) > 100:
                items.append(
                    {
                        "trend_name": main_title,
                        "year": year,
                        "hairstyle_text": "",
                        "color_text": "",
                        "description": body,
                        "source": source_name,
                    }
                )

        return items

    def crawl(self) -> None:
        print("======== [Universal Crawler] 모든 매거진 크롤링 시작 ========")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
            )
            page = context.new_page()

            for target in self.targets:
                name = target["name"]
                url = target["url"]
                base_url = target["base"]
                keywords = target["keywords"]

                print(f"\n--- [{name}] 탐색 시작 ({url}) ---")

                try:
                    page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    time.sleep(3)

                    for scroll_y in [1000, 2000, 3000]:
                        page.mouse.wheel(0, scroll_y)
                        time.sleep(1)

                    html = page.content()
                    soup = BeautifulSoup(html, "html.parser")

                    links: set[str] = set()
                    for anchor in soup.find_all("a", href=True):
                        href = anchor["href"]
                        if not self._is_article_link(href, keywords, base_url):
                            continue
                        full_url = href if href.startswith("http") else base_url + href
                        links.add(full_url)

                    target_links = list(links)[:8]
                    if not target_links:
                        print(f"[{name}] 기사 링크를 찾지 못했습니다.")
                        continue

                    results: list[dict] = []
                    for act_url in target_links:
                        print(f"  -> [{name}] 수집 중: {act_url}")
                        try:
                            page.goto(act_url, timeout=30000, wait_until="domcontentloaded")
                            time.sleep(2)
                            page.mouse.wheel(0, 1500)
                            time.sleep(1)

                            article_html = page.content()
                            results.extend(self.parse_article(article_html, name.capitalize()))
                        except Exception as exc:
                            print(f"  -> [{name}] 페이지 수집 에러 ({act_url}): {exc}")

                    unique_results: list[dict] = []
                    seen: set[str] = set()
                    for result in results:
                        key = result["trend_name"] + str(result["description"])[:30]
                        if key in seen:
                            continue
                        seen.add(key)
                        unique_results.append(result)

                    output_path = self.data_dir / f"{name}.json"
                    with output_path.open("w", encoding="utf-8") as file:
                        json.dump(unique_results, file, ensure_ascii=False, indent=2)
                    print(f"--- [{name}] 완료. {len(unique_results)}개 아이템 저장됨 ---")
                except Exception as exc:
                    print(f"[{name}] 접근 실패: {exc}")

            browser.close()


if __name__ == "__main__":
    UniversalCrawler().crawl()
