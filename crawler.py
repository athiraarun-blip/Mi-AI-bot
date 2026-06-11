"""
Crawls mindfulminerals.com and saves extracted text to crawled_data.json.

Usage:
    python crawler.py
"""

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

SITEMAP_FILE = "sitemap.xml"
OUTPUT_FILE = "crawled_data.json"
SKIP_PATHS = {"/page/sign-in"}


def load_urls_from_sitemap(path: str) -> list[str]:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = []
    seen: set[str] = set()
    for loc in root.findall("sm:url/sm:loc", ns):
        url = loc.text.strip()
        if urlparse(url).path in SKIP_PATHS:
            continue
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


async def load_page(page, url: str) -> str | None:
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(2000)
        return await page.content()
    except Exception as e:
        print(f"  [WARN] Failed to load {url}: {e}")
        return None


def _extract_prices_from_json(soup: BeautifulSoup) -> list[str]:
    results = []
    for script in soup.find_all("script", type="application/json"):
        try:
            data = json.loads(script.string or "")
            price = data.get("price") or data.get("price_min")
            compare = data.get("compare_at_price") or data.get("compare_at_price_min")
            if price:
                results.append(f"${int(price) / 100:.2f}")
            if compare and compare != price:
                results.append(f"Compare at: ${int(compare) / 100:.2f}")
        except Exception:
            pass
    return results


def extract_product_page(soup: BeautifulSoup, url: str) -> dict:
    parts = [f"Product URL: {url}"]

    title = (
        soup.find("h1")
        or soup.find(class_=re.compile(r"product[_-]title|product[_-]name", re.I))
    )
    if title:
        parts.append(f"Product Name: {title.get_text(strip=True)}")

    # Try Shopify JSON first (most reliable)
    json_prices = _extract_prices_from_json(soup)
    for p in json_prices:
        parts.append(f"Price: {p}")

    # Fall back to CSS-class price elements (.price, .money, etc.)
    seen_prices: set[str] = set(json_prices)
    for el in soup.find_all(class_=re.compile(r"\bprice\b|\bmoney\b", re.I)):
        text = el.get_text(separator=" ", strip=True)
        if text and any(c.isdigit() for c in text) and text not in seen_prices:
            seen_prices.add(text)
            parts.append(f"Price: {text}")

    # data-price / data-compare-price attributes (cents)
    for el in soup.find_all(attrs={"data-price": True}):
        try:
            val = f"${int(el['data-price']) / 100:.2f}"
            if val not in seen_prices:
                seen_prices.add(val)
                parts.append(f"Price: {val}")
        except (ValueError, TypeError):
            pass
    for el in soup.find_all(attrs={"data-compare-price": True}):
        try:
            val = f"${int(el['data-compare-price']) / 100:.2f}"
            if val not in seen_prices:
                seen_prices.add(val)
                parts.append(f"Compare at price: {val}")
        except (ValueError, TypeError):
            pass

    # Sale / discount badges
    for badge in soup.find_all(class_=re.compile(r"sale|badge|discount|savings", re.I)):
        text = badge.get_text(strip=True)
        if text and len(text) < 80:
            parts.append(f"Tag: {text}")

    # Product description
    desc_el = (
        soup.find(class_=re.compile(r"product[_-]description|product[_-]detail", re.I))
        or soup.find(id=re.compile(r"product[_-]description", re.I))
        or soup.find(class_=re.compile(r"\bdescription\b", re.I))
    )
    if desc_el:
        parts.append(f"Description: {desc_el.get_text(separator=' ', strip=True)[:1200]}")

    # Ingredients / how-to sections
    for heading in soup.find_all(["h2", "h3", "h4"]):
        heading_text = heading.get_text(strip=True).lower()
        if any(kw in heading_text for kw in ("ingredient", "how to", "benefit", "about")):
            sibling = heading.find_next_sibling()
            if sibling:
                parts.append(f"{heading.get_text(strip=True)}: {sibling.get_text(separator=' ', strip=True)[:600]}")

    return {"url": url, "type": "product", "content": "\n".join(parts)}


def extract_general_page(soup: BeautifulSoup, url: str) -> dict:
    for tag in soup.find_all(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find(id=re.compile(r"main[_-]content|content", re.I))
        or soup.body
    )
    text = (main or soup).get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return {"url": url, "type": "page", "content": f"URL: {url}\n{text}"}


def parse_page(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    path_parts = urlparse(url).path.strip("/").split("/")
    # Both /product/<handle> and /products/<handle> are individual product pages
    is_product = (
        len(path_parts) >= 2
        and path_parts[0] in ("product", "products")
        and len(path_parts[1]) > 0
    )
    if is_product:
        return extract_product_page(soup, url)
    return extract_general_page(soup, url)


async def crawl() -> list[dict]:
    urls = load_urls_from_sitemap(SITEMAP_FILE)
    print(f"Loaded {len(urls)} URLs from sitemap")

    documents: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        await ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

        for i, url in enumerate(urls, 1):
            page = await ctx.new_page()
            print(f"[{i}/{len(urls)}] {url}")
            html = await load_page(page, url)
            await page.close()
            if html:
                doc = parse_page(html, url)
                documents.append(doc)
            await asyncio.sleep(0.5)

        await browser.close()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"\nCrawl complete. {len(documents)} pages saved to {OUTPUT_FILE}")
    return documents


if __name__ == "__main__":
    asyncio.run(crawl())
