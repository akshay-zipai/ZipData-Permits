"""
Crawler Service — handles both static HTML and JS-rendered permit portals.

Strategy:
  1. Try plain httpx (fast, works for static sites)
  2. On 403 (bot block): immediately fall back to Playwright
  3. On 429 (rate limit): respect Retry-After header, then retry
  4. On thin content (JS-rendered): fall back to Playwright
  5. After all httpx retries fail (5xx / timeout): try Playwright as last resort
  6. Playwright: tries networkidle first, falls back to domcontentloaded
  7. PDF links on crawled pages are extracted and fetched (small counties
     typically publish zoning/permit codes as PDFs)
  8. Never cache empty crawl results
"""
import time
import asyncio
import hashlib
from urllib.parse import urljoin, urlparse
from typing import Optional
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.responses import CrawledPage, CrawlResponse
from app.utils.text_processing import clean_text, word_count

logger = get_logger(__name__)
settings = get_settings()

# Pages with fewer words than this are considered JS-rendered or empty
MIN_WORD_THRESHOLD = 50

# Realistic browser headers — reduces bot detection rejections
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}

# In-memory cache: cache_key → CrawlResponse
_crawl_cache: dict[str, CrawlResponse] = {}


def _extract_text(html: str) -> str:
    """Extract clean text from HTML, removing noise tags."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "iframe", "svg", "meta", "link"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return clean_text(text)


def _extract_title(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return None


def _extract_links(html: str, base_url: str, base_domain: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = str(tag["href"]).strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        full_url = urljoin(base_url, href).split("#")[0].split("?")[0]
        parsed = urlparse(full_url)
        if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
            links.append(full_url)
    return list(set(links))


def _extract_pdf_links(html: str, base_url: str) -> list[str]:
    """
    Extract PDF links from a page — cross-domain allowed since county sites
    often link to state-hosted or third-party permit/zoning PDFs.

    Catches both:
    - URLs ending in .pdf
    - Links whose visible text contains "pdf" (e.g. "Zoning Code PDF",
      "Full Text PDF") — handles CivicPlus /DocumentCenter/ and similar
      document-manager URLs that serve PDFs without a .pdf extension.
    """
    soup = BeautifulSoup(html, "html.parser")
    pdf_links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = str(tag["href"]).strip()
        if not href:
            continue
        clean = href.split("?")[0].split("#")[0].lower()
        link_text = tag.get_text(strip=True).lower()
        is_pdf_url = clean.endswith(".pdf")
        is_pdf_hint = "pdf" in link_text  # e.g. "Zoning Code PDF", "View PDF"
        if is_pdf_url or is_pdf_hint:
            full_url = urljoin(base_url, href).split("#")[0]
            parsed = urlparse(full_url)
            if parsed.scheme in ("http", "https"):
                pdf_links.append(full_url)
    return list(set(pdf_links))


def _extract_pdf_text(pdf_bytes: bytes) -> Optional[str]:
    """Extract plain text from PDF bytes using pdfminer."""
    try:
        from pdfminer.high_level import extract_text
        import io
        text = extract_text(io.BytesIO(pdf_bytes))
        return clean_text(text) if text and text.strip() else None
    except Exception as e:
        logger.warning(f"pdfminer extraction failed: {e}")
        return None


async def _fetch_with_playwright(url: str) -> Optional[str]:
    """
    Render pages using a headless Chromium browser.

    Tries networkidle first (best for fully JS-rendered apps).
    Falls back to domcontentloaded if networkidle times out — many government
    sites have analytics/tracking scripts that keep firing and block networkidle.
    """
    try:
        from playwright.async_api import async_playwright, TimeoutError as PWTimeout
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=BROWSER_HEADERS["User-Agent"],
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                ignore_https_errors=True,
            )
            page = await context.new_page()

            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
            except PWTimeout:
                logger.info(f"Playwright networkidle timed out for {url} — retrying with domcontentloaded")
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    # Extra wait for lazy-loaded content after DOM is ready
                    await asyncio.sleep(3)
                except Exception as e:
                    logger.warning(f"Playwright navigation failed for {url}: {e}")
                    await browser.close()
                    return None

            # Allow lazy-loaded content to settle
            await asyncio.sleep(2)
            html = await page.content()
            await browser.close()
            return html
    except ImportError:
        logger.warning("playwright not installed — skipping JS rendering fallback")
        return None
    except Exception as e:
        logger.warning(f"Playwright failed for {url}: {e}")
        return None


class CrawlerService:
    """
    Async web crawler with static + JS-rendered page support.
    Falls back to playwright when httpx returns thin/empty content.
    """

    def __init__(self):
        self.timeout = httpx.Timeout(settings.CRAWL_TIMEOUT_SECONDS)

    async def crawl_url(
        self,
        url: str,
        county_name: str,
        force_refresh: bool = False,
    ) -> CrawlResponse:
        cache_key = hashlib.md5(url.encode()).hexdigest()
        logger.info(
            "Starting crawl for county=%s portal=%s force_refresh=%s max_pages=%s",
            county_name,
            url,
            force_refresh,
            settings.CRAWL_MAX_PAGES_PER_DOMAIN,
        )

        if not force_refresh and cache_key in _crawl_cache:
            logger.info(f"Cache hit for {url}")
            cached = _crawl_cache[cache_key]
            return CrawlResponse(**{**cached.model_dump(), "cached": True})

        start = time.perf_counter()
        pages: list[CrawledPage] = []
        visited: set[str] = set()
        to_visit: list[str] = [url]
        base_domain = urlparse(url).netloc

        async with httpx.AsyncClient(
            headers=BROWSER_HEADERS,
            timeout=self.timeout,
            follow_redirects=True,
            verify=False,  # some county sites have cert issues
        ) as client:
            while to_visit and len(pages) < settings.CRAWL_MAX_PAGES_PER_DOMAIN:
                current_url = to_visit.pop(0)
                if current_url in visited:
                    continue
                visited.add(current_url)

                page = await self._fetch_page(client, current_url, base_domain)
                if page:
                    pages.append(page)
                    html_snapshot = page.html_content or ""
                    new_links = _extract_links(html_snapshot, current_url, base_domain)
                    pdf_links = _extract_pdf_links(html_snapshot, current_url)
                    logger.info(
                        "Crawl page success url=%s title=%s words=%s discovered_links=%s discovered_pdfs=%s",
                        page.url,
                        page.title or "n/a",
                        page.word_count,
                        len(new_links),
                        len(pdf_links),
                    )
                    to_visit.extend(lnk for lnk in new_links if lnk not in visited)
                    to_visit.extend(lnk for lnk in pdf_links if lnk not in visited)
                    # Clear html_content after link extraction to save memory
                    page.html_content = None
                else:
                    logger.info("Crawl page skipped or empty url=%s", current_url)

                await asyncio.sleep(settings.CRAWL_DELAY_SECONDS)

        elapsed = time.perf_counter() - start
        logger.info(
            "Crawl finished county=%s portal=%s pages=%s total_words=%s duration=%.1fs",
            county_name,
            url,
            len(pages),
            sum(p.word_count for p in pages),
            elapsed,
        )

        if not pages:
            logger.warning(
                f"No pages scraped from {url}. "
                "The site may require JavaScript rendering or is blocking requests."
            )

        response = CrawlResponse(
            county_name=county_name,
            portal_url=url,
            pages=pages,
            total_pages=len(pages),
            total_words=sum(p.word_count for p in pages),
            crawl_duration_seconds=round(elapsed, 2),
            cached=False,
        )
        # Only cache successful crawls — empty results must not be locked in
        if pages:
            _crawl_cache[cache_key] = response
        return response

    async def _fetch_pdf_page(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> Optional[CrawledPage]:
        """Download a PDF and return a CrawledPage with its extracted text."""
        PDF_MAX_BYTES = 20 * 1024 * 1024  # 20 MB guard
        try:
            resp = await client.get(url)
            if not (200 <= resp.status_code < 300):
                logger.warning(f"PDF fetch returned {resp.status_code}: {url}")
                return None

            content_length = int(resp.headers.get("content-length", 0))
            if content_length > PDF_MAX_BYTES:
                logger.warning(f"PDF too large ({content_length} bytes), skipping: {url}")
                return None

            pdf_bytes = resp.content
            if len(pdf_bytes) > PDF_MAX_BYTES:
                logger.warning(f"PDF too large after download, skipping: {url}")
                return None

            text = _extract_pdf_text(pdf_bytes)
            if not text:
                logger.debug(f"No text extracted from PDF: {url}")
                return None

            wc = word_count(text)
            if wc < 10:
                return None

            # Use URL filename as a human-readable title
            filename = url.split("/")[-1].split("?")[0]
            title = filename.replace(".pdf", "").replace("-", " ").replace("_", " ").title()

            logger.info(f"Extracted {wc} words from PDF: {url}")
            return CrawledPage(
                url=url,
                title=title,
                text_content=text,
                html_content=None,
                status_code=200,
                word_count=wc,
                scraped_at=datetime.utcnow(),
            )
        except Exception as e:
            logger.warning(f"PDF page fetch failed for {url}: {e}")
            return None

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        url: str,
        base_domain: str,
    ) -> Optional[CrawledPage]:
        # Route PDFs to dedicated handler
        clean_url = url.split("?")[0].split("#")[0].lower()
        if clean_url.endswith(".pdf"):
            return await self._fetch_pdf_page(client, url)

        html: Optional[str] = None
        used_playwright = False

        # ── Attempt 1: plain httpx ─────────────────────────────────────────
        for attempt in range(settings.CRAWL_MAX_RETRIES):
            try:
                resp = await client.get(url)
                logger.debug(f"HTTP {resp.status_code} — {url}")

                if resp.status_code == 403:
                    # Bot-blocked — Playwright uses a real browser fingerprint
                    logger.info(f"403 Forbidden: {url} — falling back to Playwright")
                    html = await _fetch_with_playwright(url)
                    used_playwright = True
                    break

                if resp.status_code == 429:
                    # Rate limited — honour Retry-After if present
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limited at {url} — waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status_code == 404:
                    logger.warning(f"404 Not Found: {url}")
                    return None

                if not (200 <= resp.status_code < 300):
                    logger.warning(f"HTTP {resp.status_code} for {url}")
                    if attempt < settings.CRAWL_MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    # Last resort: some sites (e.g. Cloudflare 503) serve real
                    # content to a full browser even on non-2xx responses
                    logger.info(f"All httpx attempts failed for {url} — trying Playwright")
                    html = await _fetch_with_playwright(url)
                    used_playwright = True
                    break

                # Detect PDFs served via document-manager URLs (no .pdf extension)
                content_type = resp.headers.get("content-type", "").lower()
                if "application/pdf" in content_type:
                    logger.info(f"PDF response detected by content-type at {url}")
                    text = _extract_pdf_text(resp.content)
                    if not text or word_count(text) < 10:
                        return None
                    filename = url.split("/")[-1].split("?")[0]
                    title = (filename.replace(".pdf", "").replace("-", " ")
                             .replace("_", " ").title()) or "Document"
                    wc = word_count(text)
                    logger.info(f"Extracted {wc} words from PDF: {url}")
                    return CrawledPage(
                        url=url,
                        title=title,
                        text_content=text,
                        html_content=None,
                        status_code=200,
                        word_count=wc,
                        scraped_at=datetime.utcnow(),
                    )

                html = resp.text
                break

            except httpx.TimeoutException:
                logger.warning(f"Timeout on {url}, attempt {attempt + 1}")
                if attempt == settings.CRAWL_MAX_RETRIES - 1:
                    logger.info(f"All httpx attempts timed out for {url} — trying Playwright")
                    html = await _fetch_with_playwright(url)
                    used_playwright = True
                else:
                    await asyncio.sleep(2 ** attempt)
            except httpx.ConnectError as e:
                logger.warning(f"Connect error on {url}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                return None

        if html is None:
            return None

        # ── Check if content is too thin (likely JS-rendered) ─────────────
        text = _extract_text(html)
        wc = word_count(text)

        if wc < MIN_WORD_THRESHOLD and not used_playwright:
            logger.info(f"Thin content ({wc} words) at {url} — trying Playwright JS render")
            js_html = await _fetch_with_playwright(url)
            if js_html:
                js_text = _extract_text(js_html)
                js_wc = word_count(js_text)
                if js_wc > wc:
                    logger.info(f"Playwright got {js_wc} words vs {wc} — using JS version")
                    html = js_html
                    text = js_text
                    wc = js_wc

        # ── Skip truly empty pages ─────────────────────────────────────────
        if wc < 10:
            logger.debug(f"Skipping near-empty page ({wc} words): {url}")
            return None

        return CrawledPage(
            url=url,
            title=_extract_title(html),
            text_content=text,
            html_content=html,
            status_code=200,
            word_count=wc,
            scraped_at=datetime.utcnow(),
        )


# Singleton
_crawler_service: Optional[CrawlerService] = None


def get_crawler_service() -> CrawlerService:
    global _crawler_service
    if _crawler_service is None:
        _crawler_service = CrawlerService()
    return _crawler_service
