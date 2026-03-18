from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from app.models.requests import CrawlByZipRequest, CrawlByCountyRequest, CrawlByUrlRequest
from app.models.responses import CrawlResponse
from app.services.crawling.crawler import get_crawler_service, CrawlerService
from app.services.rag.retriever import get_retriever_service, RetrieverService
from app.utils.permit_portals import get_portal_url, list_counties, get_all_portals
from app.utils.zip_lookup import zip_to_county
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/crawl", tags=["Crawling"])


def _get_crawler() -> CrawlerService:
    return get_crawler_service()


def _get_retriever() -> RetrieverService:
    return get_retriever_service()


@router.post("/by-zip", response_model=CrawlResponse, summary="Crawl permit portal by ZIP code")
async def crawl_by_zip(
    request: CrawlByZipRequest,
    auto_index: bool = True,
    crawler: CrawlerService = Depends(_get_crawler),
    retriever: RetrieverService = Depends(_get_retriever),
):
    if auto_index and not settings.ENABLE_LOCAL_RAG:
        raise HTTPException(
            status_code=503,
            detail="Auto-indexing is disabled because local RAG is disabled in this environment.",
        )
    county_name = zip_to_county(request.zip_code)
    if not county_name:
        raise HTTPException(
            status_code=404,
            detail=f"No California county found for ZIP {request.zip_code}.",
        )

    portal_url = get_portal_url(county_name)
    if not portal_url:
        raise HTTPException(
            status_code=404,
            detail=f"No permit portal registered for {county_name}",
        )

    result = await crawler.crawl_url(portal_url, county_name, request.force_refresh)

    if auto_index:
        indexed = 0
        for page in result.pages:
            if page.text_content and page.word_count > 20:
                indexed += retriever.index_content(
                    content=page.text_content,
                    county_name=county_name,
                    source_url=page.url,
                )
        logger.info(f"Auto-indexed {indexed} chunks for {county_name}")

    return result


@router.post("/by-county", response_model=CrawlResponse, summary="Crawl permit portal by county name")
async def crawl_by_county(
    request: CrawlByCountyRequest,
    auto_index: bool = True,
    crawler: CrawlerService = Depends(_get_crawler),
    retriever: RetrieverService = Depends(_get_retriever),
):
    if auto_index and not settings.ENABLE_LOCAL_RAG:
        raise HTTPException(
            status_code=503,
            detail="Auto-indexing is disabled because local RAG is disabled in this environment.",
        )
    portal_url = get_portal_url(request.county_name)
    if not portal_url:
        raise HTTPException(
            status_code=404,
            detail=f"No permit portal found for '{request.county_name}'.",
        )

    result = await crawler.crawl_url(portal_url, request.county_name, request.force_refresh)

    if auto_index:
        indexed = 0
        for page in result.pages:
            if page.text_content and page.word_count > 20:
                indexed += retriever.index_content(
                    content=page.text_content,
                    county_name=request.county_name,
                    source_url=page.url,
                )
        logger.info(f"Auto-indexed {indexed} chunks for {request.county_name}")

    return result


@router.post("/by-url", response_model=CrawlResponse, summary="Crawl an arbitrary URL")
async def crawl_by_url(
    request: CrawlByUrlRequest,
    auto_index: bool = False,
    crawler: CrawlerService = Depends(_get_crawler),
    retriever: RetrieverService = Depends(_get_retriever),
):
    if auto_index and not settings.ENABLE_LOCAL_RAG:
        raise HTTPException(
            status_code=503,
            detail="Auto-indexing is disabled because local RAG is disabled in this environment.",
        )
    county_name = request.county_name or "Unknown County"
    result = await crawler.crawl_url(str(request.url), county_name, request.force_refresh)

    if auto_index:
        for page in result.pages:
            if page.text_content and page.word_count > 20:
                retriever.index_content(
                    content=page.text_content,
                    county_name=county_name,
                    source_url=page.url,
                )

    return result


@router.get("/debug/{county_name:path}", summary="Debug — show raw crawl result with text preview")
async def debug_crawl(
    county_name: str,
    crawler: CrawlerService = Depends(_get_crawler),
):
    """
    Crawls a county portal and returns a detailed debug report:
    - HTTP status per page
    - Word count per page
    - First 500 chars of extracted text
    Use this to diagnose why indexing returns 0 chunks.
    """
    portal_url = get_portal_url(county_name)
    if not portal_url:
        raise HTTPException(status_code=404, detail=f"No portal for '{county_name}'")

    result = await crawler.crawl_url(portal_url, county_name, force_refresh=True)

    debug_pages = []
    for page in result.pages:
        debug_pages.append({
            "url": page.url,
            "status_code": page.status_code,
            "title": page.title,
            "word_count": page.word_count,
            "text_preview": page.text_content[:500] if page.text_content else "",
        })

    return {
        "county_name": county_name,
        "portal_url": portal_url,
        "total_pages": result.total_pages,
        "total_words": result.total_words,
        "crawl_duration_seconds": result.crawl_duration_seconds,
        "pages": debug_pages,
        "diagnosis": _diagnose(result.total_pages, result.total_words),
    }


def _diagnose(total_pages: int, total_words: int) -> str:
    if total_pages == 0:
        return (
            "NO PAGES SCRAPED. Likely causes: "
            "(1) Site requires JavaScript — install playwright in container. "
            "(2) Bot detection / firewall blocking requests. "
            "(3) Site is down or URL changed. "
            "Try: POST /crawl/manual-index to inject text directly."
        )
    if total_words < 100:
        return (
            f"Pages scraped ({total_pages}) but very little text ({total_words} words). "
            "Content may be JS-rendered. Install playwright for better results."
        )
    return f"OK — {total_pages} pages, {total_words} words scraped successfully."


@router.post("/manual-index", summary="Manually paste permit info text to index")
async def manual_index(
    county_name: str,
    text: str,
    source_url: Optional[str] = None,
    retriever: RetrieverService = Depends(_get_retriever),
):
    """
    Bypass crawling entirely — paste raw permit text directly.
    Useful when the portal blocks scrapers.
    """
    if not settings.ENABLE_LOCAL_RAG:
        raise HTTPException(
            status_code=503,
            detail="Manual indexing is disabled because local RAG is disabled in this environment.",
        )
    if len(text.split()) < 10:
        raise HTTPException(status_code=400, detail="Text too short (min 10 words)")

    chunks = retriever.index_content(
        content=text,
        county_name=county_name,
        source_url=source_url or f"manual:{county_name}",
    )
    return {
        "county_name": county_name,
        "chunks_indexed": chunks,
        "message": f"Successfully indexed {chunks} chunks for {county_name}",
    }


@router.get("/counties", summary="List all registered CA counties and portal URLs")
async def list_all_counties():
    portals = get_all_portals()
    return {"counties": portals, "total": len(portals)}
