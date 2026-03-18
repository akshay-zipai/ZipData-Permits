"""
RAG Pipeline — orchestrates crawl → index → retrieve → generate flow.
"""
import time
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.requests import RAGQueryRequest
from app.models.responses import RAGQueryResponse, RetrievedChunk
from app.services.crawling.crawler import get_crawler_service
from app.services.rag.retriever import get_retriever_service
from app.services.llm.generator import get_llm_service
from app.utils.permit_portals import get_portal_url
from app.utils.zip_lookup import zip_to_county
from app.utils.text_processing import truncate_text

logger = get_logger(__name__)
settings = get_settings()


def _load_prompt(filename: str) -> str:
    path = settings.PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    logger.warning(f"Prompt file not found: {path}")
    return ""


class RAGPipeline:
    def __init__(self):
        self._qa_system_prompt: Optional[str] = None
        self._rag_context_template: Optional[str] = None

    def _get_system_prompt(self) -> str:
        if self._qa_system_prompt is None:
            self._qa_system_prompt = _load_prompt("qa_system.txt")
        return self._qa_system_prompt

    def _get_context_template(self) -> str:
        if self._rag_context_template is None:
            self._rag_context_template = _load_prompt("rag_context.txt")
        return self._rag_context_template

    def _build_prompt(
        self, question: str, chunks: list[RetrievedChunk], county_name: Optional[str]
    ) -> str:
        template = self._get_context_template()
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.source_url or "unknown"
            context_parts.append(
                f"[Source {i} | {chunk.county_name or 'CA'} | {source}]\n{chunk.content}"
            )
        context_text = "\n\n---\n\n".join(context_parts)

        if template:
            return template.format(
                context=context_text,
                question=question,
                county=county_name or "California",
            )

        # Fallback inline template
        return (
            f"You are a California building permit expert.\n\n"
            f"Use the following retrieved context to answer the question.\n"
            f"If the context doesn't contain the answer, say so clearly.\n\n"
            f"County: {county_name or 'California'}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

    async def answer(self, request: RAGQueryRequest) -> RAGQueryResponse:
        overall_start = time.perf_counter()

        # Resolve county name
        county_name = request.county_name
        if not county_name and request.zip_code:
            county_name = zip_to_county(request.zip_code)
            if not county_name:
                logger.warning(f"No county found for ZIP {request.zip_code}")

        # Retrieve
        retrieval_start = time.perf_counter()
        retriever = get_retriever_service()
        chunks = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
            county_filter=county_name,
        )

        # If no results and we have a county, auto-crawl and index
        if not chunks and county_name and settings.ENABLE_LOCAL_RAG and settings.ENABLE_AUTO_CRAWL:
            logger.info(f"No indexed content for {county_name}. Auto-crawling...")
            chunks = await self._auto_crawl_and_retrieve(
                county_name, request.question, request.top_k
            )

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Generate
        generation_start = time.perf_counter()
        llm = get_llm_service()
        prompt = self._build_prompt(request.question, chunks, county_name)
        system_prompt = self._get_system_prompt()

        llm_response = await llm.generate(
            prompt=prompt,
            system_prompt=system_prompt or None,
        )
        generation_ms = (time.perf_counter() - generation_start) * 1000
        total_ms = (time.perf_counter() - overall_start) * 1000

        sources = chunks if request.include_sources else []

        return RAGQueryResponse(
            question=request.question,
            answer=llm_response.text,
            sources=sources,
            county_name=county_name,
            zip_code=request.zip_code,
            retrieval_time_ms=round(retrieval_ms, 2),
            generation_time_ms=round(generation_ms, 2),
            total_time_ms=round(total_ms, 2),
        )

    async def _auto_crawl_and_retrieve(
        self, county_name: str, question: str, top_k: int
    ) -> list[RetrievedChunk]:
        portal_url = get_portal_url(county_name)
        if not portal_url:
            logger.warning(f"No portal URL for {county_name}")
            return []

        crawler = get_crawler_service()
        try:
            crawl_result = await crawler.crawl_url(portal_url, county_name)
        except Exception as e:
            logger.error(f"Auto-crawl failed for {county_name}: {e}")
            return []

        retriever = get_retriever_service()

        # Clear stale indexed data for this county so fresh PDF content
        # replaces old HTML-only chunks rather than sitting alongside them.
        try:
            col = retriever._get_collection()
            col.delete(where={"county_name": county_name})
            retriever._bm25_corpus = [
                c for c, m in zip(retriever._bm25_corpus, retriever._bm25_metadata)
                if m.get("county_name") != county_name
            ]
            retriever._bm25_metadata = [
                m for m in retriever._bm25_metadata
                if m.get("county_name") != county_name
            ]
            retriever._rebuild_bm25()
            logger.info(f"Cleared stale index for {county_name} before re-indexing")
        except Exception as e:
            logger.warning(f"Could not clear stale index for {county_name}: {e}")

        for page in crawl_result.pages:
            if page.text_content:
                retriever.index_content(
                    content=page.text_content,
                    county_name=county_name,
                    source_url=page.url,
                )

        return retriever.retrieve(question, top_k=top_k, county_filter=county_name)


# Singleton
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
