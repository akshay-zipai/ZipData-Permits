"""
RAG Pipeline — orchestrates crawl → index → retrieve → generate flow.
"""
import re
import time
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
from app.utils.text_processing import chunk_text, clean_text

logger = get_logger(__name__)
settings = get_settings()


def _normalize_optional_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() in {"string", "none", "null"}:
        return None
    return normalized


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

    async def retrieve_context(
        self,
        question: str,
        county_name: Optional[str],
        top_k: int,
    ) -> list[RetrievedChunk]:
        logger.info(
            "Retrieval start county=%s top_k=%s local_rag=%s auto_crawl=%s question=%s",
            county_name,
            top_k,
            settings.ENABLE_LOCAL_RAG,
            settings.ENABLE_AUTO_CRAWL,
            question[:160],
        )
        if settings.ENABLE_LOCAL_RAG:
            retriever = get_retriever_service()
            chunks = retriever.retrieve(
                query=question,
                top_k=top_k,
                county_filter=county_name,
            )
            if not chunks and county_name and settings.ENABLE_AUTO_CRAWL:
                logger.info(f"No indexed content for {county_name}. Auto-crawling...")
                chunks = await self._auto_crawl_and_retrieve(
                    county_name, question, top_k
                )
            return chunks

        if county_name and settings.ENABLE_AUTO_CRAWL:
            logger.info("Local RAG disabled; using on-demand crawl retrieval for %s", county_name)
            return await self._crawl_and_rank(
                county_name=county_name,
                question=question,
                top_k=top_k,
            )

        return []

    def _score_chunk(self, question: str, chunk: str) -> float:
        question_terms = re.findall(r"\w+", question.lower())
        chunk_terms = re.findall(r"\w+", chunk.lower())
        if not question_terms or not chunk_terms:
            return 0.0

        chunk_term_set = set(chunk_terms)
        overlap = sum(1 for term in question_terms if term in chunk_term_set)
        phrase_bonus = 1 if question.lower() in chunk.lower() else 0
        return float(overlap + phrase_bonus)

    async def _crawl_and_rank(
        self,
        county_name: str,
        question: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        portal_url = get_portal_url(county_name)
        if not portal_url:
            logger.warning(f"No portal URL for {county_name}")
            return []

        crawler = get_crawler_service()
        try:
            crawl_result = await crawler.crawl_url(portal_url, county_name)
        except Exception as e:
            logger.error(f"On-demand crawl failed for {county_name}: {e}")
            return []

        logger.info(
            "On-demand crawl summary county=%s pages=%s total_words=%s portal=%s",
            county_name,
            crawl_result.total_pages,
            crawl_result.total_words,
            portal_url,
        )

        ranked: list[RetrievedChunk] = []
        for page in crawl_result.pages:
            if not page.text_content:
                continue
            for idx, chunk in enumerate(chunk_text(clean_text(page.text_content))):
                score = self._score_chunk(question, chunk)
                if score <= 0:
                    continue
                ranked.append(
                    RetrievedChunk(
                        content=chunk,
                        source_url=page.url,
                        county_name=county_name,
                        score=round(score, 4),
                        retrieval_method="crawl",
                        chunk_id=f"{page.url}#{idx}",
                        metadata={"title": page.title, "word_count": page.word_count},
                    )
                )

        ranked.sort(key=lambda item: item.score, reverse=True)
        logger.info(
            "On-demand crawl retrieval produced %s relevant chunks for %s",
            len(ranked),
            county_name,
        )
        selected = ranked[:top_k]
        if selected:
            for idx, chunk in enumerate(selected, 1):
                logger.info(
                    "Selected chunk %s county=%s score=%.4f source=%s preview=%s",
                    idx,
                    county_name,
                    chunk.score,
                    chunk.source_url or "unknown",
                    chunk.content[:220].replace("\n", " "),
                )
        else:
            logger.warning(
                "No relevant chunks selected for county=%s question=%s",
                county_name,
                question[:160],
            )
        return selected

    async def answer(self, request: RAGQueryRequest) -> RAGQueryResponse:
        overall_start = time.perf_counter()

        # Resolve county name
        county_name = _normalize_optional_value(request.county_name)
        if not county_name and request.zip_code:
            county_name = zip_to_county(request.zip_code)
            if not county_name:
                logger.warning(f"No county found for ZIP {request.zip_code}")

        # Retrieve
        retrieval_start = time.perf_counter()
        chunks = await self.retrieve_context(
            question=request.question,
            county_name=county_name,
            top_k=request.top_k,
        )

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Generate
        generation_start = time.perf_counter()
        llm = get_llm_service()
        prompt = self._build_prompt(request.question, chunks, county_name)
        system_prompt = self._get_system_prompt()
        logger.info(
            "Prompt assembled county=%s chunks=%s context_chars=%s system_prompt_chars=%s",
            county_name,
            len(chunks),
            len(prompt),
            len(system_prompt or ""),
        )

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
