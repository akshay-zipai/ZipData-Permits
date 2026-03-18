from fastapi import APIRouter, HTTPException, Depends

from app.models.requests import RAGQueryRequest, IndexRequest
from app.models.responses import RAGQueryResponse, IndexResponse
from app.services.rag.pipeline import get_rag_pipeline, RAGPipeline
from app.services.rag.retriever import get_retriever_service, RetrieverService
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])


def _get_pipeline() -> RAGPipeline:
    return get_rag_pipeline()


def _get_retriever() -> RetrieverService:
    return get_retriever_service()


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Ask a permit question using hybrid RAG",
)
async def rag_query(
    request: RAGQueryRequest,
    pipeline: RAGPipeline = Depends(_get_pipeline),
):
    """
    Ask a permit-related question. The pipeline will:
    1. Resolve ZIP → county (if zip provided)
    2. Retrieve relevant context using hybrid BM25 + semantic search
    3. Auto-crawl the permit portal if no indexed content exists
    4. Generate an answer using the LLM
    """
    try:
        return await pipeline.answer(request)
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Manually index content into the vector DB",
)
async def index_content(
    request: IndexRequest,
    retriever: RetrieverService = Depends(_get_retriever),
):
    """Index raw text content for a county into the vector database."""
    from app.core.config import get_settings
    settings = get_settings()
    if not settings.ENABLE_LOCAL_RAG:
        raise HTTPException(
            status_code=503,
            detail="Local RAG indexing is disabled in this environment.",
        )
    try:
        chunks = retriever.index_content(
            content=request.content,
            county_name=request.county_name,
            source_url=request.source_url,
            extra_metadata=request.metadata,
        )
        return IndexResponse(
            county_name=request.county_name,
            chunks_indexed=chunks,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            success=True,
        )
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="Get vector DB and BM25 index statistics")
async def get_stats(retriever: RetrieverService = Depends(_get_retriever)):
    return retriever.get_collection_stats()


@router.delete("/index/{county_name}", summary="Clear indexed content for a county")
async def clear_county_index(
    county_name: str,
    retriever: RetrieverService = Depends(_get_retriever),
):
    """Delete all indexed documents for a given county."""
    try:
        col = retriever._get_collection()
        col.delete(where={"county_name": county_name})
        # Rebuild BM25 corpus without this county
        new_corpus = []
        new_meta = []
        for text, meta in zip(retriever._bm25_corpus, retriever._bm25_metadata):
            if meta.get("county_name") != county_name:
                new_corpus.append(text)
                new_meta.append(meta)
        retriever._bm25_corpus = new_corpus
        retriever._bm25_metadata = new_meta
        retriever._rebuild_bm25()
        return {"deleted": True, "county_name": county_name}
    except Exception as e:
        logger.error(f"Failed to clear index for {county_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
