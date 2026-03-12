from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime


# ── Generic ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Crawling ──────────────────────────────────────────────────────────────────

class CrawledPage(BaseModel):
    url: str
    title: Optional[str]
    text_content: str
    html_content: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    status_code: int = 200
    word_count: int = 0


class CrawlResponse(BaseModel):
    county_name: str
    portal_url: str
    pages: list[CrawledPage]
    total_pages: int
    total_words: int
    crawl_duration_seconds: float
    cached: bool = False


# ── Embedding ─────────────────────────────────────────────────────────────────

class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int
    count: int


# ── RAG ───────────────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    content: str
    source_url: Optional[str]
    county_name: Optional[str]
    score: float
    retrieval_method: str  # "vector" | "bm25" | "hybrid"
    chunk_id: str
    metadata: Optional[dict] = None


class IndexResponse(BaseModel):
    county_name: str
    chunks_indexed: int
    collection_name: str
    success: bool


class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[RetrievedChunk]
    county_name: Optional[str]
    zip_code: Optional[str]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float


# ── LLM ───────────────────────────────────────────────────────────────────────

class LLMResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    generation_time_ms: float


# ── WebSocket ─────────────────────────────────────────────────────────────────

class WSMessage(BaseModel):
    type: str  # "answer" | "error" | "status" | "stream_chunk"
    content: Any
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
