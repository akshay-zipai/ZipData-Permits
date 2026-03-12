from pydantic import BaseModel, Field, HttpUrl
from typing import Optional


# ── Crawling ──────────────────────────────────────────────────────────────────

class CrawlByZipRequest(BaseModel):
    zip_code: str = Field(..., min_length=5, max_length=5, pattern=r"^\d{5}$", examples=["94102"])
    force_refresh: bool = Field(False, description="Re-crawl even if cached")


class CrawlByCountyRequest(BaseModel):
    county_name: str = Field(..., examples=["San Francisco County"])
    force_refresh: bool = False


class CrawlByUrlRequest(BaseModel):
    url: HttpUrl
    county_name: Optional[str] = None
    force_refresh: bool = False


# ── RAG ───────────────────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    county_name: str = Field(..., examples=["San Francisco County"])
    content: str = Field(..., description="Raw text content to index")
    source_url: Optional[str] = None
    metadata: Optional[dict] = None


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    zip_code: Optional[str] = Field(None, pattern=r"^\d{5}$")
    county_name: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)
    include_sources: bool = True


# ── LLM ───────────────────────────────────────────────────────────────────────

class LLMGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system_prompt: Optional[str] = None
    max_new_tokens: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    stream: bool = False


# ── WebSocket ─────────────────────────────────────────────────────────────────

class WSPermitQuestion(BaseModel):
    question: str
    zip_code: Optional[str] = None
    county_name: Optional[str] = None
    session_id: Optional[str] = None
