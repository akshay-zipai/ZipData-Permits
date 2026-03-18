"""
California Permit RAG System — FastAPI Application Entry Point
"""
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.api.routes import crawl, rag, llm, websocket

# ── Startup ───────────────────────────────────────────────────────────────────

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"LLM backend: {settings.LLM_BACKEND} | model: {settings.LLM_MODEL_NAME}")
    logger.info(f"Embedding model: {settings.EMBEDDING_MODEL_NAME}")
    logger.info(
        f"Local RAG enabled: {settings.ENABLE_LOCAL_RAG} | auto-crawl enabled: {settings.ENABLE_AUTO_CRAWL}"
    )

    # Ensure output directories exist
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    yield

    logger.info("Shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "A production-ready RAG system for answering California building "
            "permit questions. Combines web crawling, hybrid BM25 + vector "
            "retrieval, and LLM generation."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Routers ────────────────────────────────────────────────────────────────
    API_PREFIX = "/api/v1"
    app.include_router(crawl.router, prefix=API_PREFIX)
    app.include_router(rag.router, prefix=API_PREFIX)
    app.include_router(llm.router, prefix=API_PREFIX)
    app.include_router(websocket.router)  # WS routes at root level

    # ── Health / meta endpoints ────────────────────────────────────────────────

    @app.get("/health", tags=["Health"])
    async def health():
        return {
            "status": "ok",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @app.get("/", tags=["Health"])
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": "/health",
        }

    # ── Global exception handler ───────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    return app


app = create_app()
