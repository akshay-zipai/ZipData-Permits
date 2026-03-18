from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "CA Permit RAG System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"  # "dev" | "production"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # CORS
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Data paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    PROMPTS_DIR: Path = BASE_DIR / "prompts"
    PERMIT_PORTALS_PATH: Path = DATA_DIR / "permit_portals.json"

    # Crawling
    CRAWL_TIMEOUT_SECONDS: int = 30
    CRAWL_MAX_RETRIES: int = 3
    CRAWL_DELAY_SECONDS: float = 1.0
    CRAWL_USER_AGENT: str = (
        "Mozilla/5.0 (compatible; PermitRAGBot/1.0; +https://github.com/your-org/ca-permit-rag)"
    )
    CRAWL_MAX_PAGES_PER_DOMAIN: int = 20

    # Embedding Service
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L12-v2"
    EMBEDDING_DEVICE: str = "cpu"  # "cpu" | "cuda" | "mps"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_MAX_LENGTH: int = 512

    # LLM Service
    LLM_BACKEND: str = "ollama"  # Used outside production: "ollama" | "huggingface" | "openai"
    LLM_MODEL_NAME: str = "gemma3:4b"
    LLM_MAX_NEW_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    LLM_OLLAMA_BASE_URL: str = "http://ollama:11434"
    LLM_OLLAMA_TIMEOUT: int = 300  # seconds — PDF contexts need longer generation

    # Amazon Bedrock (used when ENVIRONMENT=production)
    BEDROCK_MODEL_ID: str = "google.gemma-3-12b-it"
    BEDROCK_REGION: str = "us-east-1"
    BEDROCK_READ_TIMEOUT: int = 300
    ENABLE_LOCAL_RAG: bool = True
    ENABLE_AUTO_CRAWL: bool = True

    # HuggingFace fallback (if LLM_BACKEND=huggingface)
    HF_MODEL_NAME: str = "google/gemma-3-4b-it"
    HF_TOKEN: Optional[str] = None
    HF_DEVICE: str = "cpu"

    # OpenAI fallback (if LLM_BACKEND=openai)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Vector DB (ChromaDB)
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "permit_docs"

    # RAG
    RAG_TOP_K: int = 5
    RAG_BM25_WEIGHT: float = 0.4
    RAG_VECTOR_WEIGHT: float = 0.6
    RAG_CHUNK_SIZE: int = 512
    RAG_CHUNK_OVERLAP: int = 64

    # WebSocket
    WS_PING_INTERVAL: int = 30
    WS_MAX_MESSAGE_SIZE: int = 65536

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
