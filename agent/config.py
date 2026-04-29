"""
Unified Agent Configuration.

Backend selection (set via ENVIRONMENT in .env):
  - local       → OpenAI GPT (OPENAI_API_KEY + OPENAI_MODEL)
  - production  → AWS Bedrock (BEDROCK_REGION + BEDROCK_MODEL_ID via boto3)

Image generation:
  - local       → DALL-E 3 (OpenAI)
  - production  → DALL-E 3 by default; set BEDROCK_IMAGE=true to use
                  an AWS Bedrock image model such as Amazon Nova Canvas
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "CA Permit & Renovation Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # "local" → OpenAI   |   "production" → Bedrock
    ENVIRONMENT: str = "local"

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    ALLOWED_ORIGINS: list[str] = ["*"]

    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROMPTS_DIR: Path = BASE_DIR / "prompts"

    # ── OpenAI  (ENVIRONMENT=local) ───────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_IMAGE_MODEL: str = "dall-e-3"

    # ── AWS Bedrock  (ENVIRONMENT=production) ─────────────────────────────────
    BEDROCK_REGION: str = "us-east-1"

    # Default: Amazon Nova Lite — no use-case form required.
    # Other options: amazon.nova-pro-v1:0, amazon.nova-micro-v1:0,
    #   meta.llama3-8b-instruct-v1:0, mistral.mistral-7b-instruct-v0:2
    BEDROCK_MODEL_ID: str = "amazon.nova-lite-v1:0"

    # Set true to use an AWS Bedrock image model for images in prod
    BEDROCK_IMAGE: bool = False
    BEDROCK_IMAGE_MODEL_ID: str = "amazon.nova-canvas-v1:0"

    # AWS credentials — leave blank to use IAM role / AWS_* env vars / ~/.aws
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None

    # ── RAG ───────────────────────────────────────────────────────────────────
    RAG_TOP_K: int = 5

    # ── Renovation ────────────────────────────────────────────────────────────
    MAX_SUGGESTIONS: int = 4
    GENERATE_IMAGES: bool = True
    IMAGE_SIZE: str = "1024x1024"
    IMAGE_QUALITY: str = "standard"
    S3_RENOVATION_BUCKET: Optional[str] = None
    S3_RENOVATION_PREFIX: str = "renovation-collages"
    S3_RENOVATION_INDEX_KEY: str = "renovation-collages/index.json"
    S3_URL_EXPIRY_SECONDS: int = 3600
    S3_REGION: Optional[str] = None

    # ── AWS DocumentDB ────────────────────────────────────────────────────────
    DOCUMENTDB_ENABLED: bool = False
    DOCUMENTDB_URI: Optional[str] = None
    DOCUMENTDB_CLUSTER_ID: Optional[str] = None
    DOCUMENTDB_HOST: Optional[str] = None
    DOCUMENTDB_PORT: int = 27017
    DOCUMENTDB_USERNAME: Optional[str] = None
    DOCUMENTDB_PASSWORD: Optional[str] = None
    DOCUMENTDB_DATABASE: str = "zipai_permits"
    DOCUMENTDB_QUESTIONS_COLLECTION: str = "dataset_questions"
    DOCUMENTDB_SESSIONS_COLLECTION: str = "conversation_sessions"
    DOCUMENTDB_TLS: bool = True
    DOCUMENTDB_TLS_CA_FILE: Optional[str] = None
    DOCUMENTDB_REPLICA_SET: str = "rs0"
    DOCUMENTDB_READ_PREFERENCE: str = "secondaryPreferred"
    DOCUMENTDB_RETRY_WRITES: bool = False
    DOCUMENTDB_DIRECT_CONNECTION: bool = False
    DOCUMENTDB_CONNECT_TIMEOUT_MS: int = 10000
    DOCUMENTDB_SERVER_SELECTION_TIMEOUT_MS: int = 10000

    # ── LLM shared ────────────────────────────────────────────────────────────
    # 4096 is required — 5 suggestions with image_prompts easily exceed 1024 tokens
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.3

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def documentdb_configured(self) -> bool:
        if self.DOCUMENTDB_URI:
            return True
        return bool(
            self.DOCUMENTDB_HOST
            and self.DOCUMENTDB_USERNAME
            and self.DOCUMENTDB_PASSWORD
        )

    def build_documentdb_uri(self, host: str) -> str:
        if self.DOCUMENTDB_URI:
            return self.DOCUMENTDB_URI

        if not self.DOCUMENTDB_USERNAME or not self.DOCUMENTDB_PASSWORD:
            raise ValueError("DocumentDB username/password are required to build the URI.")

        username = quote_plus(self.DOCUMENTDB_USERNAME)
        password = quote_plus(self.DOCUMENTDB_PASSWORD)
        tls_value = "true" if self.DOCUMENTDB_TLS else "false"
        retry_writes = "true" if self.DOCUMENTDB_RETRY_WRITES else "false"
        direct_connection = "true" if self.DOCUMENTDB_DIRECT_CONNECTION else "false"

        return (
            f"mongodb://{username}:{password}@{host}:{self.DOCUMENTDB_PORT}/"
            f"?tls={tls_value}"
            f"&replicaSet={self.DOCUMENTDB_REPLICA_SET}"
            f"&readPreference={self.DOCUMENTDB_READ_PREFERENCE}"
            f"&retryWrites={retry_writes}"
            f"&directConnection={direct_connection}"
        )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
