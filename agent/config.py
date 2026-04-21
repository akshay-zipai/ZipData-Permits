"""
Unified Agent Configuration.

Backend selection (set via ENVIRONMENT in .env):
  - local       → OpenAI GPT (OPENAI_API_KEY + OPENAI_MODEL)
  - production  → AWS Bedrock (BEDROCK_REGION + BEDROCK_MODEL_ID via boto3)

Image generation:
  - local       → DALL-E 3 (OpenAI)
  - production  → DALL-E 3 by default; set BEDROCK_IMAGE=true to use
                  Amazon Titan Image Generator instead
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache
from pathlib import Path
from typing import Optional


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

    # Default: Amazon Nova Lite — no use-case form required, available
    # immediately after enabling model access in the Bedrock console.
    # Other no-form-required options:
    #   amazon.nova-pro-v1:0          (higher quality, slightly more expensive)
    #   amazon.nova-micro-v1:0        (fastest, text-only)
    #   meta.llama3-8b-instruct-v1:0  (open-source)
    #   mistral.mistral-7b-instruct-v0:2
    BEDROCK_MODEL_ID: str = "amazon.nova-lite-v1:0"

    # Set true to use Amazon Titan Image Generator for images in prod
    # (requires Bedrock image model access in your AWS account)
    BEDROCK_IMAGE: bool = False
    BEDROCK_IMAGE_MODEL_ID: str = "amazon.titan-image-generator-v2:0"

    # AWS credentials — leave blank to use IAM role / AWS_* env vars / ~/.aws
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None

    # ── RAG ───────────────────────────────────────────────────────────────────
    RAG_TOP_K: int = 5

    # ── Renovation ────────────────────────────────────────────────────────────
    MAX_SUGGESTIONS: int = 5
    GENERATE_IMAGES: bool = True
    IMAGE_SIZE: str = "1024x1024"
    IMAGE_QUALITY: str = "standard"

    # ── LLM shared ────────────────────────────────────────────────────────────
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.3

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings()