from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Environment ───────────────────────────────────────────────────────────
    environment: Literal["dev", "prod"] = "dev"

    # ── OpenAI (dev) ──────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # ── AWS Bedrock (prod) ────────────────────────────────────────────────────
    # On EC2: attach an IAM role with bedrock:InvokeModel — no keys needed
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "google.gemma-3-12b-it"

    # ── Image generation (always DALL-E) ──────────────────────────────────────
    openai_image_api_key: str = ""  # falls back to openai_api_key if blank
    openai_image_model: str = "dall-e-3"
    image_size: str = "1024x1024"
    image_quality: str = "standard"

    # ── General ───────────────────────────────────────────────────────────────
    max_suggestions: int = 5

    @property
    def llm_provider(self) -> str:
        return "bedrock" if self.environment == "prod" else "openai"

    def image_key(self) -> str:
        return self.openai_image_api_key or self.openai_api_key

    class Config:
        env_file = ".env"


settings = Settings()