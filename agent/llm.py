"""
LLM abstraction layer.

  local       → AsyncOpenAI chat completions + DALL-E 3 images
  production  → AWS Bedrock Converse API for text
                Images: DALL-E 3 if OPENAI_API_KEY is set,
                        otherwise Amazon Titan Image Generator v2 automatically.
                        (BEDROCK_IMAGE=true forces Titan even if OpenAI key exists)

Both expose the same interface:
    client = get_llm_client()
    text   = await client.generate(system=..., user=...)
    url    = await client.generate_image(prompt=...)   # returns URL or data-URI
"""
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Optional

from agent.config import get_settings

settings = get_settings()


# ── Abstract base ─────────────────────────────────────────────────────────────

class LLMClient(ABC):
    @abstractmethod
    async def generate(self, system: str, user: str) -> str: ...

    @abstractmethod
    async def generate_image(self, prompt: str) -> Optional[str]: ...


# ── OpenAI backend (local) ────────────────────────────────────────────────────

class OpenAIClient(LLMClient):
    def __init__(self):
        from openai import AsyncOpenAI
        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env for local mode."
            )
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate(self, system: str, user: str) -> str:
        resp = await self._client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        return resp.choices[0].message.content.strip()

    async def generate_image(self, prompt: str) -> Optional[str]:
        if not settings.GENERATE_IMAGES:
            return None
        resp = await self._client.images.generate(
            model=settings.OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size=settings.IMAGE_SIZE,
            quality=settings.IMAGE_QUALITY,
            n=1,
        )
        return resp.data[0].url


# ── Bedrock backend (production) ──────────────────────────────────────────────

class BedrockClient(LLMClient):
    """
    Text  → Bedrock Converse API (any model: Nova, Llama, Mistral, etc.)
    Image → Auto-selects the best available backend:
              1. Titan Image Generator v2  (if BEDROCK_IMAGE=true OR no OpenAI key)
              2. DALL-E 3 via OpenAI       (if OPENAI_API_KEY is set and BEDROCK_IMAGE=false)
    """

    def __init__(self):
        import boto3

        kwargs: dict = {"region_name": settings.BEDROCK_REGION}
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
            if settings.AWS_SESSION_TOKEN:
                kwargs["aws_session_token"] = settings.AWS_SESSION_TOKEN

        self._br = boto3.client("bedrock-runtime", **kwargs)
        self._model_id = settings.BEDROCK_MODEL_ID

        # Decide image backend once at init so we log it clearly
        self._use_titan = settings.BEDROCK_IMAGE or not settings.OPENAI_API_KEY

        # Optional OpenAI client for DALL-E fallback
        self._openai_img = None
        if not self._use_titan and settings.OPENAI_API_KEY:
            from openai import AsyncOpenAI
            self._openai_img = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        img_backend = "Titan Image Generator v2" if self._use_titan else "DALL-E 3 (OpenAI)"
        print(f"[BedrockClient] text={self._model_id} | images={img_backend}")

    # ── Text generation ───────────────────────────────────────────────────────

    def _call_converse(self, system: str, user: str) -> str:
        body: dict = {
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {
                "maxTokens": settings.LLM_MAX_TOKENS,
                "temperature": settings.LLM_TEMPERATURE,
            },
        }
        if system:
            body["system"] = [{"text": system}]
        resp = self._br.converse(modelId=self._model_id, **body)
        return resp["output"]["message"]["content"][0]["text"].strip()

    async def generate(self, system: str, user: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call_converse, system, user)

    # ── Image generation ──────────────────────────────────────────────────────

    def _call_titan(self, prompt: str) -> str:
        """Synchronous Titan call — returns a data-URI (base64 PNG)."""
        # Titan has a 512-char prompt limit — truncate safely
        safe_prompt = prompt[:500]
        payload = json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": safe_prompt},
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "standard",
                "width": 1024,
                "height": 1024,
            },
        })
        resp = self._br.invoke_model(
            modelId=settings.BEDROCK_IMAGE_MODEL_ID,
            body=payload,
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(resp["body"].read())
        b64 = body["images"][0]
        return f"data:image/png;base64,{b64}"

    async def generate_image(self, prompt: str) -> Optional[str]:
        if not settings.GENERATE_IMAGES:
            return None

        if self._use_titan:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._call_titan, prompt)

        # DALL-E path
        if self._openai_img:
            resp = await self._openai_img.images.generate(
                model=settings.OPENAI_IMAGE_MODEL,
                prompt=prompt,
                size=settings.IMAGE_SIZE,
                quality=settings.IMAGE_QUALITY,
                n=1,
            )
            return resp.data[0].url

        # Should never reach here given init logic, but be explicit
        raise RuntimeError(
            "No image backend available. "
            "Set OPENAI_API_KEY for DALL-E or enable Titan in AWS Bedrock model access."
        )


# ── Factory ───────────────────────────────────────────────────────────────────

_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        if settings.is_production:
            _client = BedrockClient()
        else:
            _client = OpenAIClient()
    return _client