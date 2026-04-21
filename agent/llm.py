"""
LLM abstraction layer.

  local       → AsyncOpenAI chat completions
  production  → boto3 bedrock-runtime Converse API  (async via run_in_executor)

Both expose the same interface:
    client = get_llm_client()
    text   = await client.generate(system=..., user=...)
    url    = await client.generate_image(prompt=...)
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
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


# ── OpenAI backend ────────────────────────────────────────────────────────────

class OpenAIClient(LLMClient):
    def __init__(self):
        from openai import AsyncOpenAI
        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file for local mode."
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


# ── Bedrock backend ───────────────────────────────────────────────────────────

class BedrockClient(LLMClient):
    """
    Uses the Bedrock Converse API — works with Claude, Nova, Llama, Mistral, etc.
    Image generation defaults to DALL-E (via OpenAI) unless BEDROCK_IMAGE=true,
    in which case Amazon Titan Image Generator v2 is used.
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

        # Fallback image client (OpenAI DALL-E) when BEDROCK_IMAGE=false
        self._openai_image_client = None
        if settings.GENERATE_IMAGES and not settings.BEDROCK_IMAGE:
            if settings.OPENAI_API_KEY:
                from openai import AsyncOpenAI
                self._openai_image_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    def _call_converse(self, system: str, user: str) -> str:
        """Synchronous Bedrock Converse call (run in executor for async)."""
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

    def _call_titan_image(self, prompt: str) -> Optional[str]:
        """Synchronous Titan image generation — returns a data-URI string."""
        payload = json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": prompt},
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

        if settings.BEDROCK_IMAGE:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._call_titan_image, prompt)

        # Use OpenAI DALL-E even in prod (simpler, higher quality)
        if self._openai_image_client:
            resp = await self._openai_image_client.images.generate(
                model=settings.OPENAI_IMAGE_MODEL,
                prompt=prompt,
                size=settings.IMAGE_SIZE,
                quality=settings.IMAGE_QUALITY,
                n=1,
            )
            return resp.data[0].url

        return None


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
