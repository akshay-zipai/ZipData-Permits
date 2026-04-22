"""
LLM abstraction layer.

  local       → AsyncOpenAI chat completions + DALL-E 3 images
  production  → AWS Bedrock Converse API for text
                Images: DALL-E 3 when OPENAI_API_KEY is available,
                        otherwise a Bedrock image model automatically.

Both expose the same interface:
    client = get_llm_client()
    text   = await client.generate(system=..., user=...)
    url    = await client.generate_image(prompt=...)   # returns data-URI
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
            response_format="b64_json",
        )
        b64 = resp.data[0].b64_json
        if not b64:
            raise RuntimeError("OpenAI image response did not include b64_json data.")
        return f"data:image/png;base64,{b64}"


# ── Bedrock backend (production) ──────────────────────────────────────────────

class BedrockClient(LLMClient):
    """
    Text  → Bedrock Converse API (any model: Nova, Llama, Mistral, etc.)
    Image → Auto-selects the best available backend:
              1. DALL-E 3 via OpenAI       (preferred whenever OPENAI_API_KEY is set)
              2. Bedrock image model       (fallback when no OpenAI key is available)
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

        # Prefer DALL-E 3 for image generation whenever an OpenAI key is available.
        self._use_openai_image = bool(settings.OPENAI_API_KEY)
        self._image_model_id = settings.BEDROCK_IMAGE_MODEL_ID

        # Optional OpenAI client for DALL-E fallback
        self._openai_img = None
        if self._use_openai_image:
            from openai import AsyncOpenAI
            self._openai_img = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        if not self._use_openai_image:
            self._validate_image_model_id(self._image_model_id)

        img_backend = "DALL-E 3 (OpenAI)" if self._use_openai_image else self._image_model_id
        print(f"[BedrockClient] text={self._model_id} | images={img_backend}")

    def _validate_image_model_id(self, model_id: str) -> None:
        lowered = model_id.lower()
        image_markers = ("titan-image", "stability.", "image", "nova-canvas")
        text_markers = (
            "anthropic.",
            "claude",
            "llama",
            "mistral",
            "nova-lite",
            "nova-micro",
            "nova-pro",
            "nova-premier",
            "nova-2-lite",
        )

        if any(marker in lowered for marker in image_markers):
            return

        if any(marker in lowered for marker in text_markers):
            raise RuntimeError(
                "BEDROCK_IMAGE_MODEL_ID is configured with a text model/profile, "
                f"which cannot generate images: {model_id!r}. "
                "When BEDROCK_IMAGE=true, set BEDROCK_IMAGE_MODEL_ID to an image model "
                "such as 'amazon.nova-canvas-v1:0' or an image inference profile ARN."
            )

        raise RuntimeError(
            "BEDROCK_IMAGE_MODEL_ID does not look like an image-capable Bedrock model/profile: "
            f"{model_id!r}. When BEDROCK_IMAGE=true, use an image model ID or image inference profile ARN."
        )

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

    def _call_bedrock_image(self, prompt: str) -> str:
        """Synchronous Bedrock image call — returns a data-URI (base64 PNG)."""
        safe_prompt = prompt[:1000]
        payload = json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": safe_prompt,
                "negativeText": "blurry, distorted, low quality, low resolution, text, watermark",
                "style": "PHOTOREALISM",
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "standard",
                "width": 1024,
                "height": 1024,
                "cfgScale": 8.0,
                "seed": 0,
            },
        })
        resp = self._br.invoke_model(
            modelId=self._image_model_id,
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

        if self._openai_img:
            resp = await self._openai_img.images.generate(
                model=settings.OPENAI_IMAGE_MODEL,
                prompt=prompt,
                size=settings.IMAGE_SIZE,
                quality=settings.IMAGE_QUALITY,
                n=1,
                response_format="b64_json",
            )
            b64 = resp.data[0].b64_json
            if not b64:
                raise RuntimeError("OpenAI image response did not include b64_json data.")
            return f"data:image/png;base64,{b64}"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call_bedrock_image, prompt)



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
