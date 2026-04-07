"""
LLM Generation Service.
When ENVIRONMENT=production, Bedrock is used automatically.
Outside production, backend is selected by LLM_BACKEND: "ollama" | "huggingface" | "openai".
"""
import asyncio
import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.responses import LLMResponse

logger = get_logger(__name__)
settings = get_settings()


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseLLMBackend(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_new_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_new_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        ...


# ── Ollama backend ────────────────────────────────────────────────────────────

class OllamaBackend(BaseLLMBackend):
    def __init__(self):
        self.base_url = settings.LLM_OLLAMA_BASE_URL
        self.model = settings.LLM_MODEL_NAME

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = settings.LLM_MAX_NEW_TOKENS,
        temperature: float = settings.LLM_TEMPERATURE,
    ) -> LLMResponse:
        import httpx

        start = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": self._build_messages(prompt, system_prompt),
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                "top_p": settings.LLM_TOP_P,
            },
        }

        async with httpx.AsyncClient(timeout=settings.LLM_OLLAMA_TIMEOUT) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat", json=payload
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed_ms = (time.perf_counter() - start) * 1000
        message = data.get("message", {})
        usage = data.get("usage", {})

        return LLMResponse(
            text=message.get("content", ""),
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            generation_time_ms=round(elapsed_ms, 2),
        )

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = settings.LLM_MAX_NEW_TOKENS,
        temperature: float = settings.LLM_TEMPERATURE,
    ) -> AsyncGenerator[str, None]:
        import httpx
        import json

        payload = {
            "model": self.model,
            "messages": self._build_messages(prompt, system_prompt),
            "stream": True,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
            },
        }

        async with httpx.AsyncClient(timeout=settings.LLM_OLLAMA_TIMEOUT) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue


# ── Amazon Bedrock backend ────────────────────────────────────────────────────

class BedrockBackend(BaseLLMBackend):
    def __init__(self):
        import boto3
        from botocore.config import Config

        self.model = settings.BEDROCK_MODEL_ID
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.BEDROCK_REGION,
            config=Config(
                read_timeout=settings.BEDROCK_READ_TIMEOUT,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    def _build_messages(self, prompt: str) -> list[dict]:
        return [{"role": "user", "content": [{"text": prompt}]}]

    def _build_system(self, system_prompt: Optional[str]) -> list[dict]:
        if not system_prompt:
            return []
        return [{"text": system_prompt}]

    @staticmethod
    def _extract_text(message: dict) -> str:
        content = message.get("content", [])
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = settings.LLM_MAX_NEW_TOKENS,
        temperature: float = settings.LLM_TEMPERATURE,
    ) -> LLMResponse:
        start = time.perf_counter()

        def _invoke():
            return self.client.converse(
                modelId=self.model,
                messages=self._build_messages(prompt),
                system=self._build_system(system_prompt),
                inferenceConfig={
                    "maxTokens": max_new_tokens,
                    "temperature": temperature,
                    "topP": settings.LLM_TOP_P,
                },
            )

        resp = await asyncio.to_thread(_invoke)
        elapsed_ms = (time.perf_counter() - start) * 1000
        output = resp.get("output", {}).get("message", {})
        usage = resp.get("usage", {})

        return LLMResponse(
            text=self._extract_text(output),
            model=self.model,
            prompt_tokens=usage.get("inputTokens"),
            completion_tokens=usage.get("outputTokens"),
            generation_time_ms=round(elapsed_ms, 2),
        )

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = settings.LLM_MAX_NEW_TOKENS,
        temperature: float = settings.LLM_TEMPERATURE,
    ) -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _invoke_stream():
            try:
                resp = self.client.converse_stream(
                    modelId=self.model,
                    messages=self._build_messages(prompt),
                    system=self._build_system(system_prompt),
                    inferenceConfig={
                        "maxTokens": max_new_tokens,
                        "temperature": temperature,
                        "topP": settings.LLM_TOP_P,
                    },
                )
                for event in resp.get("stream", []):
                    delta = event.get("contentBlockDelta", {}).get("delta", {})
                    text = delta.get("text")
                    if text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(_invoke_stream))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            await task


# ── HuggingFace backend ───────────────────────────────────────────────────────

class HuggingFaceBackend(BaseLLMBackend):
    def __init__(self):
        self._pipe = None
        self.model_name = settings.HF_MODEL_NAME
        self.device = settings.HF_DEVICE

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch

            logger.info(f"Loading HuggingFace model '{self.model_name}'")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=settings.HF_TOKEN
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=settings.HF_TOKEN,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device,
            )
            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            logger.info("HuggingFace model loaded.")

    def _format_prompt(self, prompt: str, system_prompt: Optional[str]) -> str:
        if system_prompt:
            return f"<system>{system_prompt}</system>\n<user>{prompt}</user>\n<assistant>"
        return f"<user>{prompt}</user>\n<assistant>"

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = settings.LLM_MAX_NEW_TOKENS,
        temperature: float = settings.LLM_TEMPERATURE,
    ) -> LLMResponse:
        import asyncio

        self._load()
        formatted = self._format_prompt(prompt, system_prompt)
        start = time.perf_counter()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._pipe(
                formatted,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=settings.LLM_TOP_P,
                do_sample=temperature > 0,
                return_full_text=False,
            ),
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        text = result[0]["generated_text"] if result else ""
        return LLMResponse(
            text=text,
            model=self.model_name,
            generation_time_ms=round(elapsed_ms, 2),
        )

    async def stream(self, prompt, system_prompt=None, max_new_tokens=None, temperature=None):
        # HF streaming via text-streamer — simplified async wrapper
        response = await self.generate(prompt, system_prompt, max_new_tokens, temperature)
        yield response.text


# ── OpenAI backend ────────────────────────────────────────────────────────────

class OpenAIBackend(BaseLLMBackend):
    def __init__(self):
        import openai
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = settings.LLM_MAX_NEW_TOKENS,
        temperature: float = settings.LLM_TEMPERATURE,
    ) -> LLMResponse:
        start = time.perf_counter()
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system_prompt),
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=self.model,
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else None,
            completion_tokens=resp.usage.completion_tokens if resp.usage else None,
            generation_time_ms=round(elapsed_ms, 2),
        )

    async def stream(self, prompt, system_prompt=None, max_new_tokens=None, temperature=None):
        async for chunk in await self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system_prompt),
            max_tokens=max_new_tokens or settings.LLM_MAX_NEW_TOKENS,
            temperature=temperature or settings.LLM_TEMPERATURE,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── Service facade ────────────────────────────────────────────────────────────

class LLMService:
    """
    Unified LLM interface. Backend is determined by LLM_BACKEND env var.
    All methods are async.
    """

    def __init__(self):
        self._backend: Optional[BaseLLMBackend] = None

    def _resolve_backend_name(self) -> str:
        environment = settings.ENVIRONMENT.lower()

        # Environments: production => Bedrock by default, development => local model (Ollama) by default
        explicit = (settings.LLM_BACKEND or "").lower()

        if environment in {"prod", "production"}:
            # Production must use Bedrock unless explicitly set to a different remote backend.
            if explicit and explicit != "bedrock":
                return explicit
            return "bedrock"

        # Development (and other non-production) — prefer explicit backend, else use Ollama
        if explicit:
            return explicit
        return "ollama"

    def _get_backend(self) -> BaseLLMBackend:
        if self._backend is None:
            backend = self._resolve_backend_name()
            if backend == "ollama":
                self._backend = OllamaBackend()
            elif backend == "bedrock":
                self._backend = BedrockBackend()
            elif backend == "huggingface":
                self._backend = HuggingFaceBackend()
            elif backend == "openai":
                self._backend = OpenAIBackend()
            else:
                raise ValueError(f"Unknown LLM backend: {backend}")
            logger.info(
                "LLM backend initialized: %s (environment=%s)",
                backend,
                settings.ENVIRONMENT,
            )
        return self._backend

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        backend = self._get_backend()
        return await backend.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens or settings.LLM_MAX_NEW_TOKENS,
            temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
        )

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        backend = self._get_backend()
        async for chunk in backend.stream(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens or settings.LLM_MAX_NEW_TOKENS,
            temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
        ):
            yield chunk


# Singleton
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
