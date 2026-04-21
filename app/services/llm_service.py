import logging
from openai import AsyncOpenAI

from app.config import settings
from app.prompts import IMAGE_STYLE_SUFFIX

logger = logging.getLogger(__name__)


def _get_provider():
    if settings.llm_provider == "bedrock":
        from app.services import bedrock_provider
        return bedrock_provider
    from app.services import openai_provider
    return openai_provider


async def get_renovation_suggestions(place: str, house_part: str, query: str) -> dict:
    provider = _get_provider()
    logger.info(f"Environment: {settings.environment} | Provider: {settings.llm_provider}")
    return await provider.get_suggestions(place, house_part, query)


async def generate_renovation_image(image_prompt: str) -> str | None:
    image_key = settings.image_key()
    if not image_key:
        logger.warning("No OpenAI key for image generation — skipping")
        return None

    full_prompt = image_prompt + IMAGE_STYLE_SUFFIX
    logger.info("Generating renovation image with DALL-E")

    client = AsyncOpenAI(api_key=image_key)
    response = await client.images.generate(
        model=settings.openai_image_model,
        prompt=full_prompt,
        size=settings.image_size,
        quality=settings.image_quality,
        n=1,
    )
    return response.data[0].url