import asyncio
import json
import logging
from functools import partial

import boto3

from app.config import settings
from app.prompts import RENOVATION_SYSTEM_PROMPT, RENOVATION_USER_PROMPT

logger = logging.getLogger(__name__)


def _client():
    return boto3.client("bedrock-runtime", region_name=settings.bedrock_region)


def _invoke_sync(place: str, house_part: str, query: str) -> dict:
    user_prompt = RENOVATION_USER_PROMPT.format(
        place=place,
        house_part=house_part,
        user_query=query or "No specific requirements",
        max_suggestions=settings.max_suggestions,
    )

    # Gemma has no system role — prepend system prompt into user message
    combined = f"{RENOVATION_SYSTEM_PROMPT}\n\n{user_prompt}"

    logger.info(
        f"[Bedrock/Gemma] {house_part} in {place} | model: {settings.bedrock_model_id}"
    )

    response = _client().converse(
        modelId=settings.bedrock_model_id,
        messages=[{"role": "user", "content": [{"text": combined}]}],
        inferenceConfig={"maxTokens": 4096, "temperature": 0.8},
    )

    text = response["output"]["message"]["content"][0]["text"].strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return json.loads(text.strip())


async def get_suggestions(place: str, house_part: str, query: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(_invoke_sync, place, house_part, query)
    )