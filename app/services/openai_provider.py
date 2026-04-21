import json
import logging
from openai import AsyncOpenAI

from app.config import settings
from app.prompts import RENOVATION_SYSTEM_PROMPT, RENOVATION_USER_PROMPT

logger = logging.getLogger(__name__)


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def get_suggestions(place: str, house_part: str, query: str) -> dict:
    user_prompt = RENOVATION_USER_PROMPT.format(
        place=place,
        house_part=house_part,
        user_query=query or "No specific requirements",
        max_suggestions=settings.max_suggestions,
    )

    logger.info(f"[OpenAI] Fetching suggestions — {house_part} in {place}")

    response = await _client().chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": RENOVATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.8,
    )

    return json.loads(response.choices[0].message.content)
