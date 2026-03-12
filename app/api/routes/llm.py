from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from app.models.requests import LLMGenerateRequest
from app.models.responses import LLMResponse
from app.services.llm.generator import get_llm_service, LLMService
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/llm", tags=["LLM Inference"])


def _get_llm() -> LLMService:
    return get_llm_service()


@router.post(
    "/generate",
    response_model=LLMResponse,
    summary="Raw LLM text generation",
)
async def generate(
    request: LLMGenerateRequest,
    llm: LLMService = Depends(_get_llm),
):
    """
    Direct LLM inference endpoint.
    Use this to test prompts or perform generation outside of the RAG pipeline.
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Use /llm/stream for streaming responses.",
        )
    try:
        return await llm.generate(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream", summary="Streaming LLM generation (Server-Sent Events)")
async def generate_stream(
    request: LLMGenerateRequest,
    llm: LLMService = Depends(_get_llm),
):
    """
    Streaming LLM generation via Server-Sent Events.
    Each chunk is prefixed with 'data: ' for SSE compatibility.
    """
    async def event_generator():
        try:
            async for chunk in llm.stream(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
