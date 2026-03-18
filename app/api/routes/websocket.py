"""
WebSocket endpoint for interactive permit Q&A.
Clients can maintain a session and ask multiple questions with streaming support.
"""
import json
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.models.requests import WSPermitQuestion
from app.models.responses import WSMessage
from app.services.rag.pipeline import get_rag_pipeline
from app.services.llm.generator import get_llm_service
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(tags=["WebSocket"])


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active[session_id] = websocket
        logger.info(f"WS connected: session={session_id} | total={len(self.active)}")

    def disconnect(self, session_id: str):
        self.active.pop(session_id, None)
        logger.info(f"WS disconnected: session={session_id} | total={len(self.active)}")

    async def send_json(self, session_id: str, message: WSMessage):
        ws = self.active.get(session_id)
        if ws:
            await ws.send_text(message.model_dump_json())

    async def send_error(self, session_id: str, error: str, detail: Optional[str] = None):
        await self.send_json(
            session_id,
            WSMessage(type="error", content={"error": error, "detail": detail}, session_id=session_id),
        )


manager = ConnectionManager()


@router.websocket("/ws/permit-qa")
async def permit_qa_ws(websocket: WebSocket):
    """
    WebSocket endpoint for real-time permit Q&A with streaming support.

    **Protocol:**
    - Client sends JSON: `{"question": "...", "zip_code": "94102", "county_name": null}`
    - Server responds with status updates and the final answer (or streamed chunks)
    - Connection stays open for multiple questions

    **Message types (server → client):**
    - `status`: Processing update (crawling, retrieving, generating…)
    - `stream_chunk`: Partial answer token (if streaming enabled)
    - `answer`: Final complete answer with sources
    - `error`: Error details
    """
    session_id = str(uuid.uuid4())
    await manager.connect(websocket, session_id)

    try:
        while True:
            raw = await websocket.receive_text()

            # Parse incoming message
            try:
                data = json.loads(raw)
                question_req = WSPermitQuestion(**data)
                question_req.session_id = question_req.session_id or session_id
            except (json.JSONDecodeError, ValidationError) as e:
                await manager.send_error(session_id, "Invalid message format", str(e))
                continue

            # Acknowledge receipt
            await manager.send_json(
                session_id,
                WSMessage(
                    type="status",
                    content={"message": "Question received. Processing…", "session_id": session_id},
                    session_id=session_id,
                ),
            )

            try:
                # Resolve county context
                county_name = question_req.county_name
                zip_code = question_req.zip_code

                if not county_name and zip_code:
                    await manager.send_json(
                        session_id,
                        WSMessage(type="status", content={"message": f"Resolving county for ZIP {zip_code}…"}, session_id=session_id),
                    )
                    from app.utils.zip_lookup import zip_to_county
                    county_name = zip_to_county(zip_code)

                # Check/trigger crawl if needed
                from app.services.rag.retriever import get_retriever_service
                retriever = get_retriever_service()
                stats = retriever.get_collection_stats()

                if (
                    settings.ENABLE_LOCAL_RAG
                    and settings.ENABLE_AUTO_CRAWL
                    and stats.get("total_chunks", 0) == 0
                    and county_name
                ):
                    await manager.send_json(
                        session_id,
                        WSMessage(type="status", content={"message": f"Crawling permit portal for {county_name}…"}, session_id=session_id),
                    )

                await manager.send_json(
                    session_id,
                    WSMessage(type="status", content={"message": "Retrieving relevant permit information…"}, session_id=session_id),
                )

                # Use streaming LLM via RAG pipeline
                from app.models.requests import RAGQueryRequest
                rag_request = RAGQueryRequest(
                    question=question_req.question,
                    zip_code=zip_code,
                    county_name=county_name,
                    top_k=settings.RAG_TOP_K,
                    include_sources=True,
                )

                pipeline = get_rag_pipeline()

                # Retrieve chunks first
                from app.services.rag.retriever import get_retriever_service
                from app.utils.zip_lookup import zip_to_county as _zip2county

                effective_county = county_name
                if not effective_county and zip_code:
                    effective_county = _zip2county(zip_code)

                retriever = get_retriever_service()
                chunks = []
                if settings.ENABLE_LOCAL_RAG:
                    chunks = retriever.retrieve(
                        query=question_req.question,
                        top_k=settings.RAG_TOP_K,
                        county_filter=effective_county,
                    )

                if not chunks and effective_county and settings.ENABLE_LOCAL_RAG and settings.ENABLE_AUTO_CRAWL:
                    from app.utils.permit_portals import get_portal_url
                    from app.services.crawling.crawler import get_crawler_service

                    portal_url = get_portal_url(effective_county)
                    if portal_url:
                        await manager.send_json(
                            session_id,
                            WSMessage(type="status", content={"message": f"Auto-crawling {portal_url}…"}, session_id=session_id),
                        )
                        crawler = get_crawler_service()
                        crawl_result = await crawler.crawl_url(portal_url, effective_county)
                        for page in crawl_result.pages:
                            if page.text_content:
                                retriever.index_content(page.text_content, effective_county, page.url)
                        chunks = retriever.retrieve(question_req.question, settings.RAG_TOP_K, effective_county)

                # Stream the answer
                await manager.send_json(
                    session_id,
                    WSMessage(type="status", content={"message": "Generating answer…"}, session_id=session_id),
                )

                llm = get_llm_service()
                prompt = pipeline._build_prompt(question_req.question, chunks, effective_county)
                system_prompt = pipeline._get_system_prompt() or None

                full_answer = ""
                async for token in llm.stream(prompt=prompt, system_prompt=system_prompt):
                    full_answer += token
                    await manager.send_json(
                        session_id,
                        WSMessage(type="stream_chunk", content={"token": token}, session_id=session_id),
                    )

                # Send final answer with sources
                sources_data = [c.model_dump() for c in chunks]
                await manager.send_json(
                    session_id,
                    WSMessage(
                        type="answer",
                        content={
                            "answer": full_answer,
                            "county_name": effective_county,
                            "zip_code": zip_code,
                            "sources": sources_data,
                        },
                        session_id=session_id,
                    ),
                )

            except Exception as e:
                logger.error(f"WS handler error for session {session_id}: {e}", exc_info=True)
                await manager.send_error(session_id, "Processing failed", str(e))

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"Unexpected WS error: {e}", exc_info=True)
        manager.disconnect(session_id)


@router.get("/ws/stats", summary="Active WebSocket connections")
async def ws_stats():
    return {"active_connections": len(manager.active)}
