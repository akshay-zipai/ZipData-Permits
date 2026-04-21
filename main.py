"""
FastAPI backend — CA Permit & Renovation Agent.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent.config import get_settings
from agent.agent import AgentState, PermitRenoAgent, clear_session, get_session
from agent.permit_kb import get_permit_kb

settings = get_settings()
agent = PermitRenoAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the KB so the first request isn't slow
    kb = get_permit_kb()
    kb._load()
    print(
        f"[startup] KB loaded — {len(kb._records):,} records, "
        f"{len(kb.get_all_counties())} counties | "
        f"backend={settings.ENVIRONMENT}"
    )
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Unified CA Permit RAG + Renovation Suggestion Agent. "
        f"LLM backend: {'Bedrock' if settings.is_production else 'OpenAI'}"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class StartRequest(BaseModel):
    session_id: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    message: str
    state: str
    data: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    is_done: bool = False


class ResetRequest(BaseModel):
    session_id: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health():
    kb = get_permit_kb()
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "llm_backend": "bedrock" if settings.is_production else "openai",
        "llm_model": settings.BEDROCK_MODEL_ID if settings.is_production else settings.OPENAI_MODEL,
        "kb_records": len(kb._records),
        "kb_counties": len(kb.get_all_counties()),
    }


@app.post("/session/start", response_model=ChatResponse, tags=["Session"])
async def start_session(req: StartRequest):
    session_id = req.session_id or str(uuid.uuid4())
    clear_session(session_id)
    response = agent.start()
    ctx = get_session(session_id)
    ctx.state = response.state
    return ChatResponse(
        session_id=session_id,
        message=response.message,
        state=response.state.value,
        suggestions=response.suggestions,
        data=response.data,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    ctx = get_session(req.session_id)
    try:
        response = await agent.process(req.message, ctx)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        session_id=req.session_id,
        message=response.message,
        state=response.state.value,
        data=response.data,
        suggestions=response.suggestions,
        is_done=response.state == AgentState.DONE,
    )


@app.post("/session/reset", tags=["Session"])
async def reset_session(req: ResetRequest):
    clear_session(req.session_id)
    return {"status": "reset", "session_id": req.session_id}


@app.get("/counties", tags=["Data"])
async def get_counties():
    return {"counties": get_permit_kb().get_all_counties()}


@app.get("/counties/{county}/zips", tags=["Data"])
async def get_zips(county: str):
    return {"county": county, "zip_codes": get_permit_kb().get_zips_for_county(county)}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )
