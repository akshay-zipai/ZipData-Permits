"""
Basic tests for the CA Permit RAG system.
Run with: pytest tests/ -v
"""
import pytest
from unittest.mock import patch, MagicMock


# ── Text processing ────────────────────────────────────────────────────────────

def test_clean_text():
    from app.utils.text_processing import clean_text
    assert clean_text("  hello\r\n  world  ") == "hello\n  world"
    assert clean_text("") == ""


def test_chunk_text_short():
    from app.utils.text_processing import chunk_text
    text = "short text"
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long():
    from app.utils.text_processing import chunk_text
    words = ["word"] * 600
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1


def test_word_count():
    from app.utils.text_processing import word_count
    assert word_count("hello world foo") == 3
    assert word_count("") == 0


# ── Permit portals ─────────────────────────────────────────────────────────────

def test_get_portal_url_exact():
    from app.utils.permit_portals import get_portal_url
    url = get_portal_url("Alameda County")
    assert url is not None
    assert "acgov" in url or "alameda" in url.lower() or url.startswith("http")


def test_get_portal_url_without_county_suffix():
    from app.utils.permit_portals import get_portal_url
    url = get_portal_url("Alameda")
    assert url is not None


def test_get_portal_url_missing():
    from app.utils.permit_portals import get_portal_url
    url = get_portal_url("Nonexistent County XYZ")
    assert url is None


# ── Config ─────────────────────────────────────────────────────────────────────

def test_settings_load():
    from app.core.config import get_settings
    s = get_settings()
    assert s.APP_NAME
    assert s.EMBEDDING_MODEL_NAME == "all-MiniLM-L12-v2"
    assert s.LLM_BACKEND in ("ollama", "huggingface", "openai")


# ── FastAPI health endpoint ────────────────────────────────────────────────────

def test_health_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_root_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200


def test_counties_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.get("/api/v1/crawl/counties")
    assert resp.status_code == 200
    data = resp.json()
    assert "counties" in data
    assert data["total"] == 58


def test_rag_stats_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.get("/api/v1/rag/stats")
    assert resp.status_code == 200
