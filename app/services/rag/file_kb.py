"""File-backed knowledge base loader for Bedrock JSONL outputs.

This module loads `output/bedrock_kb_by_zip.jsonl` or `output/bedrock_kb.jsonl`
and exposes a simple `retrieve(query, top_k, county_filter)` function that
returns `RetrievedChunk` objects prioritized by simple lexical scoring.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.responses import RetrievedChunk

logger = get_logger(__name__)
settings = get_settings()


class FileKB:
    def __init__(self, path: Path):
        self.path = path
        self._data: List[dict] = []
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if not self.path.exists():
            logger.info(f"File KB not found at {self.path}")
            self._data = []
            self._loaded = True
            return
        logger.info(f"Loading File KB from {self.path}")
        out = []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    out.append(rec)
        except Exception as e:
            logger.error(f"Failed reading File KB {self.path}: {e}")
        self._data = out
        self._loaded = True

    def retrieve(self, query: str, top_k: int = 5, county_filter: Optional[str] = None) -> List[RetrievedChunk]:
        self._load()
        if not self._data:
            return []

        # filter by county_name (exact match) if provided
        candidates = self._data
        if county_filter:
            candidates = [c for c in candidates if (c.get("county_name") or "") == county_filter]

        # Score by simple lexical overlap
        q_terms = [t.lower() for t in (query or "").split() if t]

        def score(rec: dict) -> float:
            content = (rec.get("content") or "").lower()
            if not q_terms:
                return float(len(content))
            return float(sum(1 for t in q_terms if t in content))

        scored = sorted(candidates, key=score, reverse=True)
        selected = scored[:top_k]

        results: List[RetrievedChunk] = []
        for rec in selected:
            results.append(
                RetrievedChunk(
                    content=rec.get("content") or "",
                    source_url=rec.get("source_url"),
                    county_name=rec.get("county_name"),
                    score=float(score(rec)),
                    retrieval_method="file_kb",
                    chunk_id=rec.get("chunk_id") or rec.get("id") or "",
                    metadata={"word_count": rec.get("word_count")},
                )
            )

        return results


_kb_instance = None


def get_file_kb() -> Optional[FileKB]:
    global _kb_instance
    if _kb_instance is not None:
        return _kb_instance

    candidates = [
        settings.DATA_DIR / "bedrock_kb_by_zip.jsonl",
        settings.DATA_DIR / "bedrock_kb.jsonl",
        settings.OUTPUT_DIR / "bedrock_kb_by_zip.jsonl",
        settings.OUTPUT_DIR / "bedrock_kb.jsonl",
    ]
    for p in candidates:
        if p.exists():
            _kb_instance = FileKB(p)
            return _kb_instance

    return None
