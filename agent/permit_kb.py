"""
Offline File-backed Permit Knowledge Base.
Loads bedrock_kb_by_zip.jsonl (zip-keyed) or bedrock_kb.jsonl and performs
lightweight lexical retrieval. No external vector DB required.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class PermitChunk:
    content: str
    county_name: str
    zip_code: Optional[str]
    source_url: Optional[str]
    title: Optional[str]
    score: float = 0.0


class PermitKB:
    """Loads JSONL dataset and provides BM25-style lexical retrieval."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._records: List[dict] = []
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        candidates = [
            self.data_dir / "bedrock_kb_by_zip.jsonl",
            self.data_dir / "bedrock_kb.jsonl",
        ]
        for path in candidates:
            if path.exists():
                print(f"[PermitKB] Loading from {path}")
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            self._records.append(json.loads(line))
                        except Exception:
                            pass
                print(f"[PermitKB] Loaded {len(self._records)} records")
                break
        self._loaded = True

    def _score(self, query: str, content: str) -> float:
        q_terms = set(re.findall(r"\w+", query.lower()))
        c_text = content.lower()
        if not q_terms:
            return 0.0
        hits = sum(1 for t in q_terms if t in c_text)
        # Boost for exact phrase
        bonus = 2 if query.lower() in c_text else 0
        return float(hits + bonus)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        county_filter: Optional[str] = None,
        zip_filter: Optional[str] = None,
    ) -> List[PermitChunk]:
        self._load()
        candidates = self._records

        # Filter by zip first (most specific)
        if zip_filter:
            zip_matches = [r for r in candidates if r.get("zip_code") == zip_filter]
            if zip_matches:
                candidates = zip_matches
            elif county_filter:
                candidates = [r for r in candidates if r.get("county_name", "").lower() == county_filter.lower()]
        elif county_filter:
            candidates = [r for r in candidates if r.get("county_name", "").lower() == county_filter.lower()]

        scored = []
        for rec in candidates:
            content = rec.get("content", "")
            s = self._score(query, content)
            scored.append((s, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, rec in scored[:top_k]:
            results.append(
                PermitChunk(
                    content=rec.get("content", ""),
                    county_name=rec.get("county_name", ""),
                    zip_code=rec.get("zip_code"),
                    source_url=rec.get("source_url"),
                    title=rec.get("title"),
                    score=s,
                )
            )
        return results

    def county_for_zip(self, zip_code: str) -> Optional[str]:
        """Look up county name for a given ZIP code from the dataset."""
        self._load()
        for rec in self._records:
            if rec.get("zip_code") == zip_code and rec.get("county_name"):
                return rec["county_name"]
        return None

    def get_all_counties(self) -> List[str]:
        self._load()
        return sorted(set(r.get("county_name", "") for r in self._records if r.get("county_name")))

    def get_zips_for_county(self, county: str) -> List[str]:
        self._load()
        return sorted(set(
            r.get("zip_code", "")
            for r in self._records
            if r.get("county_name", "").lower() == county.lower() and r.get("zip_code")
        ))


_kb: Optional[PermitKB] = None


def get_permit_kb(data_dir: Optional[Path] = None) -> PermitKB:
    global _kb
    if _kb is None:
        from agent.config import get_settings
        settings = get_settings()
        _kb = PermitKB(data_dir or settings.DATA_DIR)
    return _kb
