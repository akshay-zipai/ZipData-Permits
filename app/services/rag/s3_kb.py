"""
S3-backed Knowledge Base loader + BM25 retriever.

Loads JSONL files from an S3 bucket/prefix (or via a manifest), builds a BM25 index
from the `content` field of each record, and exposes `retrieve(query, top_k, county_filter)`.
This provides a simple, embedding-free retrieval path suitable for Bedrock-fed datasets.
"""
from __future__ import annotations

import json
import io
from typing import Optional, List

import boto3
from botocore.config import Config as BotoConfig

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.responses import RetrievedChunk

logger = get_logger(__name__)
settings = get_settings()


class S3KnowledgeBase:
    def __init__(self):
        # Auto-enable S3 KB if a bucket is configured (convenience for production),
        # but allow explicit disable via S3_KB_ENABLED=False.
        self._enabled = bool(settings.S3_KB_ENABLED or settings.S3_KB_BUCKET)
        self._bucket = settings.S3_KB_BUCKET
        self._prefix = settings.S3_KB_PREFIX
        self._manifest_key = settings.S3_KB_MANIFEST_KEY
        self._region = settings.S3_REGION_NAME

        self._client = None
        self._loaded = False

        # In-memory BM25 corpus + metadata
        self._corpus: List[str] = []
        self._metadata: List[dict] = []
        self._bm25 = None

    def _ensure_client(self):
        if self._client is None:
            cfg = BotoConfig(region_name=self._region) if self._region else None
            self._client = boto3.client("s3", config=cfg) if cfg else boto3.client("s3")

    def refresh(self) -> int:
        """Reload dataset from S3. Returns number of chunks loaded."""
        if not self._enabled:
            logger.info("S3 KB disabled in settings; refresh skipped.")
            return 0
        if not self._bucket:
            logger.error("S3 KB bucket not configured.")
            return 0

        self._ensure_client()

        keys: list[tuple[str, str]] = []  # list of (bucket, key)
        # If a manifest key is provided, use it to enumerate files
        if self._manifest_key:
            # If bucket is configured, try to load manifest from S3; otherwise
            # treat manifest key as a local file path for testing convenience.
            if self._bucket:
                try:
                    obj = self._client.get_object(Bucket=self._bucket, Key=self._manifest_key)
                    manifest = json.load(io.TextIOWrapper(obj["Body"], encoding="utf-8"))
                except Exception as e:
                    logger.error(f"Failed to load manifest {self._manifest_key} from s3://{self._bucket}: {e}")
                    manifest = {}
            else:
                # local manifest file
                try:
                    from pathlib import Path

                    p = Path(self._manifest_key)
                    manifest = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
                except Exception as e:
                    logger.error(f"Failed to load local manifest {self._manifest_key}: {e}")
                    manifest = {}

            for f in manifest.get("files", []):
                # support s3_uri entries produced by the generator script
                s3uri = f.get("s3_uri")
                if s3uri and s3uri.startswith("s3://"):
                    # parse s3://bucket/key
                    parts = s3uri[5:].split("/", 1)
                    if len(parts) == 2:
                        bkt, k = parts
                        keys.append((bkt, k))
                        continue

                path = f.get("path") or f.get("key")
                if path:
                    # If bucket configured, assume path is object key; otherwise treat as local file path
                    if self._bucket:
                        keys.append((self._bucket, path))
                    else:
                        keys.append((None, path))

        # Otherwise list objects under prefix
        if not keys:
            paginator = self._client.get_paginator("list_objects_v2")
            page_iter = paginator.paginate(Bucket=self._bucket, Prefix=self._prefix or "")
            for page in page_iter:
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if not key:
                        continue
                    # only pick jsonl files for now
                    if key.lower().endswith(".jsonl") or key.lower().endswith(".json"):
                        keys.append((self._bucket, key))

        # Clear old
        self._corpus = []
        self._metadata = []

        loaded = 0
        from pathlib import Path

        for bkt, key in keys:
            try:
                if bkt:
                    logger.info(f"Loading S3 object s3://{bkt}/{key}")
                    obj = self._client.get_object(Bucket=bkt, Key=key)
                    body = obj["Body"].read()
                    text = body.decode("utf-8")
                else:
                    # local file path
                    p = Path(key)
                    if not p.exists():
                        logger.warning(f"Local KB file not found: {p}")
                        continue
                    text = p.read_text(encoding="utf-8")
                # If JSONL, iterate lines
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        # try to parse as JSON array
                        try:
                            arr = json.loads(text)
                            if isinstance(arr, list):
                                for rec in arr:
                                    self._ingest_record(rec)
                                loaded += len(arr)
                                break
                        except Exception:
                            logger.warning(f"Skipping line (not json): {line[:120]}")
                            continue
                    else:
                        self._ingest_record(rec)
                        loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load s3://{self._bucket}/{key}: {e}")

        # Build BM25 index
        if self._corpus:
            try:
                from rank_bm25 import BM25Okapi

                tokenized = [doc.lower().split() for doc in self._corpus]
                self._bm25 = BM25Okapi(tokenized)
                logger.info(f"Built BM25 index with {len(self._corpus)} chunks from S3.")
                self._loaded = True
            except Exception as e:
                logger.error(f"Failed to build BM25 index: {e}")
                self._bm25 = None
                self._loaded = False

        return loaded

    def _ingest_record(self, rec: dict):
        # Expecting fields: content, county_name, source_url, chunk_id
        content = rec.get("content") or rec.get("text") or rec.get("text_content")
        if not content:
            return
        county = rec.get("county_name")
        meta = {
            "source_url": rec.get("source_url") or rec.get("url") or "",
            "county_name": county,
            "title": rec.get("title"),
            "chunk_id": rec.get("chunk_id") or rec.get("id") or str(len(self._corpus)),
        }
        self._corpus.append(content)
        self._metadata.append(meta)

    def retrieve(self, query: str, top_k: int = 5, county_filter: Optional[str] = None) -> List[RetrievedChunk]:
        """Return top_k RetrievedChunk objects using BM25 over the S3-loaded corpus."""
        if not self._enabled:
            logger.info("S3 KB disabled; returning empty results.")
            return []
        if not self._loaded:
            self.refresh()
        if not self._bm25:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        indexed_scores = list(enumerate(scores))

        if county_filter:
            indexed_scores = [
                (i, s) for i, s in indexed_scores if self._metadata[i].get("county_name") == county_filter
            ]

        top_items = sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:top_k]
        if not top_items:
            return []

        max_score = top_items[0][1] if top_items else 1.0
        results: List[RetrievedChunk] = []
        for idx, score in top_items:
            meta = self._metadata[idx]
            normalized = score / max_score if max_score else 0.0
            results.append(
                RetrievedChunk(
                    content=self._corpus[idx],
                    source_url=meta.get("source_url"),
                    county_name=meta.get("county_name"),
                    score=round(float(normalized), 6),
                    retrieval_method="s3_bm25",
                    chunk_id=meta.get("chunk_id", str(idx)),
                    metadata=meta,
                )
            )

        return results


# Singleton
_s3_kb: Optional[S3KnowledgeBase] = None


def get_s3_kb() -> S3KnowledgeBase:
    global _s3_kb
    if _s3_kb is None:
        _s3_kb = S3KnowledgeBase()
    return _s3_kb
