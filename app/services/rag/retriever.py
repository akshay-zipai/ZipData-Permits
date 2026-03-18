"""
RAG Retriever — Hybrid BM25 (keyword) + ChromaDB (semantic) retrieval.
Scores are combined with configurable weights from settings.
"""
import importlib.util
import uuid
import time
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.responses import RetrievedChunk
from app.services.embedding.embedder import get_embedding_service
from app.utils.text_processing import chunk_text, clean_text

logger = get_logger(__name__)
settings = get_settings()


class RetrieverService:
    def __init__(self):
        self._enabled = settings.ENABLE_LOCAL_RAG
        if self._enabled and importlib.util.find_spec("chromadb") is None:
            logger.warning(
                "Local RAG requested but chromadb is not installed; disabling local RAG."
            )
            self._enabled = False
        self._chroma_client = None
        self._collection = None
        self._bm25_corpus: list[str] = []
        self._bm25_metadata: list[dict] = []
        self._bm25_index = None

    # ── ChromaDB setup ────────────────────────────────────────────────────────

    def _get_collection(self):
        if not self._enabled:
            raise RuntimeError("Local RAG is disabled in this environment.")
        if self._collection is None:
            import chromadb

            self._chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}' ready."
            )
        return self._collection

    # ── BM25 setup ────────────────────────────────────────────────────────────

    def _rebuild_bm25(self):
        if not self._bm25_corpus:
            self._bm25_index = None
            return
        from rank_bm25 import BM25Okapi

        tokenized = [doc.lower().split() for doc in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized)
        logger.debug(f"BM25 index rebuilt with {len(tokenized)} documents.")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_content(
        self,
        content: str,
        county_name: str,
        source_url: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> int:
        """Chunk and index content into ChromaDB and BM25. Returns chunk count."""
        if not self._enabled:
            raise RuntimeError("Local RAG indexing is disabled in this environment.")
        collection = self._get_collection()
        embedder = get_embedding_service()

        chunks = chunk_text(clean_text(content))
        if not chunks:
            logger.warning(f"No chunks extracted for {county_name}")
            return 0

        embeddings_resp = embedder.embed_texts(chunks)
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]

        meta_base = {
            "county_name": county_name,
            "source_url": source_url or "",
        }
        if extra_metadata:
            meta_base.update(extra_metadata)

        metadatas = [dict(meta_base, chunk_index=i) for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings_resp.embeddings,
            metadatas=metadatas,
            ids=chunk_ids,
        )

        # BM25 corpus update
        for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
            self._bm25_corpus.append(chunk)
            self._bm25_metadata.append({**meta, "chunk_id": chunk_ids[i]})

        self._rebuild_bm25()
        logger.info(f"Indexed {len(chunks)} chunks for {county_name}")
        return len(chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        county_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        if not self._enabled:
            logger.info("Local RAG retrieval is disabled; returning no local chunks.")
            return []
        top_k = top_k or settings.RAG_TOP_K
        vector_results = self._vector_search(query, top_k * 2, county_filter)
        bm25_results = self._bm25_search(query, top_k * 2, county_filter)

        merged = self._hybrid_merge(vector_results, bm25_results, top_k)
        return merged

    def _vector_search(
        self, query: str, top_k: int, county_filter: Optional[str]
    ) -> list[RetrievedChunk]:
        collection = self._get_collection()
        embedder = get_embedding_service()

        query_vec = embedder.embed_single(query)
        where = {"county_name": county_filter} if county_filter else None

        try:
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, max(1, collection.count())),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Vector search error: {e}")
            return []

        chunks: list[RetrievedChunk] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            # ChromaDB cosine distance → similarity
            score = max(0.0, 1.0 - dist)
            chunks.append(
                RetrievedChunk(
                    content=doc,
                    source_url=meta.get("source_url"),
                    county_name=meta.get("county_name"),
                    score=round(score, 4),
                    retrieval_method="vector",
                    chunk_id=meta.get("chunk_id", str(uuid.uuid4())),
                    metadata=meta,
                )
            )
        return chunks

    def _bm25_search(
        self, query: str, top_k: int, county_filter: Optional[str]
    ) -> list[RetrievedChunk]:
        if self._bm25_index is None or not self._bm25_corpus:
            return []

        tokens = query.lower().split()
        scores = self._bm25_index.get_scores(tokens)

        indexed_scores = list(enumerate(scores))
        # Filter by county if specified
        if county_filter:
            indexed_scores = [
                (i, s) for i, s in indexed_scores
                if self._bm25_metadata[i].get("county_name") == county_filter
            ]

        top_items = sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:top_k]
        max_score = top_items[0][1] if top_items else 1.0
        if max_score == 0:
            return []

        chunks: list[RetrievedChunk] = []
        for idx, score in top_items:
            meta = self._bm25_metadata[idx]
            normalized_score = score / max_score
            chunks.append(
                RetrievedChunk(
                    content=self._bm25_corpus[idx],
                    source_url=meta.get("source_url"),
                    county_name=meta.get("county_name"),
                    score=round(normalized_score, 4),
                    retrieval_method="bm25",
                    chunk_id=meta.get("chunk_id", str(idx)),
                    metadata=meta,
                )
            )
        return chunks

    def _hybrid_merge(
        self,
        vector_chunks: list[RetrievedChunk],
        bm25_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion + weighted score combination."""
        score_map: dict[str, dict] = {}

        for rank, chunk in enumerate(vector_chunks):
            cid = chunk.chunk_id
            if cid not in score_map:
                score_map[cid] = {"chunk": chunk, "hybrid_score": 0.0}
            rrf = 1.0 / (60 + rank + 1)
            score_map[cid]["hybrid_score"] += settings.RAG_VECTOR_WEIGHT * rrf

        for rank, chunk in enumerate(bm25_chunks):
            cid = chunk.chunk_id
            if cid not in score_map:
                score_map[cid] = {"chunk": chunk, "hybrid_score": 0.0}
            rrf = 1.0 / (60 + rank + 1)
            score_map[cid]["hybrid_score"] += settings.RAG_BM25_WEIGHT * rrf

        sorted_items = sorted(
            score_map.values(), key=lambda x: x["hybrid_score"], reverse=True
        )[:top_k]

        results: list[RetrievedChunk] = []
        for item in sorted_items:
            chunk = item["chunk"]
            chunk.score = round(item["hybrid_score"], 6)
            chunk.retrieval_method = "hybrid"
            results.append(chunk)

        return results

    def get_collection_stats(self) -> dict:
        if not self._enabled:
            return {
                "enabled": False,
                "total_chunks": 0,
                "bm25_corpus_size": 0,
                "collection_name": settings.CHROMA_COLLECTION_NAME,
            }
        try:
            col = self._get_collection()
            return {
                "enabled": True,
                "total_chunks": col.count(),
                "bm25_corpus_size": len(self._bm25_corpus),
                "collection_name": settings.CHROMA_COLLECTION_NAME,
            }
        except Exception as e:
            return {"error": str(e)}


# Singleton
_retriever_service: Optional[RetrieverService] = None


def get_retriever_service() -> RetrieverService:
    global _retriever_service
    if _retriever_service is None:
        _retriever_service = RetrieverService()
    return _retriever_service
