"""
Embedding Service — SBERT all-MiniLM-L12-v2 (default).
Swap the model by changing EMBEDDING_MODEL_NAME in .env.
"""
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.responses import EmbeddingResponse

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingService:
    """
    Wraps sentence-transformers for text embedding.
    Model is loaded lazily on first use to avoid startup overhead.
    """

    def __init__(self):
        self._model = None
        self._model_name = settings.EMBEDDING_MODEL_NAME
        self._device = settings.EMBEDDING_DEVICE

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(
                f"Loading embedding model '{self._model_name}' on {self._device}"
            )
            self._model = SentenceTransformer(
                self._model_name, device=self._device
            )
            logger.info("Embedding model loaded.")

    def embed_texts(self, texts: list[str]) -> EmbeddingResponse:
        """Embed a list of strings and return an EmbeddingResponse."""
        if not texts:
            return EmbeddingResponse(embeddings=[], model=self._model_name, dimensions=0, count=0)

        self._load_model()

        vectors = self._model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embeddings = vectors.tolist()
        dimensions = vectors.shape[1] if vectors.ndim == 2 else 0

        return EmbeddingResponse(
            embeddings=embeddings,
            model=self._model_name,
            dimensions=dimensions,
            count=len(embeddings),
        )

    def embed_single(self, text: str) -> list[float]:
        """Convenience method: embed a single string."""
        result = self.embed_texts([text])
        return result.embeddings[0] if result.embeddings else []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> Optional[int]:
        if self._model is None:
            return None
        return self._model.get_sentence_embedding_dimension()


# Singleton
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
