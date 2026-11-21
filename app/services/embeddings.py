"""Embedding service using sentence-transformers and OpenAI."""
import logging
from typing import List, Optional
import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles text embedding generation for vector search."""

    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._openai_client = None
        self._initialized = False

    def _init_model(self):
        """Lazy load sentence-transformers model."""
        if self._initialized:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformers model: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("sentence-transformers not available, will use OpenAI embeddings")
            self._model = None
        self._initialized = True

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        self._init_model()

        # Try OpenAI first if API key available
        if self.settings.OPENAI_API_KEY:
            return await self._embed_openai(text)

        # Fall back to sentence-transformers
        if self._model:
            embedding = self._model.encode(text)
            return embedding.tolist()

        # Fallback: simple hash-based embedding (not for production)
        raise RuntimeError("No embedding provider available")

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        return await self.embed_text(query)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        self._init_model()

        if self.settings.OPENAI_API_KEY:
            embeddings = []
            for text in texts:
                emb = await self._embed_openai(text)
                embeddings.append(emb)
            return embeddings

        if self._model:
            embeddings = self._model.encode(texts)
            return embeddings.tolist()

        raise RuntimeError("No embedding provider available")

    async def _embed_openai(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            from openai import AsyncOpenAI
            if not self._openai_client:
                self._openai_client = AsyncOpenAI(
                    api_key=self.settings.OPENAI_API_KEY
                )
            response = await self._openai_client.embeddings.create(
                model=self.settings.OPENAI_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {str(e)}")
            raise
