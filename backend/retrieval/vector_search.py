import logging
from typing import List, Optional

from backend.embeddings.embedding_service import BaseEmbeddingService
from backend.vectorstore.qdrant_store import QdrantVectorStore
from backend.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class DenseVectorSearch:
    """Performs dense vector retrieval against Qdrant."""

    def __init__(self, vector_store: QdrantVectorStore, embedding_service: BaseEmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def search(
        self,
        query: str,
        top_k: int = 30,
        chapter_filter: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Embeds query and queries Qdrant."""
        query_vector = self.embedding_service.embed_query(query)
        results = self.vector_store.dense_search(
            query_vector=query_vector,
            top_k=top_k,
            chapter_filter=chapter_filter,
            page_start=page_start,
            page_end=page_end,
        )
        return results
