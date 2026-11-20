import logging
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from backend.config import settings
from backend.schemas import DocumentMetadata, RetrievedChunk

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        Range,
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


class QdrantVectorStore:
    """Manages Qdrant vector database interactions, indexing, and dense retrieval."""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int = settings.EMBEDDING_DIMENSION,
    ):
        self.storage_path = storage_path or settings.QDRANT_STORAGE_PATH
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.dimension = dimension
        self.client = None
        self._fallback_points: Dict[str, Dict[str, Any]] = {}
        self._init_client()

    def _init_client(self):
        """Initializes the Qdrant client or memory fallback."""
        if HAS_QDRANT:
            try:
                if self.storage_path == ":memory:":
                    self.client = QdrantClient(location=":memory:")
                else:
                    path = Path(self.storage_path)
                    path.mkdir(parents=True, exist_ok=True)
                    self.client = QdrantClient(path=str(path))
                self._ensure_collection()
                logger.info(f"Connected to Qdrant at {self.storage_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize persistent Qdrant at {self.storage_path}: {e}. Using in-memory client.")
                try:
                    self.client = QdrantClient(location=":memory:")
                    self._ensure_collection()
                    return
                except Exception as ex2:
                    logger.error(f"In-memory Qdrant also failed: {ex2}. Falling back to internal dictionary.")
        self.client = None

    def _ensure_collection(self):
        """Ensures the collection exists with the appropriate vector parameters."""
        if not self.client:
            return
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name} (dim: {self.dimension})")

    def count(self) -> int:
        """Returns the number of indexed vectors."""
        if self.client:
            try:
                info = self.client.get_collection(collection_name=self.collection_name)
                return info.points_count or 0
            except Exception:
                pass
        return len(self._fallback_points)

    def clear(self) -> None:
        """Remove all indexed points while keeping the collection available."""
        if self.client:
            try:
                self.client.delete_collection(collection_name=self.collection_name)
                self._ensure_collection()
                return
            except Exception as e:
                raise RuntimeError(f"Could not clear Qdrant collection {self.collection_name}: {e}") from e
        self._fallback_points.clear()

    def close(self):
        """Safely close client connection."""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

    def __del__(self):
        self.close()


    def upsert_chunks(
        self,
        chunks: List[DocumentMetadata],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """Upserts chunks and their embeddings into Qdrant."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch")

        total_upserted = 0
        if self.client:
            points = []
            for chunk, emb in zip(chunks, embeddings):
                # Ensure point id is deterministic or uuid
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
                payload = {
                    "book_name": chunk.book_name,
                    "chapter": chunk.chapter,
                    "section": chunk.section,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source_file": chunk.source_file,
                    "token_count": chunk.token_count,
                    "headings_hierarchy": chunk.headings_hierarchy,
                }
                points.append(PointStruct(id=point_id, vector=emb, payload=payload))

            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch)
                total_upserted += len(batch)
        else:
            for chunk, emb in zip(chunks, embeddings):
                self._fallback_points[chunk.chunk_id] = {
                    "vector": emb,
                    "payload": {
                        "book_name": chunk.book_name,
                        "chapter": chunk.chapter,
                        "section": chunk.section,
                        "page_number": chunk.page_number,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "source_file": chunk.source_file,
                        "token_count": chunk.token_count,
                    },
                }
            total_upserted = len(chunks)

        return total_upserted

    def dense_search(
        self,
        query_vector: List[float],
        top_k: int = 30,
        chapter_filter: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Performs dense vector similarity search with optional metadata filtering."""
        results: List[RetrievedChunk] = []

        if self.client:
            must_filters = []
            if chapter_filter:
                must_filters.append(
                    FieldCondition(key="chapter", match=MatchValue(value=chapter_filter))
                )
            if page_start is not None or page_end is not None:
                range_args = {}
                if page_start is not None:
                    range_args["gte"] = page_start
                if page_end is not None:
                    range_args["lte"] = page_end
                must_filters.append(FieldCondition(key="page_number", range=Range(**range_args)))

            qdrant_filter = Filter(must=must_filters) if must_filters else None

            # Support search / query_points
            hits = []
            try:
                # qdrant-client >= 1.10 query_points or search
                if hasattr(self.client, "query_points"):
                    query_res = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        limit=top_k,
                        query_filter=qdrant_filter,
                    )
                    hits = query_res.points
                else:
                    hits = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query_vector,
                        limit=top_k,
                        query_filter=qdrant_filter,
                    )
            except Exception as e:
                logger.error(f"Qdrant search error: {e}")
                hits = []

            for hit in hits:
                payload = hit.payload or {}
                results.append(
                    RetrievedChunk(
                        chunk_id=payload.get("chunk_id", str(hit.id)),
                        text=payload.get("text", ""),
                        book_name=payload.get("book_name", "AI Engineering"),
                        chapter=payload.get("chapter", "Unknown Chapter"),
                        section=payload.get("section", "General"),
                        page_number=payload.get("page_number", 1),
                        source_file=payload.get("source_file", "AI_Engineering.pdf"),
                        dense_score=float(hit.score),
                        final_score=float(hit.score),
                        metadata=payload,
                    )
                )
        else:
            # Fallback cosine search over fallback points
            import numpy as np
            q_vec = np.array(query_vector, dtype=np.float32)
            candidates = []
            for cid, data in self._fallback_points.items():
                p = data["payload"]
                if chapter_filter and p.get("chapter") != chapter_filter:
                    continue
                pno = p.get("page_number", 1)
                if page_start is not None and pno < page_start:
                    continue
                if page_end is not None and pno > page_end:
                    continue
                v = np.array(data["vector"], dtype=np.float32)
                sim = float(np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-9))
                candidates.append((sim, p))

            candidates.sort(key=lambda x: x[0], reverse=True)
            for sim, p in candidates[:top_k]:
                results.append(
                    RetrievedChunk(
                        chunk_id=p["chunk_id"],
                        text=p["text"],
                        book_name=p["book_name"],
                        chapter=p["chapter"],
                        section=p["section"],
                        page_number=p["page_number"],
                        source_file=p["source_file"],
                        dense_score=sim,
                        final_score=sim,
                        metadata=p,
                    )
                )

        return results
