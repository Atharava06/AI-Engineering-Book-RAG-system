import time
import logging
from typing import List, Optional, Dict

from backend.config import settings
from backend.schemas import RetrievedChunk, SearchResponse
from backend.retrieval.vector_search import DenseVectorSearch
from backend.retrieval.bm25_search import BM25Index
from backend.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HybridRetrievalEngine:
    """Orchestrates Hybrid Retrieval: Dense Search + BM25 + Reciprocal Rank Fusion (RRF) + Reranking."""

    def __init__(
        self,
        vector_search: DenseVectorSearch,
        bm25_index: BM25Index,
        reranker: Optional[CrossEncoderReranker] = None,
        rrf_k: int = settings.RRF_K,
    ):
        self.vector_search = vector_search
        self.bm25_index = bm25_index
        self.reranker = reranker or CrossEncoderReranker()
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = settings.FINAL_TOP_K,
        candidate_k: int = settings.TOP_K_DENSE,
        chapter_filter: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        use_reranker: bool = True,
    ) -> SearchResponse:
        """Executes complete hybrid search pipeline."""
        start_time = time.perf_counter()

        # Step 1: Query Preprocessing
        clean_query = query.strip()
        if not clean_query:
            return SearchResponse(query="", total_candidates=0, results=[], latency_ms=0.0)

        # Step 2: Dense Vector Search
        dense_results = self.vector_search.search(
            query=clean_query,
            top_k=candidate_k,
            chapter_filter=chapter_filter,
            page_start=page_start,
            page_end=page_end,
        )

        # Step 3: BM25 Keyword Search
        bm25_results = self.bm25_index.search(
            query=clean_query,
            top_k=candidate_k,
            chapter_filter=chapter_filter,
            page_start=page_start,
            page_end=page_end,
        )

        # Step 4: Reciprocal Rank Fusion (RRF)
        # Score(d) = sum(1 / (k + rank))
        rrf_scores: Dict[str, float] = {}
        chunks_map: Dict[str, RetrievedChunk] = {}

        # Fuse Dense
        for rank, chunk in enumerate(dense_results, start=1):
            cid = chunk.chunk_id
            rrf_contrib = 1.0 / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contrib
            if cid not in chunks_map:
                chunks_map[cid] = chunk
            chunks_map[cid].dense_score = chunk.dense_score

        # Fuse BM25
        for rank, chunk in enumerate(bm25_results, start=1):
            cid = chunk.chunk_id
            rrf_contrib = 1.0 / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contrib
            if cid not in chunks_map:
                chunks_map[cid] = chunk
            chunks_map[cid].bm25_score = chunk.bm25_score

        # Assign RRF scores to candidate chunks
        fused_candidates: List[RetrievedChunk] = []
        for cid, score in rrf_scores.items():
            ch = chunks_map[cid]
            ch.rrf_score = round(score, 5)
            fused_candidates.append(ch)

        # Sort by RRF score descending
        fused_candidates.sort(key=lambda x: x.rrf_score or 0.0, reverse=True)

        # Step 5: Cross-Encoder Reranking
        if use_reranker and fused_candidates:
            # Send top 20 fused candidates to reranker
            to_rerank = fused_candidates[: min(25, len(fused_candidates))]
            final_chunks = self.reranker.rerank(query=clean_query, candidates=to_rerank, top_k=top_k)
        else:
            final_chunks = fused_candidates[:top_k]
            for c in final_chunks:
                c.final_score = c.rrf_score or c.dense_score or 0.0

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return SearchResponse(
            query=clean_query,
            total_candidates=len(fused_candidates),
            results=final_chunks,
            latency_ms=latency_ms,
        )
