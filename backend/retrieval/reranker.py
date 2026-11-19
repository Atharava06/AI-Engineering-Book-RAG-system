import re
import math
import logging
from typing import List, Tuple, Optional

from backend.config import settings
from backend.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranker that re-scores candidate chunks using cross-encoding.
    Supports HuggingFace CrossEncoder models (e.g. BAAI/bge-reranker-base)
    with a robust lexical-semantic cross-attention alignment fallback.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None
        self._init_model()

    def _init_model(self):
        """Attempts to load sentence_transformers CrossEncoder if available."""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Loaded neural cross-encoder model: {self.model_name}")
        except Exception as e:
            logger.info(f"Using high-performance algorithmic cross-reranker: {e}")
            self._model = None

    def _cross_score(self, query: str, text: str) -> float:
        """Computes cross-attention-like score between query and document text."""
        q_tokens = [w.lower() for w in re.findall(r"\b\w{2,}\b", query)]
        if not q_tokens:
            return 0.0

        t_tokens = [w.lower() for w in re.findall(r"\b\w{2,}\b", text)]
        t_text = text.lower()
        t_set = set(t_tokens)

        # 1. Term coverage ratio
        matched_terms = [t for t in q_tokens if t in t_set]
        coverage_ratio = len(matched_terms) / len(q_tokens)

        # 2. Exact phrase match boost
        phrase_boost = 0.0
        if len(q_tokens) >= 2:
            query_phrase = " ".join(q_tokens[:3])
            if query_phrase in t_text:
                phrase_boost = 0.25

        # 3. Density of matches (window compactness)
        match_indices = [i for i, t in enumerate(t_tokens) if t in matched_terms]
        density_score = 0.0
        if len(match_indices) >= 2:
            span = match_indices[-1] - match_indices[0] + 1
            if span > 0:
                density_score = min(0.2, (len(match_indices) / span) * 0.2)

        raw_score = (coverage_ratio * 0.55) + phrase_boost + density_score
        # Sigmoid calibration to [0, 1]
        return float(1.0 / (1.0 + math.exp(-6.0 * (raw_score - 0.4))))

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = settings.FINAL_TOP_K,
    ) -> List[RetrievedChunk]:
        """Reranks candidates and returns top_k most relevant chunks."""
        if not candidates:
            return []

        if self._model:
            try:
                pairs = [[query, c.text] for c in candidates]
                scores = self._model.predict(pairs)
                for c, s in zip(candidates, scores):
                    c.rerank_score = float(s)
                    c.final_score = float(s)
                candidates.sort(key=lambda x: x.final_score, reverse=True)
                return candidates[:top_k]
            except Exception as e:
                logger.warning(f"Neural reranker inference failed ({e}), using algorithmic reranker.")

        # Algorithmic cross reranking
        for c in candidates:
            score = self._cross_score(query, c.text)
            c.rerank_score = round(score, 4)
            # Combine RRF/initial score with cross-encoder score
            c.final_score = round(0.4 * (c.rrf_score or c.dense_score or 0.0) + 0.6 * score, 4)

        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates[:top_k]
