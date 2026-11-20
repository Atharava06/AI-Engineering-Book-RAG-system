import math
import pickle
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from backend.config import settings
from backend.schemas import DocumentMetadata, RetrievedChunk

logger = logging.getLogger(__name__)


class BM25Index:
    """High-performance BM25Okapi keyword search engine with persistence."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_lengths: List[int] = []
        self.doc_metadata: List[DocumentMetadata] = []
        self.idf: Dict[str, float] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.index_file = Path(settings.STORAGE_DIR) / "bm25_index.pkl"

    def _tokenize(self, text: str) -> List[str]:
        """Simple, robust alphanumeric tokenization."""
        return [w.lower() for w in re.findall(r"\b\w{2,}\b", text)]

    def fit(self, chunks: List[DocumentMetadata]):
        """Builds BM25 inverted index from document chunks."""
        # Replace the complete index, including when the new corpus is empty.
        self.doc_metadata = list(chunks)
        self.corpus_size = len(chunks)
        self.avgdl = 0.0
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.idf = {}
        if self.corpus_size == 0:
            self.save()
            return

        self.doc_lengths = []
        self.doc_term_freqs = []
        df: Dict[str, int] = {}

        total_length = 0
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_term_freqs.append(freqs)

            for t in freqs.keys():
                df[t] = df.get(t, 0) + 1

        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

        # Compute Robertson-Sparck Jones IDF
        self.idf = {}
        for term, freq in df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[term] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

        logger.info(f"Built BM25 index over {self.corpus_size} chunks with {len(self.idf)} unique terms")
        self.save()

    def save(self):
        """Persists the index to disk."""
        try:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_file, "wb") as f:
                pickle.dump(
                    {
                        "k1": self.k1,
                        "b": self.b,
                        "corpus_size": self.corpus_size,
                        "avgdl": self.avgdl,
                        "doc_lengths": self.doc_lengths,
                        "doc_metadata": self.doc_metadata,
                        "idf": self.idf,
                        "doc_term_freqs": self.doc_term_freqs,
                    },
                    f,
                )
        except Exception as e:
            logger.warning(f"Failed to persist BM25 index: {e}")

    def load(self) -> bool:
        """Loads index from disk if available."""
        if not self.index_file.exists():
            return False
        try:
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
                self.k1 = data["k1"]
                self.b = data["b"]
                self.corpus_size = data["corpus_size"]
                self.avgdl = data["avgdl"]
                self.doc_lengths = data["doc_lengths"]
                self.doc_metadata = data["doc_metadata"]
                self.idf = data["idf"]
                self.doc_term_freqs = data["doc_term_freqs"]
            logger.info(f"Loaded BM25 index with {self.corpus_size} chunks from {self.index_file}")
            return True
        except Exception as e:
            logger.warning(f"Could not load BM25 index from file: {e}")
            return False

    def search(
        self,
        query: str,
        top_k: int = 30,
        chapter_filter: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Calculates BM25 relevance scores for all documents matching filters."""
        if self.corpus_size == 0:
            return []

        # Repeated query words should not multiply the same BM25 contribution.
        query_tokens = list(dict.fromkeys(self._tokenize(query)))
        if not query_tokens:
            return []

        scored_candidates: List[Tuple[float, int]] = []

        for idx, (chunk, freqs, doc_len) in enumerate(
            zip(self.doc_metadata, self.doc_term_freqs, self.doc_lengths)
        ):
            # Metadata filter checks
            if chapter_filter and chunk.chapter != chapter_filter:
                continue
            if page_start is not None and chunk.page_number < page_start:
                continue
            if page_end is not None and chunk.page_number > page_end:
                continue

            score = 0.0
            for term in query_tokens:
                if term in freqs:
                    term_freq = freqs[term]
                    term_idf = self.idf.get(term, 0.0)
                    # BM25 term weighting formula
                    numerator = term_freq * (self.k1 + 1.0)
                    denominator = term_freq + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
                    score += term_idf * (numerator / denominator)

            if score > 0.0:
                scored_candidates.append((score, idx))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        results: List[RetrievedChunk] = []
        for score, idx in scored_candidates[:top_k]:
            chunk = self.doc_metadata[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    book_name=chunk.book_name,
                    chapter=chunk.chapter,
                    section=chunk.section,
                    page_number=chunk.page_number,
                    source_file=chunk.source_file,
                    bm25_score=round(score, 4),
                    final_score=round(score, 4),
                    metadata={
                        "headings_hierarchy": chunk.headings_hierarchy,
                        "token_count": chunk.token_count,
                    },
                )
            )

        return results
