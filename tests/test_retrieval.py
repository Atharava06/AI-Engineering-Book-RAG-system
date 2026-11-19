import pytest
from backend.schemas import DocumentMetadata
from backend.retrieval.bm25_search import BM25Index
from backend.embeddings.embedding_service import LocalFastEmbeddingService
from backend.vectorstore.qdrant_store import QdrantVectorStore
from backend.retrieval.vector_search import DenseVectorSearch
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.hybrid_search import HybridRetrievalEngine


@pytest.fixture
def sample_chunks():
    return [
        DocumentMetadata(
            book_name="AI Engineering",
            chapter="Chapter 3: Retrieval & Indexing",
            section="3.1 Dense Vectors",
            page_number=45,
            chunk_id="c1",
            text="Dense vector retrieval maps textual passages into continuous high-dimensional geometric spaces capturing semantics.",
            source_file="AI_Engineering.pdf",
        ),
        DocumentMetadata(
            book_name="AI Engineering",
            chapter="Chapter 3: Retrieval & Indexing",
            section="3.2 BM25 Keyword Search",
            page_number=52,
            chunk_id="c2",
            text="BM25 relies on exact term frequencies and inverse document frequencies for keyword search.",
            source_file="AI_Engineering.pdf",
        ),
        DocumentMetadata(
            book_name="AI Engineering",
            chapter="Chapter 4: Reranking & Fusion",
            section="4.1 RRF",
            page_number=110,
            chunk_id="c3",
            text="Reciprocal Rank Fusion merges dense and sparse rankings using 1 / (k + rank).",
            source_file="AI_Engineering.pdf",
        ),
    ]


def test_bm25_search(sample_chunks):
    bm25 = BM25Index()
    bm25.fit(sample_chunks)

    results = bm25.search("term frequencies BM25", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c2"
    assert results[0].page_number == 52


def test_hybrid_search_with_rrf(sample_chunks):
    emb_service = LocalFastEmbeddingService(dimension=64)
    vstore = QdrantVectorStore(storage_path=":memory:", dimension=64)
    embs = emb_service.embed_documents([c.text for c in sample_chunks])
    vstore.upsert_chunks(sample_chunks, embs)

    bm25 = BM25Index()
    bm25.fit(sample_chunks)

    dense = DenseVectorSearch(vstore, emb_service)
    reranker = CrossEncoderReranker()
    hybrid = HybridRetrievalEngine(dense, bm25, reranker)

    res = hybrid.search(query="Reciprocal Rank Fusion k constant", top_k=3)
    assert len(res.results) > 0
    # The RRF chunk c3 should rank at or near top
    assert any(c.chunk_id == "c3" for c in res.results)
    assert res.results[0].final_score > 0
