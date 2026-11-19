import pytest
from backend.schemas import DocumentMetadata, ChatRequest, SummarizeRequest, CompareRequest
from backend.embeddings.embedding_service import LocalFastEmbeddingService
from backend.vectorstore.qdrant_store import QdrantVectorStore
from backend.retrieval.bm25_search import BM25Index
from backend.retrieval.vector_search import DenseVectorSearch
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.hybrid_search import HybridRetrievalEngine
from backend.generation.llm_service import FallbackOfflineLLMService
from backend.rag.rag_pipeline import AIEngineeringRAGPipeline
from backend.evaluation.evaluator import RAGEvaluator


@pytest.fixture
def rag_pipeline():
    chunks = [
        DocumentMetadata(
            book_name="AI Engineering",
            chapter="Chapter 1: Foundations",
            section="1.1 Overview",
            page_number=10,
            chunk_id="c1",
            text="RAG grounds foundation models with authoritative external knowledge and verifiable source citations.",
            source_file="AI_Engineering.pdf",
        ),
        DocumentMetadata(
            book_name="AI Engineering",
            chapter="Chapter 1: Foundations",
            section="1.2 Fine-Tuning Comparison",
            page_number=18,
            chunk_id="c2",
            text="Fine-tuning modifies model parameters for specialized style or format adaptation rather than dynamic factual recall.",
            source_file="AI_Engineering.pdf",
        ),
    ]

    emb = LocalFastEmbeddingService(dimension=64)
    vstore = QdrantVectorStore(storage_path=":memory:", dimension=64)
    vstore.upsert_chunks(chunks, emb.embed_documents([c.text for c in chunks]))

    bm25 = BM25Index()
    bm25.fit(chunks)

    dense = DenseVectorSearch(vstore, emb)
    hybrid = HybridRetrievalEngine(dense, bm25, CrossEncoderReranker())
    llm = FallbackOfflineLLMService()

    return AIEngineeringRAGPipeline(hybrid, llm)


def test_answer_generation_and_citations(rag_pipeline):
    req = ChatRequest(query="How does RAG provide verifiable citations?")
    res = rag_pipeline.answer_question(req)

    assert res.answer is not None
    assert len(res.citations) > 0
    # Page citation verification
    assert any(c.page_number in [10, 18] for c in res.citations)
    assert res.latency_ms > 0


def test_summarization(rag_pipeline):
    req = SummarizeRequest(chapter="Chapter 1: Foundations", detail_level="detailed")
    res = rag_pipeline.summarize_chapter(req)

    assert "Chapter 1: Foundations" in res.chapter
    assert len(res.summary) > 50
    assert len(res.citations) > 0


def test_concept_comparison(rag_pipeline):
    req = CompareRequest(concept_a="RAG", concept_b="Fine-Tuning")
    res = rag_pipeline.compare_concepts(req)

    assert res.concept_a == "RAG"
    assert res.concept_b == "Fine-Tuning"
    assert len(res.direct_comparison) > 50


def test_evaluator_metrics(rag_pipeline):
    evaluator = RAGEvaluator(rag_pipeline)
    res = evaluator.run_benchmark(sample_size=2)

    assert res.overall_score >= 0.0
    assert len(res.metrics) == 7
    assert res.evaluated_questions == 2
