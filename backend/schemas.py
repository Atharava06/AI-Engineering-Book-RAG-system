from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    book_name: str = "AI Engineering"
    chapter: str = "Unknown Chapter"
    section: str = "General"
    page_number: int
    chunk_id: str
    text: str
    source_file: str = "AI_Engineering.pdf"
    token_count: Optional[int] = None
    headings_hierarchy: List[str] = Field(default_factory=list)


class IngestionStatusResponse(BaseModel):
    status: str  # "idle", "processing", "completed", "failed"
    book_name: str = "AI Engineering"
    source_file: Optional[str] = None
    total_pages: int = 0
    processed_pages: int = 0
    total_chunks: int = 0
    total_chapters: int = 0
    progress_percentage: float = 0.0
    message: str = ""
    error: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    book_name: str = "AI Engineering"
    chapter: str
    section: str
    page_number: int
    source_file: str
    dense_score: Optional[float] = 0.0
    bm25_score: Optional[float] = 0.0
    rrf_score: Optional[float] = 0.0
    rerank_score: Optional[float] = 0.0
    final_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 6
    chapter_filter: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    use_reranker: bool = True


class SearchResponse(BaseModel):
    query: str
    total_candidates: int
    results: List[RetrievedChunk]
    latency_ms: float


class Citation(BaseModel):
    book_name: str = "AI Engineering"
    chapter: str
    section: str
    page_number: int
    source_file: str
    snippet: str
    relevance_score: Optional[float] = None


class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = Field(default_factory=list)
    chapter_filter: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    persona_level: Optional[str] = "expert"  # "beginner", "intermediate", "expert"
    temperature: Optional[float] = 0.1


class ChatResponse(BaseModel):
    answer: str
    direct_answer: Optional[str] = None
    detailed_explanation: Optional[str] = None
    important_concepts: List[str] = Field(default_factory=list)
    examples: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float
    model_used: str


class SummarizeRequest(BaseModel):
    chapter: str
    detail_level: str = "detailed"  # "concise", "detailed", "bullet_points"


class SummarizeResponse(BaseModel):
    chapter: str
    summary: str
    key_concepts: List[str]
    key_takeaways: List[str]
    citations: List[Citation]
    latency_ms: float


class CompareRequest(BaseModel):
    concept_a: str
    concept_b: str
    chapter_context: Optional[str] = None


class CompareResponse(BaseModel):
    concept_a: str
    concept_b: str
    direct_comparison: str
    key_differences: List[Dict[str, str]]
    synthesis_and_tradeoffs: str
    citations: List[Citation]
    latency_ms: float


class ChapterInfo(BaseModel):
    chapter_title: str
    start_page: int
    end_page: int
    chunk_count: int
    sections: List[str] = Field(default_factory=list)


class ChaptersResponse(BaseModel):
    book_name: str
    total_pages: int
    chapters: List[ChapterInfo]


class EvaluateRequest(BaseModel):
    sample_size: int = 10
    include_ragas: bool = False


class MetricScore(BaseModel):
    name: str
    score: float
    benchmark_threshold: float
    passed: bool
    description: str


class EvaluationResult(BaseModel):
    timestamp: str
    overall_score: float
    metrics: List[MetricScore]
    evaluated_questions: int
    latency_avg_ms: float
    sample_evaluations: List[Dict[str, Any]]
