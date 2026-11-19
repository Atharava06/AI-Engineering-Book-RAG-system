import os
import shutil
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.schemas import (
    SearchRequest,
    SearchResponse,
    ChatRequest,
    ChatResponse,
    SummarizeRequest,
    SummarizeResponse,
    CompareRequest,
    CompareResponse,
    ChaptersResponse,
    EvaluateRequest,
    EvaluationResult,
    IngestionStatusResponse,
)
from backend.ingestion.document_parser import DocumentParser
from backend.chunking.chunker import StructureAwareChunker
from backend.embeddings.embedding_service import get_embedding_service
from backend.vectorstore.qdrant_store import QdrantVectorStore
from backend.retrieval.bm25_search import BM25Index
from backend.retrieval.vector_search import DenseVectorSearch
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.hybrid_search import HybridRetrievalEngine
from backend.generation.llm_service import get_llm_service
from backend.rag.rag_pipeline import AIEngineeringRAGPipeline
from backend.evaluation.evaluator import RAGEvaluator
from backend.utils.helpers import create_sample_ai_engineering_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AIEngineeringRAG")

app = FastAPI(
    title="AI Engineering Book RAG API",
    description="Production-grade Advanced RAG backend for the AI Engineering Book",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services
embedding_service = get_embedding_service()
vector_store = QdrantVectorStore()
bm25_index = BM25Index()
# Attempt loading persisted BM25 index
bm25_index.load()

dense_search = DenseVectorSearch(vector_store, embedding_service)
reranker = CrossEncoderReranker()
hybrid_engine = HybridRetrievalEngine(dense_search, bm25_index, reranker)
llm_service = get_llm_service()
rag_pipeline = AIEngineeringRAGPipeline(hybrid_engine, llm_service)
evaluator = RAGEvaluator(rag_pipeline)

# Ingestion state tracking
ingestion_state = {
    "status": "idle",
    "book_name": "AI Engineering",
    "source_file": None,
    "total_pages": 0,
    "processed_pages": 0,
    "total_chunks": 0,
    "total_chapters": 0,
    "progress_percentage": 0.0,
    "message": "System ready for document ingestion",
    "error": None,
}


def run_ingestion_job(pdf_path: Path, book_name: str = "AI Engineering"):
    """Background task performing end-to-end PDF ingestion, chunking, and indexing."""
    global ingestion_state
    try:
        ingestion_state["status"] = "processing"
        ingestion_state["source_file"] = pdf_path.name
        ingestion_state["book_name"] = book_name
        ingestion_state["error"] = None
        ingestion_state["message"] = f"Parsing PDF pages from {pdf_path.name}..."

        parser = DocumentParser()

        def on_parse_progress(curr, total, msg):
            ingestion_state["processed_pages"] = curr
            ingestion_state["total_pages"] = total
            ingestion_state["progress_percentage"] = round((curr / max(1, total)) * 40.0, 1)
            ingestion_state["message"] = msg

        pages = parser.parse_document(pdf_path, progress_callback=on_parse_progress)
        ingestion_state["total_pages"] = len(pages)
        ingestion_state["processed_pages"] = len(pages)
        ingestion_state["message"] = f"Chunking {len(pages)} pages with structure awareness..."
        ingestion_state["progress_percentage"] = 45.0

        chunker = StructureAwareChunker()
        chunks = chunker.chunk_all_pages(pages, book_name=book_name)
        ingestion_state["total_chunks"] = len(chunks)
        ingestion_state["message"] = f"Generated {len(chunks)} chunks. Generating embeddings..."
        ingestion_state["progress_percentage"] = 55.0

        # Batch embed and index into Qdrant
        texts = [c.text for c in chunks]
        embeddings = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embs = embedding_service.embed_documents(batch_texts)
            embeddings.extend(batch_embs)
            progress = 55.0 + ((i + len(batch_texts)) / len(texts)) * 30.0
            ingestion_state["progress_percentage"] = round(progress, 1)
            ingestion_state["message"] = f"Embedded {len(embeddings)}/{len(texts)} chunks..."

        ingestion_state["message"] = "Upserting vectors into Qdrant..."
        ingestion_state["progress_percentage"] = 88.0
        vector_store.clear()
        vector_store.upsert_chunks(chunks, embeddings)

        ingestion_state["message"] = "Building BM25 keyword index..."
        ingestion_state["progress_percentage"] = 95.0
        bm25_index.fit(chunks)

        # Count unique chapters
        unique_chapters = len(set(c.chapter for c in chunks))
        ingestion_state["total_chapters"] = unique_chapters

        ingestion_state["status"] = "completed"
        ingestion_state["progress_percentage"] = 100.0
        ingestion_state["message"] = (
            f"Successfully indexed {len(pages)} pages and {len(chunks)} chunks across {unique_chapters} chapters."
        )
        logger.info(ingestion_state["message"])
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        ingestion_state["status"] = "failed"
        ingestion_state["error"] = str(e)
        ingestion_state["message"] = f"Ingestion failed: {e}"


@app.get("/health")
def health_check():
    """Health check endpoint displaying system readiness and vector counts."""
    return {
        "status": "healthy",
        "book_name": "AI Engineering",
        "vector_count": vector_store.count(),
        "bm25_corpus_size": bm25_index.corpus_size,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "qdrant_path": settings.QDRANT_STORAGE_PATH,
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF file into the uploads directory."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_dir = Path(settings.UPLOADS_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / file.filename

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": f"Successfully uploaded {file.filename}",
        "filename": file.filename,
        "file_path": str(target_path),
        "file_size_bytes": target_path.stat().st_size,
    }


@app.post("/ingest")
def trigger_ingestion(
    filename: Optional[str] = Query(None, description="Filename from data/uploads"),
    use_sample: bool = Query(False, description="Generate and ingest sample AI Engineering book"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Triggers ingestion pipeline for an uploaded PDF or sample book."""
    if use_sample:
        sample_path = Path(settings.UPLOADS_DIR) / "AI_Engineering_Sample.pdf"
        create_sample_ai_engineering_pdf(sample_path, num_pages=35)
        pdf_path = sample_path
    elif filename:
        pdf_path = Path(settings.UPLOADS_DIR) / filename
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"File {filename} not found in uploads directory.")
    else:
        # Check if any PDF is available in uploads
        available_pdfs = list(Path(settings.UPLOADS_DIR).glob("*.pdf"))
        if not available_pdfs:
            raise HTTPException(
                status_code=400,
                detail="No PDF uploaded. Please upload the AI Engineering PDF first or pass use_sample=true.",
            )
        pdf_path = available_pdfs[0]

    # Run in background task
    background_tasks.add_task(run_ingestion_job, pdf_path, "AI Engineering")

    return {
        "message": f"Ingestion started for {pdf_path.name}",
        "target_file": pdf_path.name,
        "status": "processing",
    }


@app.get("/ingestion/status", response_model=IngestionStatusResponse)
def get_ingestion_status():
    """Returns the current state and progress of document ingestion."""
    return IngestionStatusResponse(**ingestion_state)


@app.post("/chat", response_model=ChatResponse)
def chat_with_book(request: ChatRequest):
    """Answers a user question grounded in the retrieved book content."""
    try:
        return rag_pipeline.answer_question(request)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
def search_book(request: SearchRequest):
    """Hybrid dense + BM25 keyword search across the book with metadata filters."""
    return hybrid_engine.search(
        query=request.query,
        top_k=request.top_k,
        chapter_filter=request.chapter_filter,
        page_start=request.page_start,
        page_end=request.page_end,
        use_reranker=request.use_reranker,
    )


@app.post("/summarize", response_model=SummarizeResponse)
def summarize_chapter(request: SummarizeRequest):
    """Generates an executive technical summary for a given chapter."""
    return rag_pipeline.summarize_chapter(request)


@app.post("/compare", response_model=CompareResponse)
def compare_concepts(request: CompareRequest):
    """Compares two AI engineering concepts using retrieved book evidence."""
    return rag_pipeline.compare_concepts(request)


@app.get("/chapters", response_model=ChaptersResponse)
def get_chapters():
    """Lists all indexed chapters, page ranges, and sections."""
    return rag_pipeline.get_indexed_chapters()


@app.post("/evaluate", response_model=EvaluationResult)
def evaluate_rag(request: EvaluateRequest):
    """Runs automated benchmark evaluation measuring precision, recall, faithfulness, and citations."""
    return evaluator.run_benchmark(sample_size=request.sample_size)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.BACKEND_HOST, port=settings.BACKEND_PORT, reload=True)
