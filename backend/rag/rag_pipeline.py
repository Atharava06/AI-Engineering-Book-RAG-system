import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.config import settings
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    RetrievedChunk,
    SummarizeRequest,
    SummarizeResponse,
    CompareRequest,
    CompareResponse,
    ChaptersResponse,
    ChapterInfo,
    IngestionStatusResponse,
)
from backend.retrieval.hybrid_search import HybridRetrievalEngine
from backend.generation.llm_service import BaseLLMService, get_llm_service
from backend.generation.prompt_templates import (
    SYSTEM_PROMPT,
    SUMMARY_PROMPT,
    COMPARISON_PROMPT,
    format_context_for_prompt,
)

logger = logging.getLogger(__name__)


class AIEngineeringRAGPipeline:
    """End-to-End RAG Pipeline for the AI Engineering book."""

    def __init__(
        self,
        hybrid_retrieval: HybridRetrievalEngine,
        llm_service: Optional[BaseLLMService] = None,
    ):
        self.retrieval = hybrid_retrieval
        self.llm_service = llm_service or get_llm_service()
        self.ingestion_status = IngestionStatusResponse(
            status="idle",
            book_name="AI Engineering",
            message="No document ingestion running",
        )

    def _parse_answer_sections(self, text: str) -> Dict[str, Any]:
        """Extracts structured sections from the LLM markdown response."""
        direct_match = re.search(
            r"###\s*1\.\s*Direct Answer\s*\n(.*?)(?=\n###\s*2|\Z)", text, re.DOTALL
        )
        detailed_match = re.search(
            r"###\s*2\.\s*Detailed Explanation\s*\n(.*?)(?=\n###\s*3|\Z)", text, re.DOTALL
        )
        concepts_match = re.search(
            r"###\s*3\.\s*Important Concepts\s*\n(.*?)(?=\n###\s*4|\Z)", text, re.DOTALL
        )
        example_match = re.search(
            r"###\s*4\.\s*Example\s*\n(.*?)(?=\n###\s*5|\Z)", text, re.DOTALL
        )

        direct_ans = direct_match.group(1).strip() if direct_match else ""
        detailed_ans = detailed_match.group(1).strip() if detailed_match else ""

        concepts_raw = concepts_match.group(1).strip() if concepts_match else ""
        concepts_list = [
            re.sub(r"^[-*•\d.]+\s*", "", line).strip()
            for line in concepts_raw.splitlines()
            if line.strip()
        ]

        examples_ans = example_match.group(1).strip() if example_match else ""

        return {
            "direct_answer": direct_ans,
            "detailed_explanation": detailed_ans,
            "important_concepts": concepts_list,
            "examples": examples_ans,
        }

    def _build_citations(self, chunks: List[RetrievedChunk]) -> List[Citation]:
        """Extracts structured citations with book, chapter, section, and original PDF page."""
        citations: List[Citation] = []
        seen = set()
        for c in chunks:
            section = c.section.strip() if c.section else ""
            if not section or re.fullmatch(r"(?:page\s*)?\d+", section, re.IGNORECASE):
                heading = re.search(r"(?m)^\s*(\d+(?:\.\d+)+\s+[^\n]{3,100})\s*$", c.text)
                section = heading.group(1).strip() if heading else "Section heading not identified"

            key = (c.chapter, section, c.page_number)
            if key in seen:
                continue
            seen.add(key)

            snippet_text = re.sub(r"^\[[^\]]+\]\s*", "", c.text.strip())
            snippet = snippet_text[:200].replace("\n", " ")
            if len(snippet_text) > 200:
                snippet += "..."
            citations.append(
                Citation(
                    book_name=c.book_name,
                    chapter=c.chapter,
                    section=section,
                    page_number=c.page_number,
                    source_file=c.source_file,
                    snippet=snippet,
                    relevance_score=c.final_score,
                )
            )
        return citations

    def answer_question(self, req: ChatRequest) -> ChatResponse:
        """Answers a user question grounded in retrieved book content."""
        start_time = time.perf_counter()

        # Retrieve relevant chunks via Hybrid Search
        search_res = self.retrieval.search(
            query=req.query,
            top_k=settings.FINAL_TOP_K,
            chapter_filter=req.chapter_filter,
            page_start=req.page_start,
            page_end=req.page_end,
            use_reranker=True,
        )

        context_str = format_context_for_prompt(search_res.results)

        # Build prompt including conversation history
        history_str = ""
        if req.history:
            history_blocks = []
            for h in req.history[-4:]:  # Keep last 4 turns for context
                history_blocks.append(f"{h.role.capitalize()}: {h.content}")
            history_str = f"Prior Conversation History:\n" + "\n".join(history_blocks) + "\n\n"

        prompt = (
            f"{history_str}"
            f"Retrieved Content from AI Engineering Book:\n{context_str}\n\n"
            f"User Question: {req.query}\n"
            f"Target Persona Level: {req.persona_level}\n\n"
            f"Answer following all 10 rules and output the required 5-part format."
        )

        answer_text = self.llm_service.generate(
            prompt=prompt,
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1 if req.temperature is None else req.temperature,
        )

        parsed_parts = self._parse_answer_sections(answer_text)
        citations = self._build_citations(search_res.results)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return ChatResponse(
            answer=answer_text,
            direct_answer=parsed_parts["direct_answer"] or None,
            detailed_explanation=parsed_parts["detailed_explanation"] or None,
            important_concepts=parsed_parts["important_concepts"],
            examples=parsed_parts["examples"] or None,
            citations=citations,
            retrieved_chunks=search_res.results,
            latency_ms=latency_ms,
            model_used=getattr(self.llm_service, "model", "fallback"),
        )

    def summarize_chapter(self, req: SummarizeRequest) -> SummarizeResponse:
        """Generates structured chapter summary."""
        start_time = time.perf_counter()
        search_res = self.retrieval.search(
            query=f"Overview, principles, and key concepts of {req.chapter}",
            top_k=8,
            chapter_filter=req.chapter,
            use_reranker=False,
        )

        context_str = format_context_for_prompt(search_res.results)
        prompt = (
            f"Content from {req.chapter}:\n{context_str}\n\n"
            f"Detail Level: {req.detail_level}\n"
            f"Generate an executive summary, list of core architectural principles, key technical concepts, and key takeaways."
        )

        response_text = self.llm_service.generate(
            prompt=prompt,
            system_instruction=SUMMARY_PROMPT.format(chapter=req.chapter),
            temperature=0.2,
        )

        citations = self._build_citations(search_res.results)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return SummarizeResponse(
            chapter=req.chapter,
            summary=response_text,
            key_concepts=[c.section for c in search_res.results[:5]],
            key_takeaways=[
                f"Ground decisions in {req.chapter} engineering patterns.",
                "Balance latency, cost, and reliability.",
            ],
            citations=citations,
            latency_ms=latency_ms,
        )

    def compare_concepts(self, req: CompareRequest) -> CompareResponse:
        """Generates evidence-backed comparison between two concepts."""
        start_time = time.perf_counter()

        # Retrieve for concept A and concept B
        res_a = self.retrieval.search(query=req.concept_a, top_k=4)
        res_b = self.retrieval.search(query=req.concept_b, top_k=4)
        combined_chunks = res_a.results + res_b.results

        context_str = format_context_for_prompt(combined_chunks)
        prompt = (
            f"Retrieved Content for Comparison:\n{context_str}\n\n"
            f"Compare: '{req.concept_a}' vs '{req.concept_b}'\n"
            f"Provide direct comparison, markdown table of differences, and tradeoffs."
        )

        comparison_text = self.llm_service.generate(
            prompt=prompt,
            system_instruction=COMPARISON_PROMPT.format(
                concept_a=req.concept_a, concept_b=req.concept_b
            ),
            temperature=0.1,
        )

        citations = self._build_citations(combined_chunks)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return CompareResponse(
            concept_a=req.concept_a,
            concept_b=req.concept_b,
            direct_comparison=comparison_text,
            key_differences=[
                {"dimension": "Latency", "difference": "Evaluated per architectural requirement"},
                {"dimension": "Accuracy", "difference": "Grounded in retrieved book evidence"},
            ],
            synthesis_and_tradeoffs="Choose based on retrieval accuracy, cost per query, and freshness requirements.",
            citations=citations,
            latency_ms=latency_ms,
        )

    def get_indexed_chapters(self) -> ChaptersResponse:
        """Returns indexed chapters, page ranges, and sections."""
        chunks = self.retrieval.bm25_index.doc_metadata
        if not chunks:
            return ChaptersResponse(book_name="AI Engineering", total_pages=0, chapters=[])

        chapter_map: Dict[str, Dict[str, Any]] = {}
        max_page = 1

        for c in chunks:
            ch = c.chapter
            if ch not in chapter_map:
                chapter_map[ch] = {
                    "start_page": c.page_number,
                    "end_page": c.page_number,
                    "count": 0,
                    "sections": set(),
                }
            chapter_map[ch]["start_page"] = min(chapter_map[ch]["start_page"], c.page_number)
            chapter_map[ch]["end_page"] = max(chapter_map[ch]["end_page"], c.page_number)
            chapter_map[ch]["count"] += 1
            if c.section:
                chapter_map[ch]["sections"].add(c.section)
            max_page = max(max_page, c.page_number)

        ch_list = []
        for ch_title, info in chapter_map.items():
            ch_list.append(
                ChapterInfo(
                    chapter_title=ch_title,
                    start_page=info["start_page"],
                    end_page=info["end_page"],
                    chunk_count=info["count"],
                    sections=sorted(list(info["sections"]))[:10],
                )
            )

        ch_list.sort(key=lambda x: x.start_page)

        return ChaptersResponse(
            book_name="AI Engineering",
            total_pages=max_page,
            chapters=ch_list,
        )
