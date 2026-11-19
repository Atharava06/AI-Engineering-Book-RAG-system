import pymupdf as fitz
from pathlib import Path
from typing import List, Dict, Any


def create_sample_ai_engineering_pdf(output_path: Path, num_pages: int = 35) -> Path:
    """Generates a realistic multi-chapter 'AI Engineering' PDF document
    with headers, code blocks, tables, and sections for automated testing and immediate evaluation.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()

    chapters_content = [
        {
            "chapter": "Chapter 1: Foundations of AI Engineering",
            "section": "1.1 The Emerging Discipline of AI Engineering",
            "body": (
                "AI Engineering is the systematic discipline of designing, developing, evaluating, "
                "and operating production-grade systems powered by Large Language Models and Foundation Models. "
                "Unlike traditional machine learning, which concentrates primarily on model training and gradient descent, "
                "AI engineering treats foundation models as programmable non-deterministic cognitive engines. "
                "The core engineering concerns revolve around reliability, latency budgets, cost optimization, and safety boundaries.\n\n"
                "A foundational architectural decision in modern AI systems is choosing between Retrieval-Augmented Generation (RAG) "
                "and Fine-Tuning. RAG is preferred when knowledge changes frequently, exact source citations are mandatory, "
                "and proprietary enterprise data must not be leaked into model weights. Fine-tuning is primarily suitable for domain-specific "
                "style adaptation and task-specific instruction adherence."
            ),
            "code": (
                "```python\n"
                "def evaluate_architecture(knowledge_volatility: str, need_citations: bool):\n"
                "    if knowledge_volatility == 'high' or need_citations:\n"
                "        return 'Use Hybrid RAG with Vector + BM25'\n"
                "    return 'Consider Supervised Fine-Tuning'\n"
                "```"
            ),
        },
        {
            "chapter": "Chapter 2: Document Processing & Chunking",
            "section": "2.1 Structure-Aware Chunking Strategies",
            "body": (
                "Document ingestion represents the critical first mile of any RAG pipeline. "
                "Naive chunking strategies split documents blindly every N characters or tokens, leading to broken sentence semantics, "
                "severed code blocks, and orphaned table rows. Structure-aware chunking parses document trees, respecting "
                "chapter headings, sub-sections, code fences, and tabular structures.\n\n"
                "Chunk sizing presents a core engineering trade-off: small chunks (~200 tokens) yield high embedding precision but "
                "risk context fragmentation; large chunks (~1000 tokens) preserve holistic narrative context but introduce noise and "
                "dilute dense vector representations. A sweet spot for technical literature is approximately 500 tokens with 75 tokens overlap."
            ),
            "code": (
                "```python\n"
                "# Structure-Aware Chunk Definition\n"
                "chunk = {\n"
                "    'chunk_size': 500,\n"
                "    'overlap': 75,\n"
                "    'preserve_headings': True,\n"
                "    'atomic_blocks': ['code', 'table']\n"
                "}\n"
                "```"
            ),
        },
        {
            "chapter": "Chapter 3: Retrieval & Indexing",
            "section": "3.1 Dense Vectors vs. Sparse Lexical Search",
            "body": (
                "Dense vector retrieval maps textual passages into continuous high-dimensional geometric spaces, "
                "capturing conceptual semantics, synonyms, and multilingual equivalence. However, dense embeddings struggle with "
                "exact keyword queries, serial numbers, proprietary function names, and out-of-vocabulary technical identifiers.\n\n"
                "Conversely, BM25 (Best Matching 25) calculates exact token frequency (TF) and inverse document frequency (IDF). "
                "Modern AI engineering architectures employ Hybrid Search, running dense vector queries alongside BM25 keyword indices, "
                "fusing candidate rankings to achieve optimal recall across both conceptual and lexical queries."
            ),
            "code": (
                "```python\n"
                "# Hybrid Query Execution\n"
                "dense_hits = qdrant_client.search(collection, query_vector, limit=30)\n"
                "bm25_hits = bm25_index.search(query_tokens, limit=30)\n"
                "fused_results = reciprocal_rank_fusion(dense_hits, bm25_hits, k=60)\n"
                "```"
            ),
        },
        {
            "chapter": "Chapter 4: Reranking & Fusion",
            "section": "4.1 Reciprocal Rank Fusion & Cross-Encoders",
            "body": (
                "Reciprocal Rank Fusion (RRF) merges heterogeneous ranking lists without requiring score normalization. "
                "The score of a document d across rankers is defined as: RRF(d) = sum(1 / (k + rank_i(d))), where k is typically 60.\n\n"
                "Once candidate passages (~30) are retrieved, a Cross-Encoder (such as BGE Reranker) performs joint self-attention "
                "over the query and passage tokens. While bi-encoders produce isolated vectors for fast approximate nearest neighbor search, "
                "cross-encoders evaluate all-to-all token attention, delivering superior precision for top-5 reranking."
            ),
            "code": (
                "```python\n"
                "# RRF Mathematical Implementation\n"
                "def rrf_score(ranks, k=60):\n"
                "    return sum(1.0 / (k + r) for r in ranks)\n"
                "```"
            ),
        },
        {
            "chapter": "Chapter 5: Context Management & Generation",
            "section": "5.1 Prompt Engineering & Citation Enforcement",
            "body": (
                "Context stuffing in LLM prompts often triggers the 'lost in the middle' phenomenon, where models neglect "
                "information positioned in the middle of long prompts. AI engineers structure prompt contexts hierarchically, "
                "placing instructions and critical constraints at prompt boundaries.\n\n"
                "Citation enforcement prevents hallucinations by compelling the LLM to ground assertions in source passages. "
                "The model is instructed to output exact chapter, section, and original physical PDF page numbers, and explicitly "
                "confess when the retrieved book content is insufficient to answer the query."
            ),
            "code": (
                "```python\n"
                "system_rules = [\n"
                "    'Prioritize retrieved book content',\n"
                "    'Never invent citations',\n"
                "    'Cite original PDF page numbers',\n"
                "    'Refuse when information is unavailable'\n"
                "]\n"
                "```"
            ),
        },
        {
            "chapter": "Chapter 6: Evaluation & Observability",
            "section": "6.1 RAGAS Metrics & Continuous Verification",
            "body": (
                "Building production AI systems without automated quantitative evaluation is reckless. "
                "The RAGAS framework evaluates RAG pipelines across four primary pillars: "
                "Context Precision, Context Recall, Faithfulness (absence of hallucination), and Answer Relevance.\n\n"
                "Engineering teams should execute continuous regression suites against a golden dataset of questions, "
                "monitoring retrieval latency, token expenditures, and citation accuracy across releases."
            ),
            "code": (
                "```python\n"
                "# RAGAS Evaluation Metrics\n"
                "metrics = ['faithfulness', 'answer_relevance', 'context_precision', 'context_recall']\n"
                "```"
            ),
        },
    ]

    page_num = 1
    for content in chapters_content:
        # Create multiple pages per chapter to simulate a realistic multi-page book
        pages_for_chapter = max(4, num_pages // len(chapters_content))
        for p_idx in range(pages_for_chapter):
            page = doc.new_page()

            # Insert Header
            header_text = f"AI Engineering   |   {content['chapter']}   |   Page {page_num}"
            page.insert_text(fitz.Point(50, 40), header_text, fontsize=9, color=(0.4, 0.4, 0.4))

            # Title
            if p_idx == 0:
                page.insert_text(fitz.Point(50, 90), content["chapter"], fontsize=18, color=(0.1, 0.2, 0.6))
                page.insert_text(fitz.Point(50, 120), content["section"], fontsize=14, color=(0.2, 0.2, 0.2))
                y_offset = 150
            else:
                page.insert_text(fitz.Point(50, 80), f"{content['section']} (Continued)", fontsize=13, color=(0.2, 0.2, 0.2))
                y_offset = 110

            # Body text
            rect = fitz.Rect(50, y_offset, 550, y_offset + 300)
            page.insert_textbox(rect, content["body"], fontsize=10, lineheight=1.4)

            # Code block on even pages
            if p_idx % 2 == 0 and content.get("code"):
                code_rect = fitz.Rect(50, y_offset + 310, 550, y_offset + 420)
                page.insert_textbox(code_rect, content["code"], fontsize=9, fontname="courier")

            # Footer
            footer_text = f"Page {page_num}"
            page.insert_text(fitz.Point(280, 780), footer_text, fontsize=10, color=(0.5, 0.5, 0.5))

            page_num += 1
            if page_num > num_pages:
                break
        if page_num > num_pages:
            break

    doc.save(str(output_path))
    doc.close()
    return output_path
