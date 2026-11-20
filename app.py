import os
import time
import streamlit as st
from pathlib import Path

# Configure page layout and aesthetics
st.set_page_config(
    page_title="AI Engineering Book Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

from backend.config import settings
from backend.main import (
    rag_pipeline,
    vector_store,
    bm25_index,
    evaluator,
    run_ingestion_job,
    ingestion_state,
)
from backend.schemas import (
    SearchRequest,
    SummarizeRequest,
    CompareRequest,
    EvaluateRequest,
)
from backend.utils.helpers import create_sample_ai_engineering_pdf
from backend.generation.llm_service import get_llm_service
from frontend.components.sidebar import render_sidebar
from frontend.components.chat_view import render_chat_interface
from frontend.components.source_card import render_citation_card, render_retrieved_chunk_card

# Top header
page = render_sidebar()

# -----------------------------------------------------------------------------
# PAGE 1: CHAT WITH BOOK
# -----------------------------------------------------------------------------
if page == "💬 Chat with Book":
    render_chat_interface(rag_pipeline)

# -----------------------------------------------------------------------------
# PAGE 2: EXPLORE CHAPTERS
# -----------------------------------------------------------------------------
elif page == "📖 Explore Chapters":
    st.markdown("### 📖 Explore AI Engineering Chapters")
    st.caption("Inspect indexed chapters, original page spans, and constituent sections.")

    chapters_resp = rag_pipeline.get_indexed_chapters()

    if not chapters_resp.chapters:
        st.warning("No chapters have been indexed yet. Head to **📂 Document Management** to upload and index the book.")
        if st.button("🚀 Seed Sample Book (35 Pages) Now"):
            sample_pdf = Path(settings.UPLOADS_DIR) / "AI_Engineering_Sample.pdf"
            create_sample_ai_engineering_pdf(sample_pdf, num_pages=35)
            run_ingestion_job(sample_pdf)
            st.success("Ingestion started! Check progress below or reload.")
            st.rerun()
    else:
        st.info(f"Book: **{chapters_resp.book_name}** | Total Pages Indexed: **{chapters_resp.total_pages}** | Chapters: **{len(chapters_resp.chapters)}**")

        for ch in chapters_resp.chapters:
            with st.expander(f"📚 {ch.chapter_title} (Pages {ch.start_page}–{ch.end_page}) — {ch.chunk_count} Chunks"):
                st.markdown(f"**Original Page Range:** {ch.start_page} to {ch.end_page}")
                st.markdown(f"**Indexed Chunks:** {ch.chunk_count}")
                if ch.sections:
                    st.markdown("**Key Sections & Headings:**")
                    for sec in ch.sections:
                        st.markdown(f"- `{sec}`")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Summarize {ch.chapter_title[:20]}...", key=f"sum_{ch.chapter_title}"):
                        st.session_state["target_chapter_to_summarize"] = ch.chapter_title
                        st.info("Navigate to '📑 Summarize Chapters' to generate summary.")

# -----------------------------------------------------------------------------
# PAGE 3: SEARCH CONCEPTS
# -----------------------------------------------------------------------------
elif page == "🔍 Search Concepts":
    st.markdown("### 🔍 Hybrid Concept Search")
    st.caption("Perform dense semantic + BM25 keyword search with Reciprocal Rank Fusion and Cross-Encoder reranking.")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Enter search query or concept", placeholder="e.g. cross-encoder reranking latency")
    with col2:
        top_k = st.slider("Top Results", min_value=1, max_value=15, value=5)

    with st.expander("Filter Options (Chapter / Page Range)"):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            ch_filter = st.text_input("Chapter Filter", placeholder="e.g. Chapter 4")
        with col_f2:
            p_start = st.number_input("Start Page", min_value=1, max_value=2000, value=None)
        with col_f3:
            p_end = st.number_input("End Page", min_value=1, max_value=2000, value=None)

    if st.button("🔎 Search Book", use_container_width=True) and search_query:
        with st.spinner("Executing hybrid retrieval..."):
            req = SearchRequest(
                query=search_query,
                top_k=top_k,
                chapter_filter=ch_filter if ch_filter else None,
                page_start=int(p_start) if p_start else None,
                page_end=int(p_end) if p_end else None,
                use_reranker=True,
            )
            res = rag_pipeline.retrieval.search(
                query=req.query,
                top_k=req.top_k,
                chapter_filter=req.chapter_filter,
                page_start=req.page_start,
                page_end=req.page_end,
                use_reranker=True,
            )

            st.success(f"Retrieved {len(res.results)} chunks from {res.total_candidates} candidates in {res.latency_ms} ms")

            for i, chunk in enumerate(res.results, start=1):
                render_retrieved_chunk_card(chunk, i)

# -----------------------------------------------------------------------------
# PAGE 4: SUMMARIZE CHAPTERS
# -----------------------------------------------------------------------------
elif page == "📑 Summarize Chapters":
    st.markdown("### 📑 Chapter Summarizer")
    st.caption("Synthesize comprehensive technical summaries, principles, and key takeaways for any chapter.")

    chapters_resp = rag_pipeline.get_indexed_chapters()
    chapter_names = [ch.chapter_title for ch in chapters_resp.chapters] if chapters_resp.chapters else ["Chapter 1: Foundations of AI Engineering"]

    default_idx = 0
    if "target_chapter_to_summarize" in st.session_state and st.session_state["target_chapter_to_summarize"] in chapter_names:
        default_idx = chapter_names.index(st.session_state["target_chapter_to_summarize"])

    col1, col2 = st.columns([2, 1])
    with col1:
        chosen_chapter = st.selectbox("Select Chapter to Summarize", chapter_names, index=default_idx)
    with col2:
        detail_level = st.selectbox("Detail Level", ["Detailed Architecture", "Executive Concise", "Key Takeaways"])

    if st.button("⚡ Generate Chapter Summary", use_container_width=True):
        with st.spinner(f"Retrieving and synthesizing {chosen_chapter}..."):
            req = SummarizeRequest(chapter=chosen_chapter, detail_level=detail_level)
            summary_res = rag_pipeline.summarize_chapter(req)

            st.markdown(summary_res.summary)

            st.markdown("---")
            st.markdown("#### 🎯 Key Architectural Takeaways")
            for t in summary_res.key_takeaways:
                st.markdown(f"- {t}")

            if summary_res.citations:
                st.markdown("#### 📚 Verified Source Passages")
                for idx, cit in enumerate(summary_res.citations, start=1):
                    render_citation_card(cit, idx)

# -----------------------------------------------------------------------------
# PAGE 5: COMPARE CONCEPTS
# -----------------------------------------------------------------------------
elif page == "⚖️ Compare Concepts":
    st.markdown("### ⚖️ Technical Concept Comparison")
    st.caption("Compare two concepts or architectural choices using retrieved evidence and trade-off matrices.")

    col1, col2 = st.columns(2)
    with col1:
        concept_a = st.text_input("Concept A", value="Retrieval-Augmented Generation (RAG)")
    with col2:
        concept_b = st.text_input("Concept B", value="Fine-Tuning")

    if st.button("⚖️ Run Comparative Analysis", use_container_width=True):
        with st.spinner(f"Comparing '{concept_a}' and '{concept_b}' across the book..."):
            req = CompareRequest(concept_a=concept_a, concept_b=concept_b)
            comp_res = rag_pipeline.compare_concepts(req)

            st.markdown(comp_res.direct_comparison)

            st.markdown("---")
            st.markdown("#### ⚖️ Engineering Synthesis & Trade-offs")
            st.info(comp_res.synthesis_and_tradeoffs)

            if comp_res.citations:
                st.markdown("#### 📚 Sources")
                for idx, cit in enumerate(comp_res.citations, start=1):
                    render_citation_card(cit, idx)

# -----------------------------------------------------------------------------
# PAGE 6: LEARNING ASSISTANT
# -----------------------------------------------------------------------------
elif page == "🎓 Learning Assistant":
    st.markdown("### 🎓 AI Engineering Learning Assistant")
    st.caption("Multi-tier concept explanations and interactive self-quizzes grounded in the book.")

    tab1, tab2 = st.tabs(["💡 Multi-Level Concept Explainer", "📝 Self-Quiz Generator"])

    with tab1:
        concept_to_explain = st.text_input("Enter concept to learn", value="Dense Vector vs BM25 Hybrid Search")
        level = st.radio("Proficiency Level", ["Beginner (Intuitive Analogy)", "Intermediate (Implementation Details)", "Advanced (Architectural Trade-offs)"], horizontal=True)

        if st.button("🎓 Explain Concept"):
            with st.spinner("Generating grounded multi-tier explanation..."):
                lvl_key = "beginner" if "Beginner" in level else ("intermediate" if "Intermediate" in level else "expert")
                from backend.schemas import ChatRequest
                res = rag_pipeline.answer_question(ChatRequest(
                    query=f"Explain {concept_to_explain} tailored for a {level} audience. Provide clear examples from the AI Engineering book.",
                    persona_level=lvl_key,
                ))
                st.markdown(res.answer)

    with tab2:
        quiz_topic = st.text_input("Quiz Topic or Chapter", value="RAG Architecture and Evaluation")
        num_q = st.slider("Number of Questions", min_value=1, max_value=5, value=3)

        if st.button("📝 Generate Interactive Quiz"):
            with st.spinner("Generating questions from book content..."):
                from backend.schemas import ChatRequest
                res = rag_pipeline.answer_question(ChatRequest(
                    query=f"Generate a {num_q}-question technical multiple choice quiz on '{quiz_topic}' based on the AI Engineering book. For each question, provide 4 options (A, B, C, D), indicate the correct answer, and cite the chapter/page.",
                ))
                st.markdown(res.answer)

# -----------------------------------------------------------------------------
# PAGE 7: DOCUMENT MANAGEMENT
# -----------------------------------------------------------------------------
elif page == "📂 Document Management":
    st.markdown("### 📂 Document Ingestion & Management")
    st.caption("Upload your 991-page 'AI Engineering' digital PDF or generate a verified sample book to index.")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 1. Upload Book PDF")
        uploaded_file = st.file_uploader("Upload 'AI Engineering' PDF", type=["pdf"])

        if uploaded_file is not None:
            if st.button("💾 Save & Ingest Uploaded PDF", type="primary"):
                upload_dir = Path(settings.UPLOADS_DIR)
                upload_dir.mkdir(parents=True, exist_ok=True)
                dest_file = upload_dir / uploaded_file.name
                with open(dest_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved {uploaded_file.name} to {dest_file}")
                run_ingestion_job(dest_file)
                st.rerun()

    with col2:
        st.markdown("#### 2. Quick Demo / Sample Seeder")
        st.caption("Generates a realistic 35-page multi-chapter AI Engineering book with code, tables, and sections.")
        if st.button("⚡ Generate & Ingest 35-Page Sample Book"):
            sample_file = Path(settings.UPLOADS_DIR) / "AI_Engineering_Sample.pdf"
            create_sample_ai_engineering_pdf(sample_file, num_pages=35)
            run_ingestion_job(sample_file)
            st.success("Sample book created and ingestion triggered!")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📊 Live Ingestion Status")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Status", ingestion_state.get("status", "idle").upper())
    with col_s2:
        st.metric("Pages Processed", f"{ingestion_state.get('processed_pages', 0)}/{ingestion_state.get('total_pages', 0)}")
    with col_s3:
        st.metric("Chunks Generated", ingestion_state.get("total_chunks", 0))
    with col_s4:
        st.metric("Indexed Chapters", ingestion_state.get("total_chapters", 0))

    progress = float(ingestion_state.get("progress_percentage", 0.0)) / 100.0
    st.progress(min(1.0, max(0.0, progress)))
    st.info(f"**Current Operation:** {ingestion_state.get('message', 'Ready')}")

    if ingestion_state.get("error"):
        st.error(f"Error: {ingestion_state['error']}")

# -----------------------------------------------------------------------------
# PAGE 8: EVALUATION DASHBOARD
# -----------------------------------------------------------------------------
elif page == "📊 Evaluation Dashboard":
    st.markdown("### 📊 RAG Evaluation & Observability Dashboard")
    st.caption("Measure Context Precision, Faithfulness, Relevance, Citation Accuracy, and Latency against the benchmark dataset.")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Benchmark Dataset contains representative AI Engineering queries covering chapters 1 through 6.")
    with col2:
        sample_size = st.slider("Evaluation Questions", min_value=2, max_value=7, value=5)

    if st.button("🚀 Run Comprehensive Evaluation Benchmark", type="primary", use_container_width=True):
        with st.spinner("Running benchmark suite and calculating faithfulness, precision, and citation metrics..."):
            result = evaluator.run_benchmark(sample_size=sample_size)

            st.markdown(f"### Overall System Benchmark Score: **{result.overall_score * 100:.1f}%**")
            st.caption(f"Evaluated {result.evaluated_questions} questions | Average Latency: {result.latency_avg_ms:.1f} ms")

            # Metrics cards
            cols = st.columns(len(result.metrics))
            for i, m in enumerate(result.metrics):
                with cols[i]:
                    st.metric(
                        m.name,
                        f"{m.score * 100:.1f}%",
                        delta=f"Threshold: {m.benchmark_threshold * 100:.0f}%",
                        delta_color="normal" if m.passed else "inverse",
                    )

            st.markdown("---")
            st.markdown("#### 🔬 Per-Question Detailed Breakdown")
            import pandas as pd
            df_eval = pd.DataFrame(result.sample_evaluations)
            st.dataframe(df_eval, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 9: SETTINGS
# -----------------------------------------------------------------------------
elif page == "⚙️ Settings":
    st.markdown("### ⚙️ System & Model Configuration")
    st.caption("Adjust LLM providers, embedding backends, vector store parameters, and retrieval hyperparameters.")

    active_llm = rag_pipeline.llm_service
    if getattr(active_llm, "api_key", None):
        st.success(f"LLM active: {active_llm.model} via {settings.LLM_PROVIDER}")
    else:
        st.warning("LLM API key is not loaded. The app is using its offline fallback generator.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### LLM Provider Configuration")
        provider_options = ["gemini", "groq", "openai", "ollama", "mock"]
        provider_labels = {
            "gemini": "Google Gemini",
            "groq": "Groq",
            "openai": "OpenAI",
            "ollama": "Ollama (local)",
            "mock": "Offline fallback",
        }
        provider = st.selectbox(
            "LLM Provider",
            provider_options,
            format_func=lambda value: provider_labels[value],
            index=provider_options.index(settings.LLM_PROVIDER)
            if settings.LLM_PROVIDER in provider_options
            else 0,
        )
        provider_models = {
            "gemini": "gemini-3.8-flash",
            "groq": "openai/gpt-oss-120b",
        }
        model_default = settings.LLM_MODEL if provider == settings.LLM_PROVIDER else provider_models.get(provider, settings.LLM_MODEL)
        model_name = st.text_input(
            "LLM Model Name",
            value=model_default,
            key=f"llm_model_name_{provider}",
        )
        if provider == "groq":
            key_default = settings.GROQ_API_KEY
        elif provider == settings.LLM_PROVIDER:
            key_default = settings.LLM_API_KEY
        else:
            key_default = ""
        api_key = st.text_input(
            f"{provider_labels[provider]} API Key (stored only in this app session)",
            type="password",
            value=key_default or "",
            key=f"llm_api_key_{provider}",
            placeholder="Paste your provider API key",
        )
        if provider == "groq":
            st.markdown("Get a Groq key from the [Groq Console](https://console.groq.com/keys).")
        temperature = st.slider("Generation Temperature", min_value=0.0, max_value=1.0, value=settings.LLM_TEMPERATURE, step=0.05)

    with col2:
        st.markdown("#### Retrieval & Chunking Hyperparameters")
        emb_provider = st.selectbox("Embedding Provider", ["local_fast", "bge_m3", "openai"], index=0)
        chunk_size = st.number_input("Chunk Size (Tokens)", min_value=200, max_value=1500, value=settings.CHUNK_SIZE)
        chunk_overlap = st.number_input("Chunk Overlap (Tokens)", min_value=20, max_value=300, value=settings.CHUNK_OVERLAP)
        top_k_final = st.number_input("Final Top-K Chunks to LLM", min_value=3, max_value=15, value=settings.FINAL_TOP_K)

    if st.button("💾 Apply Settings"):
        settings.LLM_PROVIDER = provider
        settings.LLM_MODEL = model_name
        settings.LLM_API_KEY = api_key.strip() or None
        provider_base_urls = {
            "groq": "https://api.groq.com/openai/v1",
            "openai": "https://api.openai.com/v1",
        }
        if provider in provider_base_urls:
            settings.LLM_BASE_URL = provider_base_urls[provider]
        settings.LLM_TEMPERATURE = temperature
        settings.CHUNK_SIZE = chunk_size
        settings.CHUNK_OVERLAP = chunk_overlap
        settings.FINAL_TOP_K = top_k_final
        rag_pipeline.llm_service = get_llm_service(
            provider=provider,
            api_key=settings.LLM_API_KEY,
            model=model_name,
        )
        if getattr(rag_pipeline.llm_service, "api_key", None):
            st.success(f"Settings applied. Active LLM: {model_name}.")
        else:
            st.warning("Settings applied, but no API key is configured. The offline fallback remains active.")
