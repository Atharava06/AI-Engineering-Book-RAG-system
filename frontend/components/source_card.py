import streamlit as st
from typing import List, Dict, Any
from backend.schemas import Citation, RetrievedChunk


def render_citation_card(citation: Citation, index: int = 1):
    """Renders a clean citation badge and expandable source evidence card."""
    with st.expander(
        f"📌 [{index}] {citation.chapter} — {citation.section} (Page {citation.page_number})",
        expanded=False,
    ):
        st.markdown(f"**Source Document:** `{citation.source_file}`")
        st.markdown(f"**Original Book Page:** **{citation.page_number}**")
        st.markdown(f"**Chapter:** `{citation.chapter}`")
        st.markdown(f"**Section:** `{citation.section}`")
        if citation.relevance_score is not None:
            st.caption(f"Relevance Score: `{citation.relevance_score:.4f}`")
        st.info(f"**Retrieved Evidence:**\n\n\"{citation.snippet}\"")


def render_retrieved_chunk_card(chunk: RetrievedChunk, index: int = 1):
    """Renders detailed chunk breakdown with dense, BM25, and reranker debug scores."""
    st.markdown(
        f"#### #{index} — {chunk.chapter} | {chunk.section} (Page {chunk.page_number})"
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Score", f"{chunk.final_score:.4f}")
    with col2:
        st.metric("Dense Score", f"{chunk.dense_score:.4f}" if chunk.dense_score else "N/A")
    with col3:
        st.metric("BM25 Score", f"{chunk.bm25_score:.4f}" if chunk.bm25_score else "N/A")
    with col4:
        st.metric("Rerank Score", f"{chunk.rerank_score:.4f}" if chunk.rerank_score else "N/A")

    st.code(chunk.text, language="markdown")
    st.divider()
