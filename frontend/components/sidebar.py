import streamlit as st
from backend.config import settings


def render_sidebar():
    """Renders the primary navigation sidebar and active book status."""
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding-bottom: 15px;">
            <h2 style="margin-bottom: 0px; color: #1E88E5;">📘 AI Engineering</h2>
            <p style="font-size: 13px; color: #888; margin-top: 2px;">Advanced Hybrid RAG Assistant</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    menu_options = [
        "💬 Chat with Book",
        "📖 Explore Chapters",
        "🔍 Search Concepts",
        "📑 Summarize Chapters",
        "⚖️ Compare Concepts",
        "🎓 Learning Assistant",
        "📂 Document Management",
        "📊 Evaluation Dashboard",
        "⚙️ Settings",
    ]

    selected_page = st.sidebar.radio(
        "Navigation",
        menu_options,
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # Document & Vector status pill
    try:
        from backend.main import vector_store, bm25_index
        v_count = vector_store.count()
        b_count = bm25_index.corpus_size
        st.sidebar.markdown(f"**Index Status:** 🟢 Active")
        st.sidebar.markdown(f"• **Qdrant Vectors:** `{v_count}`")
        st.sidebar.markdown(f"• **BM25 Chunks:** `{b_count}`")
    except Exception:
        st.sidebar.markdown(f"**Index Status:** ⚪ Initializing")

    st.sidebar.markdown("---")
    st.sidebar.caption("Grounded strictly in the 991-page digital book *AI Engineering*.")

    return selected_page
