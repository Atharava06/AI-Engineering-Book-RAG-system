import streamlit as st
from backend.schemas import ChatRequest, ChatMessage
from frontend.components.source_card import render_citation_card


def render_chat_interface(rag_pipeline):
    """Renders conversational Chat with Book interface."""
    st.markdown("### 💬 Chat with the AI Engineering Book")
    st.caption("Ask questions, explore architectures, or request code patterns with verifiable page citations.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Controls row
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        persona = st.selectbox(
            "Explanation Persona",
            ["Expert AI Engineer", "Intermediate Practitioner", "Beginner / Conceptual"],
            index=0,
        )
    with col2:
        filter_chapter = st.text_input("Filter by Chapter (Optional)", placeholder="e.g. Chapter 3")
    with col3:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()

    # Display prior conversation messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                st.markdown("**Verified Book Sources:**")
                for idx, cit in enumerate(msg["citations"], start=1):
                    render_citation_card(cit, idx)

    # Chat user input
    user_query = st.chat_input("Ask any question about AI Engineering (e.g., 'Compare dense vector search and BM25')...")

    if user_query:
        # Display user prompt
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Prepare request
        history_msgs = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in st.session_state.messages[:-1]
        ]

        req = ChatRequest(
            query=user_query,
            history=history_msgs,
            chapter_filter=filter_chapter if filter_chapter else None,
            persona_level=persona.lower().split()[0],
        )

        with st.chat_message("assistant"):
            with st.spinner("Searching book passages, fusing rankings & generating answer..."):
                try:
                    res = rag_pipeline.answer_question(req)

                    st.markdown(res.answer)

                    if res.citations:
                        st.markdown("---")
                        st.markdown(f"**📚 Verified Citations ({len(res.citations)}):**")
                        for idx, cit in enumerate(res.citations, start=1):
                            render_citation_card(cit, idx)

                    st.caption(f"⚡ Latency: {res.latency_ms} ms | Model: {res.model_used}")

                    # Save to state
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": res.answer,
                            "citations": res.citations,
                        }
                    )
                except Exception as e:
                    st.error(f"Error answering question: {e}")
