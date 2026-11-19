import pytest
from backend.chunking.chunker import StructureAwareChunker


def test_structure_aware_chunking_preserves_metadata():
    chunker = StructureAwareChunker(chunk_size=150, chunk_overlap=30)
    page_data = {
        "page_number": 42,
        "chapter": "Chapter 4: Reranking & Fusion",
        "section": "4.1 Reciprocal Rank Fusion",
        "source_file": "AI_Engineering.pdf",
        "text": (
            "Reciprocal Rank Fusion (RRF) is a method that combines multiple search results "
            "into a single unified ranking without needing score normalization. "
            "It computes the reciprocal of rank plus constant k. "
            "Cross-encoders are then used for final top-5 reranking."
        ),
    }

    chunks = chunker.chunk_page(page_data, book_name="AI Engineering")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.page_number == 42
        assert c.chapter == "Chapter 4: Reranking & Fusion"
        assert c.section == "4.1 Reciprocal Rank Fusion"
        assert c.source_file == "AI_Engineering.pdf"
        assert "Chapter 4: Reranking & Fusion" in c.text
        assert "Page 42" in c.text


def test_chunking_protects_code_blocks():
    chunker = StructureAwareChunker(chunk_size=200, chunk_overlap=20)
    code_block = (
        "```python\n"
        "def rrf(ranks, k=60):\n"
        "    return sum(1.0 / (k + r) for r in ranks)\n"
        "```"
    )
    page_data = {
        "page_number": 15,
        "chapter": "Chapter 4: Reranking",
        "section": "Code Examples",
        "source_file": "AI_Engineering.pdf",
        "text": f"Below is the implementation:\n\n{code_block}\n\nNotice the simplicity of the formula.",
    }

    chunks = chunker.chunk_page(page_data)
    assert len(chunks) >= 1
    # Check that code block is kept intact
    has_code = any("def rrf(ranks, k=60):" in c.text for c in chunks)
    assert has_code is True
