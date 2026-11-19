import re
import uuid
import logging
from typing import List, Dict, Any, Optional

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from backend.config import settings
from backend.schemas import DocumentMetadata

logger = logging.getLogger(__name__)


class StructureAwareChunker:
    """Structure-aware chunker that respects chapters, sections, code blocks, tables,
    and sentence boundaries while preserving PDF page citations.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        min_chunk_size: int = settings.MIN_CHUNK_SIZE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        if HAS_TIKTOKEN:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Calculates token count using tiktoken or fallback word-ratio."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        # Fast fallback: 1 token ≈ 4 characters or 0.75 words
        return max(1, int(len(text) / 4))

    def _split_into_atomic_units(self, text: str) -> List[Dict[str, Any]]:
        """Splits page text into atomic semantic blocks (code blocks, paragraphs, lists)."""
        # Protect fenced code blocks
        code_block_pattern = re.compile(r"(```[\s\S]*?```)")
        parts = code_block_pattern.split(text)

        units = []
        for part in parts:
            if not part:
                continue
            if part.startswith("```") and part.endswith("```"):
                units.append({"type": "code", "text": part, "tokens": self.count_tokens(part)})
            else:
                # Split by double newline into paragraphs
                paragraphs = [p.strip() for p in part.split("\n\n") if p.strip()]
                for para in paragraphs:
                    para_tokens = self.count_tokens(para)
                    if para_tokens <= self.chunk_size:
                        units.append({"type": "paragraph", "text": para, "tokens": para_tokens})
                    else:
                        # Split large paragraph by sentence boundaries
                        sentences = re.split(r"(?<=[.!?])\s+", para)
                        for sentence in sentences:
                            s = sentence.strip()
                            if s:
                                units.append(
                                    {
                                        "type": "sentence",
                                        "text": s,
                                        "tokens": self.count_tokens(s),
                                    }
                                )
        return units

    def chunk_page(
        self,
        page_data: Dict[str, Any],
        book_name: str = "AI Engineering",
    ) -> List[DocumentMetadata]:
        """Chunks a single page while maintaining page number, chapter, and section metadata."""
        page_num = page_data["page_number"]
        chapter = page_data.get("chapter", "Unknown Chapter")
        section = page_data.get("section", "General")
        source_file = page_data.get("source_file", "AI_Engineering.pdf")
        text = page_data.get("text", "").strip()

        if not text:
            return []

        # If page has tables, append table representations
        if page_data.get("tables"):
            table_strings = []
            for tbl in page_data["tables"]:
                # Convert list of rows to markdown table
                md_rows = [" | ".join(str(cell or "").strip() for cell in row) for row in tbl]
                if len(md_rows) > 1:
                    header = md_rows[0]
                    separator = " | ".join(["---"] * len(tbl[0]))
                    table_strings.append(f"\n{header}\n{separator}\n" + "\n".join(md_rows[1:]))
            if table_strings:
                text = text + "\n\n" + "\n\n".join(table_strings)

        units = self._split_into_atomic_units(text)
        if not units:
            return []

        chunks: List[DocumentMetadata] = []
        current_units: List[str] = []
        current_tokens = 0
        chunk_idx = 1

        for unit in units:
            unit_tokens = unit["tokens"]
            unit_text = unit["text"]

            if current_tokens + unit_tokens > self.chunk_size and current_units:
                # Flush current chunk
                combined_body = "\n\n".join(current_units).strip()
                # Prepend contextual heading for dense/BM25 retrieval quality
                contextual_header = (
                    f"[{book_name} | {chapter} | {section} | Page {page_num}]\n"
                )
                full_chunk_text = contextual_header + combined_body

                chunk_id = f"p{page_num:04d}_c{chunk_idx:02d}_{uuid.uuid4().hex[:6]}"
                chunks.append(
                    DocumentMetadata(
                        book_name=book_name,
                        chapter=chapter,
                        section=section,
                        page_number=page_num,
                        chunk_id=chunk_id,
                        text=full_chunk_text,
                        source_file=source_file,
                        token_count=self.count_tokens(full_chunk_text),
                        headings_hierarchy=[book_name, chapter, section],
                    )
                )
                chunk_idx += 1

                # Handle overlap by keeping the trailing units that fit in chunk_overlap
                overlap_units = []
                overlap_tokens = 0
                for u in reversed(current_units):
                    u_tok = self.count_tokens(u)
                    if overlap_tokens + u_tok <= self.chunk_overlap:
                        overlap_units.insert(0, u)
                        overlap_tokens += u_tok
                    else:
                        break

                current_units = overlap_units + [unit_text]
                current_tokens = overlap_tokens + unit_tokens
            else:
                current_units.append(unit_text)
                current_tokens += unit_tokens

        # Flush remaining units
        if current_units:
            combined_body = "\n\n".join(current_units).strip()
            if len(combined_body) >= 20:  # Skip trivial fragments
                contextual_header = (
                    f"[{book_name} | {chapter} | {section} | Page {page_num}]\n"
                )
                full_chunk_text = contextual_header + combined_body
                chunk_id = f"p{page_num:04d}_c{chunk_idx:02d}_{uuid.uuid4().hex[:6]}"
                chunks.append(
                    DocumentMetadata(
                        book_name=book_name,
                        chapter=chapter,
                        section=section,
                        page_number=page_num,
                        chunk_id=chunk_id,
                        text=full_chunk_text,
                        source_file=source_file,
                        token_count=self.count_tokens(full_chunk_text),
                        headings_hierarchy=[book_name, chapter, section],
                    )
                )

        return chunks

    def chunk_all_pages(
        self,
        pages: List[Dict[str, Any]],
        book_name: str = "AI Engineering",
    ) -> List[DocumentMetadata]:
        """Chunks all extracted pages across the book, returning unified metadata list."""
        all_chunks: List[DocumentMetadata] = []
        for p in pages:
            page_chunks = self.chunk_page(p, book_name=book_name)
            all_chunks.extend(page_chunks)
        return all_chunks
