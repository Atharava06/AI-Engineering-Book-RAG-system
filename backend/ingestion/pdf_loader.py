import hashlib
import logging
from pathlib import Path
from typing import Generator, Dict, Any, Optional
import pymupdf as fitz

from backend.ingestion.ocr_processor import OCRProcessor
from backend.ingestion.metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)


class PDFLoader:
    """Streams and parses PDF pages efficiently without loading all 991+ pages into RAM."""

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")
        self.ocr_processor = OCRProcessor()
        self.metadata_extractor = MetadataExtractor()
        self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculates SHA-256 hash of the PDF file to prevent duplicate processing."""
        sha256 = hashlib.sha256()
        with open(self.file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_document_info(self) -> Dict[str, Any]:
        """Gets page count and metadata without parsing all text."""
        doc = fitz.open(self.file_path)
        try:
            total_pages = len(doc)
            metadata = doc.metadata or {}
            toc = doc.get_toc()  # Table of contents if present
            return {
                "file_name": self.file_path.name,
                "file_size_mb": round(self.file_path.stat().st_size / (1024 * 1024), 2),
                "total_pages": total_pages,
                "checksum": self.checksum,
                "author": metadata.get("author", "Unknown"),
                "title": metadata.get("title", "AI Engineering"),
                "has_toc": len(toc) > 0,
                "toc_entries": len(toc),
            }
        finally:
            doc.close()

    def stream_pages(
        self, start_page: int = 1, end_page: Optional[int] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Generator yielding parsed page data one page at a time.
        Page numbers are 1-indexed to match physical PDF pages.
        """
        doc = fitz.open(self.file_path)
        total_pages = len(doc)
        end = min(end_page or total_pages, total_pages)

        # Build TOC mapping if available (page_num -> chapter/section title)
        toc = doc.get_toc()
        toc_map: Dict[int, str] = {}
        for entry in toc:
            lvl, title, p_num = entry[0], entry[1], entry[2]
            if p_num not in toc_map:
                toc_map[p_num] = title

        current_chapter = "Chapter 1: Foundations"
        current_section = "Introduction"

        try:
            for pno in range(start_page - 1, end):
                original_page_number = pno + 1
                page = doc.load_page(pno)

                # Check TOC
                if original_page_number in toc_map:
                    toc_title = toc_map[original_page_number]
                    if any(term in toc_title.lower() for term in ["chapter", "part", "preface", "appendix"]):
                        current_chapter = toc_title
                        current_section = "Overview"
                    else:
                        current_section = toc_title

                # Extract text blocks
                raw_text = page.get_text("text")
                images = page.get_images()
                page_lines = raw_text.splitlines()

                # Detect chapter/section in page text if not strictly from TOC
                detected_ch = self.metadata_extractor.detect_chapter(page_lines)
                if detected_ch:
                    current_chapter = detected_ch
                    current_section = "Introduction"

                detected_sec = self.metadata_extractor.detect_section(page_lines)
                if detected_sec:
                    current_section = detected_sec

                # OCR fallback for scanned pages
                is_scanned = self.ocr_processor.is_scanned_page(raw_text, len(images) > 0)
                ocr_used = False
                if is_scanned:
                    pix = page.get_pixmap()
                    ocr_text, success = self.ocr_processor.ocr_page_pixmap(pix.tobytes())
                    if success and len(ocr_text) > len(raw_text):
                        raw_text = ocr_text
                        page_lines = raw_text.splitlines()
                        ocr_used = True

                # Clean headers and footers
                cleaned_lines = self.metadata_extractor.clean_page_lines(page_lines, original_page_number)
                cleaned_text = "\n".join(cleaned_lines).strip()

                # Extract potential tables
                tables_extracted = []
                try:
                    tabs = page.find_tables()
                    for t in tabs:
                        df_extracted = t.extract()
                        if df_extracted and len(df_extracted) > 1:
                            tables_extracted.append(df_extracted)
                except Exception:
                    pass

                yield {
                    "page_number": original_page_number,
                    "chapter": current_chapter,
                    "section": current_section,
                    "text": cleaned_text,
                    "raw_char_count": len(raw_text),
                    "is_scanned": is_scanned,
                    "ocr_used": ocr_used,
                    "has_tables": len(tables_extracted) > 0,
                    "tables": tables_extracted,
                    "source_file": self.file_path.name,
                }
        finally:
            doc.close()
