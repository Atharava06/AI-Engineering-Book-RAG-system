import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

from backend.config import settings
from backend.ingestion.pdf_loader import PDFLoader

logger = logging.getLogger(__name__)


class DocumentParser:
    """Orchestrates document parsing, progress reporting, and caching."""

    def __init__(self, processed_dir: Optional[Path] = None):
        self.processed_dir = processed_dir or settings.PROCESSED_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def parse_document(
        self,
        pdf_path: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        force_reload: bool = False,
    ) -> List[Dict[str, Any]]:
        """Parses the PDF document, utilizing cached results if available."""
        loader = PDFLoader(pdf_path)
        doc_info = loader.get_document_info()
        total_pages = doc_info["total_pages"]
        checksum = doc_info["checksum"]

        cache_file = self.processed_dir / f"{checksum}_pages.json"

        # Check cache
        if not force_reload and cache_file.exists():
            logger.info(f"Loading cached parsed pages from {cache_file}")
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    pages = json.load(f)
                if progress_callback:
                    progress_callback(total_pages, total_pages, "Loaded from cache")
                return pages
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}. Reparsing...")

        pages: List[Dict[str, Any]] = []
        for page_data in loader.stream_pages():
            pages.append(page_data)
            current_pno = page_data["page_number"]
            if progress_callback and current_pno % 25 == 0:
                progress_callback(
                    current_pno,
                    total_pages,
                    f"Parsed page {current_pno}/{total_pages} ({page_data['chapter']})",
                )

        if progress_callback:
            progress_callback(total_pages, total_pages, "Completed page extraction")

        # Save to cache
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(pages, f, ensure_ascii=False)
            logger.info(f"Saved parsed pages cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not write cache file: {e}")

        return pages
