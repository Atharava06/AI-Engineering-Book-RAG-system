import pytest
from pathlib import Path
from backend.utils.helpers import create_sample_ai_engineering_pdf
from backend.ingestion.pdf_loader import PDFLoader
from backend.ingestion.metadata_extractor import MetadataExtractor
from backend.ingestion.ocr_processor import OCRProcessor


@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = tmp_path / "test_ai_engineering.pdf"
    create_sample_ai_engineering_pdf(pdf_path, num_pages=12)
    return pdf_path


def test_pdf_loader_info(sample_pdf):
    loader = PDFLoader(sample_pdf)
    info = loader.get_document_info()

    assert info["total_pages"] == 12
    assert "checksum" in info
    assert len(info["checksum"]) == 64  # SHA-256


def test_pdf_streaming_and_page_numbers(sample_pdf):
    loader = PDFLoader(sample_pdf)
    pages = list(loader.stream_pages(start_page=1, end_page=4))

    assert len(pages) == 4
    # Ensure 1-indexed original PDF page numbers are strictly preserved
    assert [p["page_number"] for p in pages] == [1, 2, 3, 4]
    assert all("chapter" in p for p in pages)
    assert all(len(p["text"]) > 20 for p in pages)


def test_metadata_extractor_cleaning():
    extractor = MetadataExtractor()
    lines = [
        "1",  # Lone page number header
        "AI Engineering",  # Running header
        "Chapter 3: Retrieval & Indexing",
        "3.1 Dense Vectors vs. Sparse Lexical Search",
        "Dense vector retrieval maps text into semantic space.",
        "Page 1",  # Running footer
    ]
    cleaned = extractor.clean_page_lines(lines, page_number=1)

    assert "1" not in cleaned
    assert "AI Engineering" not in cleaned
    assert "Dense vector retrieval maps text into semantic space." in cleaned


def test_ocr_processor():
    ocr = OCRProcessor(min_char_threshold=50)
    # Page with 10 chars and images is considered scanned
    assert ocr.is_scanned_page("short text", page_has_images=True) is True
    # Page with ample text is not scanned
    assert (
        ocr.is_scanned_page(
            "This is a rich text page containing extensive descriptions of AI engineering concepts and architectures.",
            page_has_images=False,
        )
        is False
    )
