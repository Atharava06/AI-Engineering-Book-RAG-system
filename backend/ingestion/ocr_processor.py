import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Check optional OCR tools
try:
    import pytesseract
    from PIL import Image
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


class OCRProcessor:
    """Processes scanned or low-text PDF pages using OCR when required."""

    def __init__(self, min_char_threshold: int = 50):
        self.min_char_threshold = min_char_threshold

    def is_scanned_page(self, page_text: str, page_has_images: bool) -> bool:
        """Determines if a page is likely a scanned image page."""
        clean_text = page_text.strip()
        # If very little text is extracted but the page contains raster graphics/images
        if len(clean_text) < self.min_char_threshold and page_has_images:
            return True
        return False

    def ocr_page_pixmap(self, pixmap_bytes: bytes) -> Tuple[str, bool]:
        """Runs OCR on page rendered image if OCR is available.
        Returns: (ocr_text, success_flag)
        """
        if not HAS_PYTESSERACT:
            return (
                "[Notice: Scanned page detected. Install pytesseract/tesseract-ocr for image OCR.]",
                False,
            )
        try:
            import io
            image = Image.open(io.BytesIO(pixmap_bytes))
            text = pytesseract.image_to_string(image)
            return text.strip(), True
        except Exception as e:
            logger.warning(f"OCR processing failed: {e}")
            return f"[OCR Error: {e}]", False
