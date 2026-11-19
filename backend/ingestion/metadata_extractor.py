import re
from typing import List, Tuple, Optional


class MetadataExtractor:
    """Extracts chapter, section headings, and strips running headers/footers."""

    CHAPTER_PATTERNS = [
        re.compile(r"^(?:CHAPTER|Chapter)\s+(\d+|[IVXLCDM]+)[\s:.\-]+(.*)$", re.IGNORECASE),
        re.compile(r"^(\d+)\.\s+([A-Z][A-Za-z0-9\s,\-:]{3,60})$"),
        re.compile(r"^(?:PART|Part)\s+(\d+|[IVXLCDM]+)[\s:.\-]+(.*)$", re.IGNORECASE),
        re.compile(r"^(?:APPENDIX|Appendix)\s+([A-Z\d]+)[\s:.\-]+(.*)$", re.IGNORECASE),
        re.compile(r"^(Preface|Introduction|Foreword|Index|Glossary|Bibliography)$", re.IGNORECASE),
    ]

    SECTION_PATTERNS = [
        re.compile(r"^(\d+\.\d+)\s+([A-Z][A-Za-z0-9\s,\-:]{2,60})$"),
        re.compile(r"^(\d+\.\d+\.\d+)\s+([A-Z][A-Za-z0-9\s,\-:]{2,60})$"),
        re.compile(r"^[A-Z][A-Za-z0-9\s]{3,40}:?$"),
    ]

    HEADER_FOOTER_PATTERNS = [
        re.compile(r"^\s*\d+\s*$"),  # Lone page numbers
        re.compile(r"^\s*AI Engineering\s*$", re.IGNORECASE),
        re.compile(r"^\s*Chapter\s+\d+\b.*$", re.IGNORECASE),
        re.compile(r"^\s*O'REILLY\s*$", re.IGNORECASE),
        re.compile(r"^\s*www\..*\.com\s*$", re.IGNORECASE),
    ]

    def detect_chapter(self, text_lines: List[str]) -> Optional[str]:
        """Detects if any line in the top portion of the page starts a new chapter."""
        # Check first 8 non-empty lines of page
        sample_lines = [l.strip() for l in text_lines[:10] if l.strip()]
        for line in sample_lines:
            for pattern in self.CHAPTER_PATTERNS:
                match = pattern.match(line)
                if match:
                    groups = match.groups()
                    if len(groups) == 2:
                        return f"Chapter {groups[0]}: {groups[1].strip()}"
                    elif len(groups) == 1:
                        return f"Chapter: {groups[0].strip()}"
        return None

    def detect_section(self, text_lines: List[str]) -> Optional[str]:
        """Detects prominent section headings on the page."""
        sample_lines = [l.strip() for l in text_lines if l.strip()]
        for line in sample_lines:
            for pattern in self.SECTION_PATTERNS:
                match = pattern.match(line)
                if match:
                    return line
        return None

    def clean_page_lines(self, lines: List[str], page_number: int) -> List[str]:
        """Removes header and footer noise from page text lines."""
        cleaned = []
        total_lines = len(lines)
        for i, line in enumerate(lines):
            stripped = line.strip()
            # If top 2 lines or bottom 2 lines, check header/footer patterns
            if i < 3 or i >= total_lines - 3:
                # Check lone page numbers
                if stripped == str(page_number) or stripped == str(page_number - 1) or stripped == str(page_number + 1):
                    continue
                is_noise = False
                for p in self.HEADER_FOOTER_PATTERNS:
                    if p.match(stripped):
                        is_noise = True
                        break
                if is_noise:
                    continue
            cleaned.append(line)
        return cleaned
