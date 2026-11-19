import abc
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import httpx

from backend.config import settings
from backend.schemas import ChatMessage, RetrievedChunk, Citation
from backend.generation.prompt_templates import (
    SYSTEM_PROMPT,
    SUMMARY_PROMPT,
    COMPARISON_PROMPT,
    format_context_for_prompt,
)

logger = logging.getLogger(__name__)


class BaseLLMService(abc.ABC):
    """Abstract interface for LLM response generation."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generates a text completion."""
        pass


class OpenAICompatibleLLMService(BaseLLMService):
    """Calls OpenAI-compatible LLM endpoints (OpenAI, Gemini, Ollama, Groq, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or settings.LLM_API_KEY or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.model = model or settings.LLM_MODEL
        self.base_url = (base_url or settings.LLM_BASE_URL or "https://api.openai.com/v1").rstrip("/")

    def generate(
        self,
        prompt: str,
        system_instruction: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        if not self.api_key:
            raise ValueError("LLM API Key is missing. Please configure LLM_API_KEY in .env or settings.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        with httpx.Client(timeout=90.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                logger.error(f"LLM API request failed: {resp.status_code} - {resp.text}")
                raise RuntimeError(f"LLM API error ({resp.status_code}): {resp.text}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class FallbackOfflineLLMService(BaseLLMService):
    """Deterministic offline answer generator that synthesizes the retrieved chunks
    strictly respecting all system prompt rules and citation constraints.
    Used when no API key is provided or for offline unit tests.
    """

    def generate(
        self,
        prompt: str,
        system_instruction: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        # Extract passages if present in prompt
        passages = re.findall(
            r"--- PASSAGE \d+ ---\s+Book: (.*?)\s+Chapter: (.*?)\s+Section: (.*?)\s+Page Number: (\d+)\s+Content:\s+(.*?)(?=\n\n--- PASSAGE|\Z)",
            prompt,
            re.DOTALL,
        )

        if not passages:
            return (
                "### 1. Direct Answer\nThe retrieved book content does not contain sufficient information to answer this query.\n\n"
                "### 2. Detailed Explanation\nNo matching chapters or sections were found in the indexed material.\n\n"
                "### 3. Important Concepts\n- Insufficient Context\n\n"
                "### 4. Example\nNo explicit example provided in retrieved sections.\n\n"
                "### 5. Sources\n- None"
            )

        # Synthesize top passage
        top_p = passages[0]
        book, chapter, section, page, content = top_p[0], top_p[1], top_p[2], top_p[3], top_p[4].strip()

        # Clean content lines
        clean_lines = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("[")]
        primary_text = " ".join(clean_lines[:4]) if clean_lines else content[:300]

        # Extract code snippet if present
        code_match = re.search(r"```[\s\S]*?```", content)
        example_str = code_match.group(0) if code_match else "No explicit code snippet provided in the retrieved sections."

        # Compile sources
        sources_list = []
        for p in passages[:5]:
            sources_list.append(f"- Chapter: {p[1]} | Section: {p[2]} | Page: {p[3]}")

        sources_formatted = "\n".join(sources_list)

        return (
            f"### 1. Direct Answer\n"
            f"Based on {chapter} ({section}, Page {page}), {primary_text}\n\n"
            f"### 2. Detailed Explanation\n"
            f"The AI Engineering book elaborates that this concept is fundamental to the system architecture described in {chapter}. "
            f"Specifically: {clean_lines[1] if len(clean_lines) > 1 else primary_text}\n\n"
            f"### 3. Important Concepts\n"
            f"- {section} Architecture\n"
            f"- Grounded System Design\n"
            f"- Evaluation & Verification\n\n"
            f"### 4. Example\n"
            f"{example_str}\n\n"
            f"### 5. Sources\n"
            f"{sources_formatted}"
        )


def get_llm_service(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMService:
    """Factory returning configured LLM service."""
    prov = (provider or settings.LLM_PROVIDER).lower()
    key = api_key or settings.LLM_API_KEY or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if prov == "mock" or not key:
        logger.info("Using FallbackOfflineLLMService (no API key configured).")
        return FallbackOfflineLLMService()

    return OpenAICompatibleLLMService(
        api_key=key,
        model=model or settings.LLM_MODEL,
        base_url=settings.LLM_BASE_URL,
    )
