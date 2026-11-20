import abc
import os
import re
import logging
import time
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
        max_tokens_parameter: str = "max_tokens",
    ):
        self.api_key = api_key or settings.LLM_API_KEY or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.model = model or settings.LLM_MODEL
        self.base_url = (base_url or settings.LLM_BASE_URL or "https://api.openai.com/v1").rstrip("/")
        self.max_tokens_parameter = max_tokens_parameter

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
            self.max_tokens_parameter: max_tokens,
        }

        with httpx.Client(timeout=90.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                logger.error(f"LLM API request failed: {resp.status_code} - {resp.text}")
                raise RuntimeError(f"LLM API error ({resp.status_code}): {resp.text}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class GeminiInteractionsLLMService(BaseLLMService):
    """Calls Gemini models through Google's Interactions API."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/interactions"

    def generate(
        self,
        prompt: str,
        system_instruction: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        payload = {
            "model": self.model,
            "input": [{"type": "user_input", "content": prompt}],
            "system_instruction": system_instruction,
            "store": False,
            "stream": False,
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        }
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Rate-limit responses can describe long quota windows (hours or days);
        # blindly retrying 429s only spends more requests without helping.
        retryable_statuses = {500, 502, 503, 504}
        max_attempts = 4
        response = None
        with httpx.Client(timeout=90.0) as client:
            for attempt in range(max_attempts):
                try:
                    response = client.post(self.base_url, json=payload, headers=headers)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt == max_attempts - 1:
                        raise RuntimeError(
                            f"Gemini request failed after {max_attempts} attempts: {exc}"
                        ) from exc
                    delay = min(8, 2 ** attempt)
                    logger.warning(
                        "Gemini request transport failure; retrying in %ss (attempt %s/%s)",
                        delay, attempt + 1, max_attempts,
                    )
                    time.sleep(delay)
                    continue

                if response.status_code == 200:
                    break
                if response.status_code not in retryable_statuses or attempt == max_attempts - 1:
                    logger.error(
                        "Gemini Interactions request failed: %s - %s",
                        response.status_code, response.text,
                    )
                    if response.status_code == 503:
                        raise RuntimeError(
                            f"Gemini is temporarily overloaded after {max_attempts} attempts. Please try again shortly."
                        )
                    raise RuntimeError(f"Gemini API error ({response.status_code}): {response.text}")

                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = min(8, max(1, int(float(retry_after)))) if retry_after else min(8, 2 ** attempt)
                except ValueError:
                    delay = min(8, 2 ** attempt)
                logger.warning(
                    "Gemini returned HTTP %s; retrying in %ss (attempt %s/%s)",
                    response.status_code, delay, attempt + 1, max_attempts,
                )
                time.sleep(delay)

        if response is None:
            raise RuntimeError("Gemini request ended without a response.")

        data = response.json()
        output_text = data.get("output_text")
        if output_text:
            return output_text

        text_parts = []
        for step in data.get("steps", []):
            if step.get("type") != "model_output":
                continue
            for item in step.get("content", []):
                if item.get("type") == "text" and item.get("text"):
                    text_parts.append(item["text"])
        if text_parts:
            return "\n".join(text_parts)

        raise RuntimeError("Gemini API completed without returning model text.")


class FallbackOfflineLLMService(BaseLLMService):
    """Deterministic extractive answer generator for offline use.

    It cannot reason like a language model, so it builds a useful answer from
    the most query-relevant retrieved passages and avoids inventing concepts.
    """

    @staticmethod
    def _clean_passage(text: str) -> str:
        lines = []
        for line in text.splitlines():
            value = line.strip()
            if not value or value.startswith("["):
                continue
            if re.match(r"^(?:AI Engineering\s*\|\s*)?Chapter\b.*\|\s*Page\s+\d+\s*$", value, re.I):
                continue
            if re.fullmatch(r"Page\s+\d+", value, re.I):
                continue
            lines.append(value)
        return "\n".join(lines).strip()

    @staticmethod
    def _section_name(section: str, content: str) -> str:
        """Repair common page-number metadata misclassified as a section heading."""
        if section and not re.fullmatch(r"(?:page\s*)?\d+", section.strip(), re.I):
            return section.strip()
        heading = re.search(r"(?m)^\s*(\d+(?:\.\d+)+\s+[^\n]{3,100})\s*$", content)
        return heading.group(1).strip() if heading else "Section heading not identified"

    def generate(
        self,
        prompt: str,
        system_instruction: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        # Extract passages if present in prompt.
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

        query_match = re.search(r"User Question:\s*(.*?)\s*\nTarget Persona Level:", prompt, re.DOTALL)
        query = query_match.group(1).strip() if query_match else ""
        query_terms = set(re.findall(r"\b[a-z0-9]{2,}\b", query.lower()))

        records = []
        for book, chapter, section, page, raw_content in passages:
            content = self._clean_passage(raw_content)
            clean_section = self._section_name(section, content)
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
            # Keep short list/table lines together, but do not treat headings as evidence.
            for paragraph in paragraphs:
                if re.fullmatch(r"\d+(?:\.\d+)+\s+.{3,100}", paragraph):
                    continue
                if "```" in paragraph or "|" in paragraph or paragraph.lstrip().startswith(("- ", "* ", "• ")):
                    blocks = [paragraph]
                else:
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph.replace("\n", " ")) if s.strip()]
                    # Pair adjacent sentences to retain context while allowing
                    # relevant material later in a chunk to surface.
                    blocks = [" ".join(sentences[i:i + 2]) for i in range(0, len(sentences), 2)] or [paragraph]
                for block in blocks:
                    words = set(re.findall(r"\b[a-z0-9]{2,}\b", block.lower()))
                    overlap = len(query_terms & words)
                    records.append({
                        "book": book.strip(), "chapter": chapter.strip(),
                        "section": clean_section, "page": page,
                        "text": block, "score": overlap,
                    })

        # Use all retrieved passages, ranked by query-term coverage. Stable order
        # breaks ties so the response is repeatable for offline use.
        records.sort(key=lambda item: item["score"], reverse=True)
        evidence = []
        seen_text = set()
        for record in records:
            normalized = re.sub(r"\W+", " ", record["text"].lower()).strip()
            if len(normalized) < 35 or normalized in seen_text:
                continue
            seen_text.add(normalized)
            evidence.append(record)
            if len(evidence) >= 7:
                break

        if not evidence:
            return (
                "### 1. Direct Answer\nThe retrieved passages do not contain enough readable text to answer this question.\n\n"
                "### 2. Detailed Explanation\nTry re-ingesting the source document or asking about a topic covered in the indexed pages.\n\n"
                "### 3. Important Concepts\n- No relevant concepts could be extracted\n\n"
                "### 4. Example\nNo example was available in the retrieved passages.\n\n"
                "### 5. Sources\n- None"
            )

        first = evidence[0]
        direct = first["text"].replace("\n", " ")
        direct_sentences = re.split(r"(?<=[.!?])\s+", direct)
        direct = " ".join(direct_sentences[:2]).strip()
        if len(direct) > 650:
            direct = direct[:647].rsplit(" ", 1)[0] + "..."

        detail_lines = []
        for item in evidence:
            text = item["text"].replace("\n", " ").strip()
            if len(text) > 550:
                text = text[:547].rsplit(" ", 1)[0] + "..."
            detail_lines.append(f"- **{item['section']} (p. {item['page']}):** {text}")

        concepts = []
        for item in evidence:
            label = item["section"]
            if label != "Section heading not identified" and label not in concepts:
                concepts.append(label)
            if len(concepts) == 5:
                break
        concepts_text = "\n".join(f"- {label}" for label in concepts) or "- See the evidence points above; section headings were not identified in the source."

        code_match = next((re.search(r"```[\s\S]*?```", p[4]) for p in passages if "```" in p[4]), None)
        example = code_match.group(0) if code_match else "No explicit code example was present in the retrieved passages."

        sources = []
        source_keys = set()
        for item in evidence:
            key = (item["chapter"], item["section"], item["page"])
            if key in source_keys:
                continue
            source_keys.add(key)
            sources.append(f"- Chapter: {item['chapter']} | Section: {item['section']} | Page: {item['page']}")

        return (
            f"### 1. Direct Answer\n{direct}\n\n"
            f"### 2. Detailed Explanation\n{chr(10).join(detail_lines)}\n\n"
            f"### 3. Important Concepts\n{concepts_text}\n\n"
            f"### 4. Example\n{example}\n\n"
            f"### 5. Sources\n{chr(10).join(sources)}"
        )


def get_llm_service(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMService:
    """Factory returning configured LLM service."""
    prov = (provider or settings.LLM_PROVIDER).lower()
    key = api_key or settings.LLM_API_KEY
    if not key:
        if prov == "groq":
            key = settings.GROQ_API_KEY
        env_key = {
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
        }.get(prov)
        key = key or (os.environ.get(env_key) if env_key else None)

    if prov == "mock" or not key:
        logger.info("Using FallbackOfflineLLMService (no API key configured).")
        return FallbackOfflineLLMService()

    if prov == "gemini":
        return GeminiInteractionsLLMService(
            api_key=key,
            model=model or settings.LLM_MODEL,
        )

    if prov == "groq":
        return OpenAICompatibleLLMService(
            api_key=key,
            model=model or "openai/gpt-oss-120b",
            base_url="https://api.groq.com/openai/v1",
            max_tokens_parameter="max_completion_tokens",
        )

    return OpenAICompatibleLLMService(
        api_key=key,
        model=model or settings.LLM_MODEL,
        base_url=settings.LLM_BASE_URL,
    )
