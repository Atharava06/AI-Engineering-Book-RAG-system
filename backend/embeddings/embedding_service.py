import abc
import hashlib
import logging
from typing import List, Optional
import numpy as np

from backend.config import settings

logger = logging.getLogger(__name__)


class BaseEmbeddingService(abc.ABC):
    """Abstract base class for embedding generation."""

    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of documents."""
        pass

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generates an embedding for a query string."""
        pass

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Returns the vector dimensionality."""
        pass


class LocalFastEmbeddingService(BaseEmbeddingService):
    """Deterministic, high-performance local dense semantic embedding model.
    Produces L2-normalized 384-dimensional dense vectors using character and word
    n-gram hashing projections. Requires zero external API calls or GPU/Torch dependencies.
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec

        # Project unigrams and bigrams
        for i, word in enumerate(words):
            # Unigram hash
            h_uni = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx_uni = h_uni % self._dim
            sign_uni = 1.0 if (h_uni // self._dim) % 2 == 0 else -1.0
            vec[idx_uni] += sign_uni

            # Bigram hash
            if i < len(words) - 1:
                bigram = f"{word}_{words[i+1]}"
                h_bi = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
                idx_bi = h_bi % self._dim
                sign_bi = 1.0 if (h_bi // self._dim) % 2 == 0 else -1.0
                vec[idx_bi] += sign_bi * 1.5

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vector(t).tolist() for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_vector(text).tolist()


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """BGE-M3 or MiniLM sentence-transformers embedding backend."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None
        self._dim = 1024 if "bge-m3" in model_name.lower() else 384

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        self._load_model()
        emb = self._model.encode(text, normalize_embeddings=True)
        return emb.tolist()


class OpenAICompatibleEmbeddingService(BaseEmbeddingService):
    """OpenAI / Gemini / vLLM API embedding backend."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dimension: int = 1536,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import httpx
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"input": texts, "model": self.model}
        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embedding_service(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    dimension: Optional[int] = None,
) -> BaseEmbeddingService:
    """Factory creating embedding service according to configuration."""
    prov = (provider or settings.EMBEDDING_PROVIDER).lower()
    dim = dimension or settings.EMBEDDING_DIMENSION

    if prov in ["bge_m3", "sentence_transformers", "hf"]:
        try:
            return SentenceTransformerEmbeddingService(model_name or settings.EMBEDDING_MODEL)
        except Exception as e:
            logger.warning(
                f"SentenceTransformer not available ({e}). Falling back to LocalFastEmbeddingService."
            )
            return LocalFastEmbeddingService(dimension=dim)
    elif prov == "openai":
        if settings.LLM_API_KEY:
            return OpenAICompatibleEmbeddingService(
                api_key=settings.LLM_API_KEY,
                model=model_name or "text-embedding-3-small",
                base_url=settings.LLM_BASE_URL or "https://api.openai.com/v1",
                dimension=dim,
            )
        logger.warning("No API key for OpenAI embedding. Falling back to LocalFastEmbeddingService.")
        return LocalFastEmbeddingService(dimension=dim)

    return LocalFastEmbeddingService(dimension=dim)
