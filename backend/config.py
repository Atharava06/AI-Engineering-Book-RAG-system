from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server Settings
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 8501

    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOADS_DIR: Path = BASE_DIR / "data" / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    EVALUATION_DIR: Path = BASE_DIR / "data" / "evaluation"
    STORAGE_DIR: Path = BASE_DIR / "storage"
    QDRANT_STORAGE_PATH: str = str(BASE_DIR / "storage" / "qdrant")
    QDRANT_COLLECTION: str = "ai_engineering_book"

    # LLM Settings
    LLM_PROVIDER: str = "gemini"  # "gemini", "openai", "ollama", "mock"
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # Embedding Settings
    EMBEDDING_PROVIDER: str = "local_fast"  # "bge_m3", "openai", "local_fast"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 64

    # Reranker Settings
    RERANKER_TYPE: str = "cross_encoder"  # "cross_encoder", "bm25_dense_hybrid", "none"

    # Chunking Hyperparameters
    CHUNK_SIZE: int = 500  # approximate tokens
    CHUNK_OVERLAP: int = 75
    MIN_CHUNK_SIZE: int = 100

    # Retrieval Hyperparameters
    TOP_K_DENSE: int = 30
    TOP_K_BM25: int = 30
    RRF_K: int = 60
    FINAL_TOP_K: int = 6
    SIMILARITY_THRESHOLD: float = 0.0

    def init_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for path in [
            self.DATA_DIR,
            self.UPLOADS_DIR,
            self.PROCESSED_DIR,
            self.EVALUATION_DIR,
            self.STORAGE_DIR,
            Path(self.QDRANT_STORAGE_PATH) if self.QDRANT_STORAGE_PATH != ":memory:" else None,
        ]:
            if path:
                path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.init_directories()
