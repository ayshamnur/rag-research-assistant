"""Central configuration for the RAG assistant backend.

All values can be overridden with environment variables (see .env.example).
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Anthropic (or swap for any OpenAI-compatible provider)
    anthropic_api_key: str = ""
    generation_model: str = "claude-sonnet-5"

    # Retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_dir: str = str(BASE_DIR / "data" / "chroma")
    corpus_path: str = str(BASE_DIR / "data" / "corpus.json")
    chunk_size_tokens: int = 220
    chunk_overlap_tokens: int = 40
    top_k_dense: int = 8
    top_k_bm25: int = 8
    top_k_final: int = 5

    # Hybrid retrieval fusion weight (0 = pure BM25, 1 = pure dense)
    dense_weight: float = 0.6

    # API
    cors_allow_origins: list[str] = ["*"]


settings = Settings()
