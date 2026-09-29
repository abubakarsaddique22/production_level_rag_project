"""
Central configuration for the Nexora RAG project.

All configurable values are loaded from environment variables
(via .env) or fall back to the defaults below.

Secrets should always come from environment variables.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        case_sensitive=False,
        extra="ignore",
    )

    # ============================================================
    # Environment
    # ============================================================

    env: str = "dev"  # dev | test | prod
    log_level: str = "INFO"

    # ============================================================
    # Data / Ingestion
    # ============================================================

    data_processed_dir: Path = Path("data/processed")
    document_registry_path: Path = Path("data/document_registry.yaml")

    # Minimum selectable text characters before falling back to OCR
    min_selectable_chars_per_page: int = 100

    # ============================================================
    # Vector DB - Qdrant
    # ============================================================
    qdrant_url: str
    qdrant_api_key: str | None = None
    collection: str = "nexora_kb"

    # ============================================================
    # Embeddings
    # ============================================================

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_batch_size: int = 32

    # ============================================================
    # Chunking
    # ============================================================

    chunk_tokens: int = 600
    chunk_overlap: int = 80

    # ============================================================
    # Retrieval
    # ============================================================

    top_k_retrieve: int = 30
    top_k_final: int = 5
    rrf_k: int = 60

    # ============================================================
    # LLM
    # ============================================================

    # llm_provider: str = "openrouter"
    # llm_model: str = "google/gemma-4-26b-a4b-it:free"
    # llm_temperature: float = 0.1

    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    # groq_model: str = "qwen/qwen3.8-27b"

    ollama_base_url: str = "http://localhost:11434"

    # ============================================================
    # PostgreSQL
    # ============================================================

    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/nexora"
    )

    # ============================================================
    # Redis
    # ============================================================

    redis_url: str = "redis://localhost:6379/0"

      # ============================================================
    # Authentication
    # ============================================================

    jwt_secret: str = "CHANGE_ME_IN_ENV"   # load from .env: RAG_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # ============================================================
    # Observability - Langfuse
    # ============================================================

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"




    answer_cache_ttl_seconds: int = 3600
# ================================================================
# Cached settings instance
# ================================================================

@lru_cache
def get_settings() -> Settings:
    """
    Create and cache the application settings.

    Settings are loaded once from environment variables / .env
    and reused throughout the application.
    """
    return Settings()


settings = get_settings()