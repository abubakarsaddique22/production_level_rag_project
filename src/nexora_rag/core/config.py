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

    qdrant_url: str = "http://localhost:6333"
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

    llm_provider: str = "anthropic"  # anthropic | openai | ollama
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.1

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

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

    jwt_secret: str = "change-me-in-env"
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 60

    # ============================================================
    # Observability - Langfuse
    # ============================================================

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


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