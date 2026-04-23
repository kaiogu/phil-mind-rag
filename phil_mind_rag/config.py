"""Application configuration — loaded from environment variables / .env file."""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration. Values come from env vars or .env file."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Generation -----------------------------------------------------
    llm_provider: Literal["openai", "openrouter"] = "openai"
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-5-mini"
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_chat_model: str = "openrouter/free"

    # --- OpenAI ---------------------------------------------------------
    openai_embedding_model: str = "text-embedding-3-small"
    openai_web_search_model: str = "gpt-5-mini"
    openalex_email: str | None = None

    # --- Embeddings -----------------------------------------------------
    embedding_provider: Literal[
        "openai", "sentence_transformers", "openrouter_free_auto"
    ] = "openai"
    sentence_transformers_embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    openrouter_embedding_model_preferences: tuple[str, ...] = (
        "qwen/qwen3-embedding-0.6b",
        "voyage/voyage-3-lite",
        "openai/text-embedding-3-small",
    )

    @field_validator("openrouter_embedding_model_preferences", mode="before")
    @classmethod
    def _parse_embedding_preferences(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    # --- ChromaDB -------------------------------------------------------
    chroma_persist_dir: Path = _PROJECT_ROOT / "data" / "chroma"
    chroma_collection_name: str = "phil_mind_papers"

    # --- Registry -----------------------------------------------------------
    registry_path: Path = _PROJECT_ROOT / "data" / "registry.json"

    # --- Ingestion ------------------------------------------------------
    max_document_size_mb: int = 50
    allowed_extensions: set[str] = {".pdf"}
    chunk_size: int = 512
    chunk_overlap: int = 64
    source_download_dir: Path = _PROJECT_ROOT / "data" / "raw" / "discovered"

    # --- App ------------------------------------------------------------
    gradio_server_port: int = 7860
    log_level: str = "INFO"


def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
