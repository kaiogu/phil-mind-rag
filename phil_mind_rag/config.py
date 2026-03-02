"""Application configuration — loaded from environment variables / .env file."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration. Values come from env vars or .env file."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI ---------------------------------------------------------
    openai_api_key: SecretStr
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-5-mini"

    # --- ChromaDB -------------------------------------------------------
    chroma_persist_dir: Path = _PROJECT_ROOT / "data" / "chroma"
    chroma_collection_name: str = "phil_mind_papers"

    # --- Ingestion ------------------------------------------------------
    max_document_size_mb: int = 50
    allowed_extensions: set[str] = {".pdf"}
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- App ------------------------------------------------------------
    gradio_server_port: int = 7860
    log_level: str = "INFO"


def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()  # type: ignore[call-arg]
