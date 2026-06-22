"""Central configuration, loaded from environment variables / the .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    openai_api_key: str = ""

    agent_model: str = "anthropic:claude-opus-4-1"

    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Toggle the HyDE (Hypothetical Document Embeddings) strategy. When false, the
    # agent loses its hyde_search tool and the /search endpoint rejects method=hyde.
    hyde_enabled: bool = True

    chunk_size: int = 1200
    chunk_overlap: int = 200

    database_url: str = "postgresql://search:search@db:5432/search"


settings = Settings()