"""Configuration for the evaluator service, loaded from the environment."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # The search-agent backend (the `api` service in docker-compose).
    api_url: str = "http://api:8000"

    # Anthropic key for the LLM judges (Claude Haiku 4.5).
    anthropic_api_key: str = ""
    judge_model: str = "claude-sonnet-4-6"

    # ViDoRe v3 benchmark subset. We pull a handful of golden pages from one
    # domain and the questions that map onto them. Kept tiny on purpose -- this
    # is a classroom demo, not a leaderboard run.
    vidore_dataset: str = "vidore/vidore_v3_industrial"
    max_documents: int = 10
    max_questions: int = 15

    # Opik (the local instance started with `make opik`). Containers reach it
    # at host.docker.internal:5173. When unset, scores/datasets are skipped.
    opik_url_override: str = ""
    opik_project_name: str = "search-agent"


settings = Settings()