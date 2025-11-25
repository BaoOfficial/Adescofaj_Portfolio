"""
Configuration management for the AI chatbot backend.
Loads settings from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str
    model_name: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-3-small"

    # Server Configuration
    backend_port: int = 8000
    frontend_url: str = "http://localhost:8080"

    # Vector Store Configuration
    chroma_persist_dir: str = "./vector_store/chroma_db"
    knowledge_base_path: str = "./knowledge_base/amirlahi_portfolio.md"

    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    similarity_threshold: float = 0.6

    # Hybrid Search Weights
    semantic_weight: float = 0.6
    bm25_weight: float = 0.4

    class Config:
        env_file = ".env"
        case_sensitive = False
        protected_namespaces = ('settings_',)


# Global settings instance
settings = Settings()
