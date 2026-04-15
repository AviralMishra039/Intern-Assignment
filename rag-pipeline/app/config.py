"""Application configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration for the RAG pipeline application."""

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./metadata.db")
    MAX_DOCUMENTS: int = int(os.getenv("MAX_DOCUMENTS", "20"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "40"))


settings = Settings()
