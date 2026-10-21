"""Application configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable (true/1/yes/on are truthy)."""
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Central configuration for the RAG pipeline application."""

    # --- Google Gemini (LLM + embeddings) ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_LLM_MODEL: str = os.getenv("GEMINI_LLM_MODEL", "gemini-3.8-flash")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    # gemini-embedding-001 outputs 3072-d by default; Matryoshka training lets us
    # truncate to 768-d with little quality loss and 4x less storage.
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    # Free tier is rate limited (~100 requests/min, ~30k tokens/min) so batches are small.
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))

    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    # Collection is tied to an embedding model/dimension; change it if you change either.
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "documents_gemini")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./metadata.db")
    MAX_DOCUMENTS: int = int(os.getenv("MAX_DOCUMENTS", "20"))

    # --- Chunking (measured in tokens, not characters) ---
    CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "256"))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))
    TOKENIZER_ENCODING: str = os.getenv("TOKENIZER_ENCODING", "cl100k_base")

    # --- Retrieval ---
    # Final number of chunks sent to the LLM.
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    # Candidates pulled from EACH retriever (vector / BM25) before fusion.
    RETRIEVAL_CANDIDATES: int = int(os.getenv("RETRIEVAL_CANDIDATES", "30"))
    # Hybrid = vector search + BM25 keyword search fused with RRF.
    HYBRID_SEARCH_ENABLED: bool = _env_bool("HYBRID_SEARCH_ENABLED", True)
    # Constant in Reciprocal Rank Fusion: score = 1 / (RRF_K + rank).
    RRF_K: int = int(os.getenv("RRF_K", "60"))

    # --- Reranking (cross-encoder, runs locally via FlashRank/ONNX) ---
    RERANKER_ENABLED: bool = _env_bool("RERANKER_ENABLED", True)
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")
    RERANKER_CACHE_DIR: str = os.getenv("RERANKER_CACHE_DIR", "./.flashrank_cache")


settings = Settings()
