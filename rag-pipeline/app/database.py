"""Database setup for SQLAlchemy ORM and ChromaDB client."""

from datetime import datetime, timezone

import chromadb
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

# ---------------------------------------------------------------------------
# SQLAlchemy setup
# ---------------------------------------------------------------------------

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Document(Base):
    """Metadata record for an uploaded document."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # pdf / docx / txt
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    file_size_kb = Column(Float, default=0.0)
    upload_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    status = Column(String, default="processing")  # processing / ready / failed


def init_db():
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# ChromaDB setup (embedded / persistent mode)
# ---------------------------------------------------------------------------

chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)


def get_chroma_collection():
    """Return (or create) the main documents collection in ChromaDB."""
    return chroma_client.get_or_create_collection(name="documents")
