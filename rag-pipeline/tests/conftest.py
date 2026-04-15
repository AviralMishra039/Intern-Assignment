"""Shared test fixtures — isolated DB, ChromaDB client, and FastAPI test client."""

import os
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db, get_chroma_collection
from app.main import app

import chromadb

# ---------------------------------------------------------------------------
# In-memory SQLite for test isolation
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite:///./test_metadata.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Temporary ChromaDB for test isolation
# ---------------------------------------------------------------------------

_test_chroma_dir = tempfile.mkdtemp()
_test_chroma_client = chromadb.PersistentClient(path=_test_chroma_dir)


def override_get_chroma_collection():
    return _test_chroma_client.get_or_create_collection(name="test_documents")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all tables at the start of the test session and drop them at the end."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    # Dispose engine to release file handles before deletion (required on Windows)
    test_engine.dispose()
    # Clean up test DB file
    if os.path.exists("./test_metadata.db"):
        try:
            os.remove("./test_metadata.db")
        except PermissionError:
            pass  # Best-effort on Windows
    # Clean up test chroma dir
    if os.path.exists(_test_chroma_dir):
        shutil.rmtree(_test_chroma_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_tables():
    """Truncate all tables between tests for isolation."""
    db = TestSessionLocal()
    for table in reversed(Base.metadata.sorted_tables):
        db.execute(table.delete())
    db.commit()
    db.close()

    # Also clear test chroma collection
    try:
        _test_chroma_client.delete_collection("test_documents")
    except Exception:
        pass
    _test_chroma_client.get_or_create_collection(name="test_documents")

    yield


@pytest.fixture()
def client():
    """Return a FastAPI TestClient with overridden dependencies."""
    app.dependency_overrides[get_db] = override_get_db
    # Monkey-patch the chroma collection getter used by services
    import app.database as db_module
    original_fn = db_module.get_chroma_collection
    db_module.get_chroma_collection = override_get_chroma_collection

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    db_module.get_chroma_collection = original_fn


@pytest.fixture()
def db_session():
    """Provide a raw test DB session."""
    db = TestSessionLocal()
    yield db
    db.close()
