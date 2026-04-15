"""Tests for document management endpoints."""

import io
from unittest.mock import patch, MagicMock

from app.database import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_txt_file(content: str = "Hello world. This is a test document."):
    """Return a file-like tuple suitable for TestClient uploads."""
    return ("file", ("test.txt", io.BytesIO(content.encode()), "text/plain"))


def _create_test_pdf_file():
    """Return a minimal fake PDF file for upload testing."""
    # We provide a real tiny PDF header so the upload endpoint accepts it.
    # The actual parsing is mocked.
    pdf_bytes = b"%PDF-1.4 fake content"
    return ("file", ("report.pdf", io.BytesIO(pdf_bytes), "application/pdf"))


def _mock_process_document(*args, **kwargs):
    """Mock that returns (chunk_count=5, page_count=3) without calling OpenAI."""
    return (5, 3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("app.routers.documents.process_document", side_effect=_mock_process_document)
def test_upload_pdf(mock_proc, client):
    """Upload a PDF file and verify the response."""
    response = client.post("/documents/upload", files=[_create_test_pdf_file()])
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "report.pdf"
    assert data["status"] == "ready"
    assert data["chunk_count"] == 5
    assert data["page_count"] == 3


def test_upload_unsupported_type(client):
    """Uploading an unsupported file type should return 400."""
    file = ("file", ("image.png", io.BytesIO(b"fake png"), "image/png"))
    response = client.post("/documents/upload", files=[file])
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"]


@patch("app.routers.documents.process_document", side_effect=_mock_process_document)
def test_list_documents(mock_proc, client):
    """Upload a document then list — it should appear."""
    client.post("/documents/upload", files=[_create_test_txt_file()])
    response = client.get("/documents/")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert any(d["filename"] == "test.txt" for d in data["documents"])


@patch("app.routers.documents.process_document", side_effect=_mock_process_document)
def test_get_document(mock_proc, client):
    """Upload a document then get it by ID."""
    upload_resp = client.post("/documents/upload", files=[_create_test_txt_file()])
    doc_id = upload_resp.json()["id"]

    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["filename"] == "test.txt"


def test_get_document_not_found(client):
    """Getting a non-existent document should return 404."""
    response = client.get("/documents/9999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@patch("app.routers.documents.process_document", side_effect=_mock_process_document)
def test_delete_document(mock_proc, client):
    """Upload then delete a document — subsequent GET should 404."""
    upload_resp = client.post("/documents/upload", files=[_create_test_txt_file()])
    doc_id = upload_resp.json()["id"]

    delete_resp = client.delete(f"/documents/{doc_id}")
    assert delete_resp.status_code == 200
    assert "successfully" in delete_resp.json()["message"]

    # Verify it's gone
    get_resp = client.get(f"/documents/{doc_id}")
    assert get_resp.status_code == 404


@patch("app.routers.documents.process_document", side_effect=_mock_process_document)
def test_max_document_limit(mock_proc, client, db_session):
    """When MAX_DOCUMENTS is reached, new uploads should be rejected."""
    from app.config import settings
    from datetime import datetime, timezone

    # Insert MAX_DOCUMENTS records directly into the test DB
    for i in range(settings.MAX_DOCUMENTS):
        doc = Document(
            filename=f"doc_{i}.txt",
            original_name=f"doc_{i}.txt",
            file_type="txt",
            status="ready",
            upload_time=datetime.now(timezone.utc),
        )
        db_session.add(doc)
    db_session.commit()

    # Next upload should be rejected
    response = client.post("/documents/upload", files=[_create_test_txt_file()])
    assert response.status_code == 400
    assert "Maximum document limit" in response.json()["detail"]
