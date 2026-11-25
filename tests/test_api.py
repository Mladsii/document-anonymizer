import io
import zipfile
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("NER_ENABLED", "0")
    with TestClient(app) as client:
        yield client


def test_multipart_any_field_and_legacy_route(client):
    for route in ("/api/anonymize", "/webhook/anonymize-upload"):
        result = client.post(route, files={"data": ("test.txt", "Почта me@example.org".encode(), "text/plain")})
        assert result.status_code == 200
        assert "example.org" not in result.text
        assert "attachment" in result.headers["content-disposition"]
        assert result.headers["cache-control"] == "no-store"


def test_raw_and_json(client):
    assert client.post("/api/anonymize", content=b"me@example.org", headers={"Content-Type": "text/plain"}).text == "[EMAIL_1]"
    assert client.post("/api/anonymize", json={"text": "me@example.org"}).text == "[EMAIL_1]"
    assert client.post("/api/anonymize", json=[]).status_code == 400


def test_docx_hyperlink_and_table_text(client):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:hyperlink><w:r><w:t>me@example.org</w:t></w:r></w:hyperlink></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>example.org</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>')
    response = client.post("/api/anonymize", files={"file": ("test.docx", content.getvalue())})
    assert response.status_code == 200
    assert response.text == "[EMAIL_1]\n[ССЫЛКА_1]"


def test_text_pdf_and_scanned_pdf_rejection(client):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 10 100 Td (me@example.org) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    result = client.post("/api/anonymize", files={"file": ("test.pdf", output.getvalue())})
    assert result.status_code == 200
    assert result.text.strip() == "[EMAIL_1]"
    writer.add_blank_page(width=300, height=200)
    partial = io.BytesIO()
    writer.write(partial)
    assert client.post("/api/anonymize", files={"file": ("partial.pdf", partial.getvalue())}).status_code == 400


def test_errors_and_busy(client):
    assert client.post("/api/anonymize", files={"file": ("file.doc", b"fake")}).status_code == 400
    assert client.post("/api/anonymize", files={"file": ("file.txt", b"")}).status_code == 400
    assert client.post("/api/anonymize", files={"file": ("file.pdf", b"broken")}).status_code == 400
    assert client.post("/api/anonymize", json={"text": "a" * 300001}).status_code == 400
    assert client.get("/healthz").json() == {"status": "ok", "ner": False}


def test_duplicate_files_rejected(client):
    response = client.post("/api/anonymize", files=[("file", ("a.txt", b"one")), ("file", ("b.txt", b"two"))])
    assert response.status_code == 400


def test_size_limit(client, monkeypatch):
    monkeypatch.setattr("app.main.MAX_BYTES", 32)
    response = client.post("/api/anonymize", files={"file": ("big.txt", b"a" * 33)})
    assert response.status_code == 400


def test_busy_does_not_queue_documents(client):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    started, release = threading.Event(), threading.Event()
    original = app.state.engine.anonymize

    def slow(text):
        started.set()
        assert release.wait(5)
        return original(text)

    app.state.engine.anonymize = slow
    with ThreadPoolExecutor() as executor:
        running = executor.submit(client.post, "/api/anonymize", json={"text": "me@example.org"})
        try:
            assert started.wait(5)
            rejected = client.post("/api/anonymize", json={"text": "another@example.org"})
            assert rejected.status_code == 503
            assert rejected.headers["retry-after"] == "5"
        finally:
            release.set()
        assert running.result().status_code == 200
