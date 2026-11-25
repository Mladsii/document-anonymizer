from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app


def test_real_model_loads_and_processes_without_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Outbound network must not be used")

    monkeypatch.setattr("socket.socket.connect", denied)
    monkeypatch.setattr("socket.create_connection", denied)
    monkeypatch.setenv("NER_ENABLED", "1")
    source = Path(__file__).resolve().parents[1] / "examples/test_anonimizator.txt"
    with TestClient(app) as client:
        assert client.get("/healthz").json()["ner"] is True
        response = client.post("/api/anonymize", files={"file": (source.name, source.read_bytes())})
        assert response.status_code == 200
        for secret in ("ООО «Пример»", "Иван Петров", "example.org", "192.0.2.15", "567890", "1234567890"):
            assert secret not in response.text
        assert "00:10 Спикер B:" in response.text
        assert "бюджет 5000 рублей" in response.text
