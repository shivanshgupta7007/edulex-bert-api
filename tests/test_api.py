from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info_reports_backend_honestly():
    info = client.get("/model-info").json()
    assert info["backend"] in {"heuristic-fallback", "bert-token-classifier"}
    assert "note" in info


def test_detect_flags_long_words_with_spans():
    resp = client.post("/detect-complex-words",
                       json={"text": "Reading is fundamental and enjoyable."})
    data = resp.json()
    words = [d["word"] for d in data["detections"]]
    assert "fundamental" in words
    assert "Reading" not in words
    for d in data["detections"]:
        assert data["detections"] and d["end"] > d["start"]


def test_detect_empty_text_rejected():
    assert client.post("/detect-complex-words", json={"text": ""}).status_code == 422


def test_simplify_swaps_known_synonyms():
    resp = client.post("/simplify",
                       json={"text": "We require approximately sufficient funds."})
    data = resp.json()
    assert "about" in data["simplified_text"]
    assert "enough" in data["simplified_text"]
    swapped = {r["word"].lower(): r["suggestion"] for r in data["replacements"]}
    assert swapped.get("approximately") == "about"
    assert swapped.get("sufficient") == "enough"


def test_syllable_estimator():
    from app.model import estimate_syllables
    assert estimate_syllables("cat") == 1
    assert estimate_syllables("fundamental") >= 3
