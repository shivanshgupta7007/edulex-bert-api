"""EduLex BERT API - serve complex-word detection and text simplification
as a production-style HTTP service.

POST /detect-complex-words  {text} -> flagged words with spans and reasons
POST /simplify              {text} -> simplified text + replacement list
GET  /model-info            which backend is live (bert or heuristic fallback)
GET  /health                liveness
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .model import load_detector
from .schemas import (DetectRequest, DetectResponse, DetectionOut,
                      SimplifyRequest, SimplifyResponse, Suggestion)

# Small starter map for the fallback simplifier. The EduLex pipeline uses
# WordNet + BERT-ranked candidates; the API ships this honest minimal map so
# /simplify works end to end even without the trained model.
SIMPLE_SYNONYMS = {
    "approximately": "about",
    "commence": "start",
    "demonstrate": "show",
    "difficult": "hard",
    "endeavour": "try",
    "furthermore": "also",
    "however": "but",
    "numerous": "many",
    "obtain": "get",
    "purchase": "buy",
    "require": "need",
    "sufficient": "enough",
    "therefore": "so",
    "utilize": "use",
}

app = FastAPI(title="EduLex BERT API", version="1.0.0")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = load_detector()
    return _detector


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    detector = get_detector()
    note = (
        "Serving the fine-tuned BERT checkpoint from MODEL_DIR."
        if detector.name == "bert-token-classifier"
        else "Heuristic fallback active - place the fine-tuned checkpoint in MODEL_DIR to serve BERT."
    )
    return {"backend": detector.name, "note": note}


@app.post("/detect-complex-words", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    detector = get_detector()
    detections = [
        DetectionOut(word=d.word, start=d.start, end=d.end, reason=d.reason, backend=d.backend)
        for d in detector.detect(req.text)
    ]
    return DetectResponse(detections=detections, backend=detector.name)


@app.post("/simplify", response_model=SimplifyResponse)
def simplify(req: SimplifyRequest) -> SimplifyResponse:
    detector = get_detector()
    detections = detector.detect(req.text)
    replacements: list[Suggestion] = []
    simplified = req.text
    for d in sorted(detections, key=lambda d: d.start, reverse=True):
        suggestion = SIMPLE_SYNONYMS.get(d.word.lower())
        if suggestion:
            simplified = simplified[:d.start] + suggestion + simplified[d.end:]
            replacements.append(Suggestion(word=d.word, suggestion=suggestion))
    replacements.reverse()
    return SimplifyResponse(
        simplified_text=simplified, replacements=replacements, backend=detector.name
    )
