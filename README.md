# EduLex BERT API

**Live API:** https://edulex-bert-api.onrender.com - try `GET /model-info`, `POST /detect-complex-words` with `{"text": "..."}`.

The [EduLex](https://github.com/shivanshgupta7007) dyslexia-aware learning assistant's fine-tuned BERT token classifier - served as a production-style HTTP API with Docker and CI. Flags phonologically complex and long-form words in text, and simplifies them, so reading material can adapt to the learner.

## What it demonstrates

- **Model serving**: a trained transformer checkpoint behind FastAPI, with request/response schemas, validation, and a `/model-info` endpoint that reports what is actually being served.
- **Honest fallback**: the API always runs - without the trained weights it serves a clearly-labelled heuristic (word length + syllable estimate), never passed off as the BERT model. Drop the checkpoint into `model/` and the same endpoints serve BERT.
- **MLOps basics**: Dockerfile, dockerignore, and a GitHub Actions CI workflow running the test suite on every push.

## Stack

Python · FastAPI · PyTorch / Hugging Face Transformers (BERT) · Docker · GitHub Actions · pytest

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/detect-complex-words` | Flag complex words with character spans and reasons |
| POST | `/simplify` | Return simplified text with a replacement list |
| GET | `/model-info` | Which backend is live (`bert-token-classifier` or `heuristic-fallback`) |
| GET | `/health` | Liveness |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Run with Docker

```bash
docker build -t edulex-api .
docker run -p 8000:8000 edulex-api
```

## Serve the trained model

See `scripts/ADD_MODEL_HERE.md`: copy the fine-tuned checkpoint into `model/`, install `torch transformers`, restart. `/model-info` confirms the switch.

## Tests

```bash
pytest   # runs fully against the heuristic fallback - no weights needed
```
