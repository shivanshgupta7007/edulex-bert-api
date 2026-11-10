# EduLex BERT API - How to Explain It (Study Guide)

## The 30-second version

"This takes the BERT token classifier I fine-tuned for EduLex - it detects complex words with 92% precision - and serves it as a real API: FastAPI, request validation, Docker, CI on every push. The point of the project is the MLOps story: a notebook model isn't a product until something can call it reliably. One honest design choice: if the trained weights aren't present, the API serves a clearly-labelled rule-based fallback instead of failing, and the /model-info endpoint always says which backend is live."

## End-to-end flow

1. Client POSTs text to `/detect-complex-words`.
2. FastAPI validates the request with a Pydantic schema (empty text is rejected with 422).
3. `load_detector()` serves the BERT checkpoint when `MODEL_DIR` has one, else the heuristic fallback.
4. Detections come back as `{word, start, end, reason, backend}` - character spans, so a UI can highlight in place.
5. `/simplify` runs detection, then replaces flagged words with simpler synonyms (a starter map in the fallback; WordNet + BERT-ranked candidates in the full EduLex pipeline).

## Key code to know cold

- `app/model.py` -> `load_detector()`: backend selection, and why the fallback exists.
- `app/model.py` -> `estimate_syllables()`: vowel-group counting minus silent trailing 'e'.
- `app/main.py` -> `simplify()`: replacements applied in reverse span order so earlier character offsets don't shift.

## Why these choices

- **Why a fallback at all?** Tests, CI and the Docker image must run without the trained weights (they're large and not in git). The fallback keeps every endpoint exercisable while being explicitly labelled - observability over pretense.
- **Why token classification for this?** Complex-word detection is a span-labeling problem: for each token, complex or not. BERT's contextual embeddings catch that "read" is simple in "read a book" but the model learns patterns like morphological complexity that rules miss.
- **Why FastAPI?** Async, automatic OpenAPI docs at /docs, Pydantic validation for free. It's the standard Python model-serving choice.
- **Why does /simplify edit spans in reverse order?** Replacements change string length; applying them right-to-left keeps the remaining offsets valid. Classic off-by-one trap.
- **Why is /model-info honest about the backend?** In production, silent degradation is the worst kind. If the model fails to load, you want the API to say so - loud and checkable - not to quietly serve something else.

## Likely interview questions

**Q: How was the BERT model trained?**
Fine-tuned a pretrained BERT token classifier on labeled words (complex vs simple). The pipeline: tokenize with the BERT tokenizer, add a classification head, train with a small learning rate (2e-5), evaluate with precision/recall per token. EduLex hit 92% precision on phonologically complex and long-form words.

**Q: Why is 92% *precision* the headline, not accuracy?**
In this application, false alarms are worse than misses - marking a simple word as complex degrades the reading experience. High precision means when it flags a word, it's usually right.

**Q: How does CI work here?**
GitHub Actions on every push: checkout, set up Python, install requirements, run pytest. The tests run entirely against the heuristic fallback, so CI needs no model weights and no GPU.

**Q: How would you deploy this for real traffic?**
The Docker image as-is behind a load balancer; cache model load at startup; batch requests; add request logging and latency metrics. For scale: an inference server (TorchServe/Triton) or a managed endpoint, with autoscaling.

**Q: How would you handle model updates?**
Version checkpoints, keep /model-info reporting the version, ship canary releases, and watch precision on a labeled holdout before promoting a new checkpoint.
