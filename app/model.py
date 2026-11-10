"""Complex-word detection for dyslexia-aware reading support.

Two backends, one interface:

1. BertTokenClassifier - the fine-tuned BERT token classifier from the
   EduLex project (92% precision on phonologically complex / long-form
   words). Loaded when MODEL_DIR points at a real checkpoint.
2. HeuristicDetector - a transparent rule-based fallback (word length +
   estimated syllables) so the API always runs, in tests and in Docker,
   without the trained weights. It is labelled as a fallback everywhere
   it surfaces; it is not passed off as the BERT model.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass
class Detection:
    word: str
    start: int
    end: int
    reason: str
    backend: str


_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.IGNORECASE)


def estimate_syllables(word: str) -> int:
    """Rough syllable count: vowel groups, minus a silent trailing 'e'."""
    groups = _VOWEL_GROUP.findall(word.lower())
    count = max(1, len(groups))
    if word.lower().endswith("e") and count > 1:
        count -= 1
    return count


class HeuristicDetector:
    name = "heuristic-fallback"

    def detect(self, text: str) -> list[Detection]:
        out = []
        for match in re.finditer(r"[A-Za-z']+", text):
            word = match.group(0)
            syllables = estimate_syllables(word)
            if len(word) >= 8 or syllables >= 3:
                reason = []
                if len(word) >= 8:
                    reason.append(f"length {len(word)} >= 8")
                if syllables >= 3:
                    reason.append(f"~{syllables} syllables")
                out.append(Detection(word, match.start(), match.end(),
                                     "; ".join(reason), backend=self.name))
        return out


class BertTokenClassifier:
    name = "bert-token-classifier"

    def __init__(self, model_dir: str):
        from transformers import pipeline  # lazy: heavy dependency

        self._nlp = pipeline("token-classification", model=model_dir,
                             aggregation_strategy="simple")

    def detect(self, text: str) -> list[Detection]:
        out = []
        for ent in self._nlp(text):
            if ent["entity_group"].upper() in {"COMPLEX", "B-COMPLEX", "I-COMPLEX", "LABEL_1"}:
                out.append(Detection(ent["word"], ent["start"], ent["end"],
                                     f"BERT score {ent['score']:.2f}", backend=self.name))
        return out


def load_detector():
    """Prefer the trained checkpoint; fall back to the heuristic honestly."""
    model_dir = os.environ.get("MODEL_DIR", "model/")
    if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
        try:
            return BertTokenClassifier(model_dir)
        except Exception:
            pass  # unreadable checkpoint: fall through to heuristic
    return HeuristicDetector()
