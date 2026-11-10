from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to scan for complex words")


class DetectionOut(BaseModel):
    word: str
    start: int
    end: int
    reason: str
    backend: str


class DetectResponse(BaseModel):
    detections: list[DetectionOut]
    backend: str


class SimplifyRequest(BaseModel):
    text: str = Field(..., min_length=1)


class Suggestion(BaseModel):
    word: str
    suggestion: str


class SimplifyResponse(BaseModel):
    simplified_text: str
    replacements: list[Suggestion]
    backend: str
