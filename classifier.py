"""Document type classifier using keyword + layout heuristics."""

from typing import Optional
import re

KEYWORD_MAP = {
    "aadhaar": [
        "aadhaar", "uidai", "unique identification", "government of india",
        "aadhaar number", "enrolment no", "enrolment number", "www.uidai.gov.in",
    ],
    "pan": [
        "permanent account number", "income tax department", "pan card",
        "incometaxindia", "permanent account no", "न्ायकर", "आयकर",
    ],
    "voter_id": [
        "election commission", "epic", "voter identity card", "voter id",
        "electoral registration", "elector's photo identity",
    ],
    "driving_licence": [
        "driving licence", "driving license", "motor vehicles department",
        "dl no", "licence no", "validity", "vehicle class",
    ],
    "passport": [
        "passport", "republic of india", "passport number",
        "nationality", "date of expiry", "type p",
    ],
}

MRZ_PATTERN = re.compile(r"[A-Z0-9<]{44}")

LAYOUT_SIGNALS = {
    "aadhaar": {"aspect_ratio_range": (0.62, 0.8), "min_text_density": 0.08},
    "pan": {"aspect_ratio_range": (0.55, 0.72), "min_text_density": 0.05},
    "voter_id": {"aspect_ratio_range": (0.57, 0.78), "min_text_density": 0.08},
    "driving_licence": {"aspect_ratio_range": (0.50, 0.85), "min_text_density": 0.06},
    "passport": {"aspect_ratio_range": (0.66, 0.85), "min_text_density": 0.10},
}


def classify_document(text: str, image_shape: Optional[tuple] = None) -> tuple[Optional[str], float]:
    """
    Classify a document based on OCR'd text and optional image shape.
    Returns (document_type, confidence).
    document_type is None if no match is found.
    """
    if not text:
        return None, 0.0

    text_lower = text.lower()
    scores: dict[str, int] = {}

    for doc_type, keywords in KEYWORD_MAP.items():
        score = 0
        for kw in keywords:
            if kw in text_lower:
                score += 1
        scores[doc_type] = score

    if image_shape:
        h, w = image_shape[:2]
        aspect = w / h if h > 0 else 0
        for doc_type, signals in LAYOUT_SIGNALS.items():
            lo, hi = signals["aspect_ratio_range"]
            if lo <= aspect <= hi:
                scores[doc_type] = scores.get(doc_type, 0) + 1

    if MRZ_PATTERN.search(text):
        scores["passport"] = scores.get("passport", 0) + 3

    if not text:
        return None, 0.0

    if not scores:
        return None, 0.0

    max_type = max(scores, key=scores.get)  # type: ignore[arg-type]
    max_score = scores[max_type]

    total_signal = max_score
    if total_signal == 0:
        return None, 0.0

    if total_signal >= 3:
        confidence = 0.95
    elif total_signal == 2:
        confidence = 0.75
    elif total_signal == 1:
        confidence = 0.45
    else:
        confidence = 0.25

    if confidence < 0.40:
        return None, confidence

    return max_type, confidence
