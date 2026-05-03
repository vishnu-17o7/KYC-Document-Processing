"""Base extractor interface."""

from abc import ABC, abstractmethod
from models import FieldConfidence


class BaseExtractor(ABC):
    document_type: str = ""

    @abstractmethod
    def extract(self, text: str) -> dict:
        ...

    def get_field_confidences(self, text: str, fields: dict) -> list[FieldConfidence]:
        confidences = []
        for key, value in fields.items():
            if value:
                conf = 0.70
                if str(value).lower() in text.lower():
                    conf = 0.90
                confidences.append(FieldConfidence(field=key, confidence=conf, raw_text=str(value)))
            else:
                confidences.append(FieldConfidence(field=key, confidence=0.0))
        return confidences
