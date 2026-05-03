import re
from extractors.base import BaseExtractor
from models import FieldConfidence


class PanExtractor(BaseExtractor):
    document_type = "pan"

    PAN_PATTERN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
    NAME_PATTERNS = [
        re.compile(r"Name\s*[:\-]\s*(.+)", re.IGNORECASE),
        re.compile(r"(?:Name|नाम)\s*[:\-]?\s*([A-Z][A-Za-z\s]+)", re.MULTILINE),
    ]
    FATHER_PATTERNS = [
        re.compile(r"(?:Father(?:'s)?\s*Name|Father)\s*[:\-]\s*(.+)", re.IGNORECASE),
        re.compile(r"(?:पिता\s*का\s*नाम)\s*[:\-]?\s*(.+)", re.IGNORECASE),
    ]
    DOB_PATTERNS = [
        re.compile(r"(?:DOB|Date\s*of\s*Birth|जन्म\s*तारीख)\s*[:\-]\s*(\d{2}[/\-]\d{2}[/\-]\d{4})", re.IGNORECASE),
    ]

    def _normalize_date(self, date_str: str) -> str:
        date_str = date_str.replace("-", "/")
        parts = date_str.split("/")
        if len(parts) == 3:
            return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
        return date_str

    def extract(self, text: str) -> dict:
        fields = {}

        pan_match = self.PAN_PATTERN.search(text)
        if pan_match:
            fields["pan_number"] = pan_match.group(1)

        for pattern in self.NAME_PATTERNS:
            m = pattern.search(text)
            if m:
                name = m.group(1).strip().rstrip(".,;")
                if len(name) > 2:
                    fields["name"] = name
                    break

        for pattern in self.FATHER_PATTERNS:
            m = pattern.search(text)
            if m:
                fname = m.group(1).strip().rstrip(".,;")
                if len(fname) > 2:
                    fields["fathers_name"] = fname
                    break

        for pattern in self.DOB_PATTERNS:
            m = pattern.search(text)
            if m:
                fields["dob"] = self._normalize_date(m.group(1))
                break

        return fields

    def get_field_confidences(self, text: str, fields: dict) -> list[FieldConfidence]:
        confidences = []
        for key, value in fields.items():
            raw = str(value) if value else ""
            if value:
                conf = 0.85 if raw in text else 0.75
                confidences.append(FieldConfidence(field=key, confidence=conf, raw_text=raw))
            else:
                confidences.append(FieldConfidence(field=key, confidence=0.0))
        return confidences
