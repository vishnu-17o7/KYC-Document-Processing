import re
from extractors.base import BaseExtractor
from models import FieldConfidence


class VoterIdExtractor(BaseExtractor):
    document_type = "voter_id"

    VOTER_ID_PATTERN = re.compile(r"\b([A-Z]{3}\d{7})\b")
    NAME_PATTERN = re.compile(r"(?:Name|नाम)\s*[:\-]\s*(.+)", re.IGNORECASE)
    RELATIVE_PATTERNS = [
        re.compile(r"(?:Father(?:'s)?|Husband(?:'s)?)\s*Name\s*[:\-]\s*(.+)", re.IGNORECASE),
        re.compile(r"(?:पिता|पति)\s*का\s*नाम\s*[:\-]?\s*(.+)", re.IGNORECASE),
    ]
    DOB_PATTERN = re.compile(r"(?:DOB|Date\s*of\s*Birth)\s*[:\-]\s*(\d{2}[/\-]\d{2}[/\-]\d{4})", re.IGNORECASE)
    CONSTITUENCY_PATTERN = re.compile(r"(?:Constituency|Assembly)\s*[:\-]\s*(.+)", re.IGNORECASE)

    def _normalize_date(self, date_str: str) -> str:
        date_str = date_str.replace("-", "/")
        parts = date_str.split("/")
        if len(parts) == 3:
            return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
        return date_str

    def extract(self, text: str) -> dict:
        fields = {}

        m = self.VOTER_ID_PATTERN.search(text)
        if m:
            fields["voter_id_number"] = m.group(1)

        m = self.NAME_PATTERN.search(text)
        if m:
            fields["name"] = m.group(1).strip().rstrip(".,;")

        for pattern in self.RELATIVE_PATTERNS:
            m = pattern.search(text)
            if m:
                fields["relative_name"] = m.group(1).strip().rstrip(".,;")
                break

        m = self.DOB_PATTERN.search(text)
        if m:
            fields["dob"] = self._normalize_date(m.group(1))

        m = self.CONSTITUENCY_PATTERN.search(text)
        if m:
            fields["constituency"] = m.group(1).strip()

        lines = text.split("\n")
        address_lines = []
        capture = False
        for line in lines:
            low = line.lower()
            if "address" in low:
                capture = True
                continue
            if capture:
                if any(kw in low for kw in ["constituency", "date", "election commission"]):
                    capture = False
                elif line.strip():
                    address_lines.append(line.strip())
        if address_lines:
            fields["address"] = ", ".join(address_lines)

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
