import re
from extractors.base import BaseExtractor
from models import FieldConfidence


class DLExtractor(BaseExtractor):
    document_type = "driving_licence"

    DL_PATTERN = re.compile(r"(?:DL\s*No|Licence\s*No|License\s*No)\s*[:\-]?\s*([A-Z0-9\s\-/]+)", re.IGNORECASE)
    NAME_PATTERN = re.compile(r"(?:Name|नाम)\s*[:\-]\s*(.+)", re.IGNORECASE)
    DOB_PATTERN = re.compile(r"(?:DOB|Date\s*of\s*Birth)\s*[:\-]\s*(\d{2}[/\-]\d{2}[/\-]\d{4})", re.IGNORECASE)
    VALID_FROM_PATTERN = re.compile(r"(?:Valid\s*From|Issued|Issue\s*Date)\s*[:\-]\s*(\d{2}[/\-]\d{2}[/\-]\d{4})", re.IGNORECASE)
    VALID_TO_PATTERN = re.compile(r"(?:Valid\s*(?:To|Until)|Expiry)\s*[:\-]\s*(\d{2}[/\-]\d{2}[/\-]\d{4})", re.IGNORECASE)
    VEHICLE_CLASS_PATTERN = re.compile(r"(?:Vehicle\s*Class|Class\s*of\s*Vehicle|COV)\s*[:\-]\s*(.+)", re.IGNORECASE)

    def _normalize_date(self, date_str: str) -> str:
        date_str = date_str.replace("-", "/")
        parts = date_str.split("/")
        if len(parts) == 3:
            return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
        return date_str

    def extract(self, text: str) -> dict:
        fields = {}

        m = self.DL_PATTERN.search(text)
        if m:
            fields["dl_number"] = m.group(1).strip()

        m = self.NAME_PATTERN.search(text)
        if m:
            fields["name"] = m.group(1).strip().rstrip(".,;")

        m = self.DOB_PATTERN.search(text)
        if m:
            fields["dob"] = self._normalize_date(m.group(1))

        m = self.VALID_FROM_PATTERN.search(text)
        if m:
            fields["valid_from"] = self._normalize_date(m.group(1))

        m = self.VALID_TO_PATTERN.search(text)
        if m:
            fields["valid_to"] = self._normalize_date(m.group(1))

        m = self.VEHICLE_CLASS_PATTERN.search(text)
        if m:
            fields["vehicle_classes"] = m.group(1).strip()

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
