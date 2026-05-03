import re
from extractors.base import BaseExtractor
from models import FieldConfidence


class AadhaarExtractor(BaseExtractor):
    document_type = "aadhaar"

    NAME_PATTERNS = [
        re.compile(r"Name\s*[:\-]\s*(.+)", re.IGNORECASE),
        re.compile(r"^(?:नाम|Name)\s*[:\-]?\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    ]
    DOB_PATTERNS = [
        re.compile(r"DOB\s*[:\-]\s*(\d{2}[/\-]\d{2}[/\-]\d{4})", re.IGNORECASE),
        re.compile(r"Date\s*of\s*Birth\s*[:\-]\s*(\d{2}[/\-]\d{2}[/\-]\d{4})", re.IGNORECASE),
        re.compile(r"जन्म\s*तारीख\s*[:\-]?\s*(\d{2}[/\-]\d{2}[/\-]\d{4})", re.IGNORECASE),
    ]
    GENDER_PATTERNS = [
        re.compile(r"(?:Gender|लिंग)\s*[:\-]\s*(Male|Female|Transgender|MALE|FEMALE)", re.IGNORECASE),
        re.compile(r"\b(Male|Female|MALE|FEMALE)\b"),
    ]
    AADHAAR_PATTERN = re.compile(r"(\d{4}\s?\d{4}\s?\d{4})")
    ADDRESS_KEYWORDS = ["address", "पता", "address:", "s/o", "d/o", "w/o", "c/o"]

    def _normalize_date(self, date_str: str) -> str:
        date_str = date_str.replace("-", "/")
        parts = date_str.split("/")
        if len(parts) == 3:
            return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
        return date_str

    def extract(self, text: str) -> dict:
        fields = {}

        for pattern in self.NAME_PATTERNS:
            m = pattern.search(text)
            if m:
                fields["name"] = m.group(1).strip().rstrip(".,;")
                break

        for pattern in self.DOB_PATTERNS:
            m = pattern.search(text)
            if m:
                fields["dob"] = self._normalize_date(m.group(1))
                break

        for pattern in self.GENDER_PATTERNS:
            m = pattern.search(text)
            if m:
                fields["gender"] = m.group(1).strip().title()
                break

        aadhaar_matches = self.AADHAAR_PATTERN.findall(text)
        for match in aadhaar_matches:
            cleaned = match.replace(" ", "")
            if len(cleaned) == 12 and cleaned.isdigit():
                fields["aadhaar_number"] = f"{cleaned[:4]} {cleaned[4:8]} {cleaned[8:]}"
                break

        lines = text.split("\n")
        address_lines = []
        capture = False
        for line in lines:
            low = line.lower().strip()
            if any(kw in low for kw in self.ADDRESS_KEYWORDS):
                capture = True
                address_lines.append(line.strip())
            elif capture:
                if any(kw in low for kw in ["date", "mobile", "phone", "email", "aadhaar number"]):
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
                conf = 0.80
                if raw.lower() in text.lower():
                    conf = 0.92
                confidences.append(FieldConfidence(field=key, confidence=conf, raw_text=raw))
            else:
                confidences.append(FieldConfidence(field=key, confidence=0.0))
        return confidences
