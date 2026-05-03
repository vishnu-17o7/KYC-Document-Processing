import re
from extractors.base import BaseExtractor
from models import FieldConfidence


class PassportExtractor(BaseExtractor):
    document_type = "passport"

    MRZ_LINE1 = re.compile(r"P<[A-Z]{3}([A-Z]+)<<([A-Z]+)")
    MRZ_LINE2 = re.compile(r"([A-Z0-9<]{44})")

    NAME_PATTERN = re.compile(r"(?:Surname|Given\s*Name|Name)\s*[:\-]\s*(.+)", re.IGNORECASE)
    DOB_PATTERN = re.compile(r"(?:Date\s*of\s*Birth|DOB)\s*[:\-]\s*(\d{2}[/\-]\d{2}[/\-]\d{4})", re.IGNORECASE)
    PASSPORT_NO_PATTERN = re.compile(r"(?:Passport\s*No|Passport\s*Number)\s*[:\-]\s*([A-Z]\d{7})", re.IGNORECASE)
    NATIONALITY_PATTERN = re.compile(r"(?:Nationality|राष्ट्रीयता)\s*[:\-]\s*(.+)", re.IGNORECASE)
    EXPIRY_PATTERN = re.compile(r"(?:Date\s*of\s*Expiry|Expiry)\s*[:\-]\s*(\d{2}[/\-]\d{2}[/\-]\d{4})", re.IGNORECASE)

    def _normalize_date(self, date_str: str) -> str:
        date_str = date_str.replace("-", "/")
        parts = date_str.split("/")
        if len(parts) == 3:
            return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
        return date_str

    def _parse_mrz(self, text: str) -> dict:
        """Parse MRZ lines to extract name, passport number, DOB, nationality."""
        result = {}

        lines = text.split("\n")
        mrz_lines = []
        for line in lines:
            line = line.strip()
            if re.match(r"^[A-Z0-9<]{30,50}$", line) and "<<" in line:
                mrz_lines.append(line)

        if len(mrz_lines) >= 2:
            line1 = mrz_lines[0]
            line2 = mrz_lines[1]

            result["mrz_line1"] = line1
            result["mrz_line2"] = line2

            mrz1_match = self.MRZ_LINE1.search(line1)
            if mrz1_match:
                surname = mrz1_match.group(1).replace("<", " ").strip()
                given_name = mrz1_match.group(2).replace("<", " ").strip()
                result["name"] = f"{given_name} {surname}".strip()

            if len(line2) >= 44:
                result["passport_number"] = line2[0:9].replace("<", "")

                dob_raw = line2[13:19]
                if dob_raw.isdigit():
                    y = dob_raw[0:2]
                    m = dob_raw[2:4]
                    d = dob_raw[4:6]
                    year = int(y)
                    full_year = 1900 + year if year > 50 else 2000 + year
                    result["dob"] = f"{full_year}-{m}-{d}"

                result["nationality"] = line1[2:5]

                expiry_raw = line2[21:27]
                if expiry_raw.isdigit():
                    y = expiry_raw[0:2]
                    m = expiry_raw[2:4]
                    d = expiry_raw[4:6]
                    year = int(y)
                    full_year = 1900 + year if year > 50 else 2000 + year
                    result["expiry_date"] = f"{full_year}-{m}-{d}"

        return result

    def extract(self, text: str) -> dict:
        fields = {}

        mrz_fields = self._parse_mrz(text)
        fields.update(mrz_fields)

        for pattern, key, label in [
            (self.NAME_PATTERN, "name", "name"),
            (self.DOB_PATTERN, "dob", "dob"),
            (self.PASSPORT_NO_PATTERN, "passport_number", "passport_number"),
            (self.NATIONALITY_PATTERN, "nationality", "nationality"),
            (self.EXPIRY_PATTERN, "expiry_date", "expiry_date"),
        ]:
            if key in fields and fields[key]:
                continue
            m = pattern.search(text)
            if m:
                val = m.group(1).strip().rstrip(".,;")
                if "date" in key.lower() or key in ("dob", "expiry_date"):
                    val = self._normalize_date(val)
                fields[key] = val

        if "name" not in fields or not fields.get("name"):
            for line in text.split("\n"):
                line = line.strip().upper()
                if line in ("PASSPORT", "PASSPORT NUMBER", "REPUBLIC OF INDIA", ""):
                    continue
                if re.match(r"^[A-Z][A-Z\s]+$", line) and len(line) > 5:
                    fields["name"] = line.title()
                    break

        return fields

    def get_field_confidences(self, text: str, fields: dict) -> list[FieldConfidence]:
        confidences = []
        mrz_keys = {"mrz_line1", "mrz_line2"}
        for key, value in fields.items():
            raw = str(value) if value else ""
            if value:
                conf = 0.95 if key in mrz_keys else 0.85
                confidences.append(FieldConfidence(field=key, confidence=conf, raw_text=raw))
            else:
                confidences.append(FieldConfidence(field=key, confidence=0.0))
        return confidences
