"""Aadhaar masking enforcement layer."""

import re
import logging

logger = logging.getLogger(__name__)

AADHAAR_CLEAN_PATTERN = re.compile(r"(\d{4})\s?(\d{4})\s?(\d{4})")


def mask_aadhaar(value: str) -> str:
    """Mask first 8 digits of an Aadhaar number. Returns 'XXXX XXXX 3456' format."""
    m = AADHAAR_CLEAN_PATTERN.search(value.replace(" ", ""))
    if m:
        digits = value.replace(" ", "")
        if len(digits) == 12 and digits.isdigit():
            return f"XXXX XXXX {digits[8:]}"

    cleaned = value.replace(" ", "")
    if len(cleaned) == 12 and cleaned.isdigit():
        return f"XXXX XXXX {cleaned[8:]}"

    return value


def mask_fields(fields: dict, document_type: str) -> dict:
    """Apply masking to extracted fields. Only Aadhaar numbers are masked."""
    if document_type != "aadhaar":
        return fields

    masked = dict(fields)
    if "aadhaar_number" in masked and masked["aadhaar_number"]:
        original = masked["aadhaar_number"]
        masked["aadhaar_number"] = mask_aadhaar(original)
        logger.info("Aadhaar number masked in output")
    return masked


def safe_log(text: str) -> str:
    """Scrub any unmasked Aadhaar numbers from log messages."""
    return AADHAAR_CLEAN_PATTERN.sub(
        lambda m: f"XXXX XXXX {m.group(3)}", text
    )
