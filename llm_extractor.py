"""LLM-based extraction using OpenRouter — vision model + text model."""

import base64
import json
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_VISION_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_TEXT_MODEL = "qwen/qwen3-coder:free"

VISION_PROMPT = """You are a KYC document extraction system. Analyze this Indian identity document image carefully.

Identify the document type and extract ALL visible fields. Return ONLY valid JSON — no markdown, no commentary.

Document types: aadhaar, pan, voter_id, driving_licence, passport

Output format:
{
  "document_type": "aadhaar",
  "confidence": 0.95,
  "fields": {
    "name": "...",
    "dob": "...",
    ...
  },
  "extraction_warnings": []
}

Field names for each type:
- aadhaar: name, dob, gender, address, aadhaar_number
- pan: name, fathers_name, dob, pan_number
- voter_id: name, relative_name, dob, voter_id_number, address, constituency
- driving_licence: name, dob, dl_number, valid_from, valid_to, vehicle_classes, address
- passport: name, dob, passport_number, nationality, expiry_date

Standardize dates to YYYY-MM-DD. Standardize Aadhaar number as "XXXX XXXX NNNN" (include all 12 digits if visible — masking will be applied later).
Standardize PAN as 5 letters + 4 digits + 1 letter.

If the image is too blurry or unreadable, set confidence low and add "low_image_quality" to warnings.
If the document is not one of the five supported Indian types, set document_type to null and add "unsupported_document" to warnings."""

TEXT_PROMPT = """You are a KYC data extractor. Given OCR'd text and a document type, extract all fields as JSON.

Document type: {document_type}

OCR text:
\"\"\"
{ocr_text}
\"\"\"

Return ONLY valid JSON — no markdown, no commentary:
{
  "document_type": "{document_type}",
  "confidence": 0.85,
  "fields": {
    "name": "...",
    ...
  },
  "extraction_warnings": []
}

Field names:
- aadhaar: name, dob, gender, address, aadhaar_number
- pan: name, fathers_name, dob, pan_number
- voter_id: name, relative_name, dob, voter_id_number, address, constituency
- driving_licence: name, dob, dl_number, valid_from, valid_to, vehicle_classes, address
- passport: name, dob, passport_number, nationality, expiry_date

Standardize dates to YYYY-MM-DD. If a field is not found in the text, set it to null. Be precise."""


def _get_api_key() -> Optional[str]:
    return os.getenv("OPENROUTER_API_KEY", "")


def _parse_llm_response(content: str) -> dict:
    """Parse the LLM response, handling markdown code fences."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {"document_type": None, "confidence": 0.0, "fields": {}, "extraction_warnings": ["llm_parse_failed"]}


async def extract_with_vision(image_bytes: bytes, api_key: Optional[str] = None) -> dict:
    """Send a document image to a vision model and get structured fields back."""
    key = api_key or _get_api_key()
    if not key:
        raise ValueError("OpenRouter API key not configured")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = "image/jpeg" if image_bytes[:2] == b"\xff\xd8" else "image/png"
    data_url = f"data:{mime};base64,{b64}"

    payload = {
        "model": os.getenv("VISION_MODEL", DEFAULT_VISION_MODEL),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "KYC Document Processor",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    logger.info(f"Vision model response: {content[:200]}...")
    return _parse_llm_response(content)


async def extract_from_text(ocr_text: str, document_type: str, api_key: Optional[str] = None) -> dict:
    """Send OCR'd text to an LLM and get structured fields back."""
    key = api_key or _get_api_key()
    if not key:
        raise ValueError("OpenRouter API key not configured")

    prompt = TEXT_PROMPT.format(document_type=document_type, ocr_text=ocr_text[:4000])

    payload = {
        "model": os.getenv("TEXT_LLM_MODEL", DEFAULT_TEXT_MODEL),
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "KYC Document Processor",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    logger.info(f"Text LLM response: {content[:200]}...")
    return _parse_llm_response(content)
