import base64
from io import BytesIO
from typing import Optional
import httpx
import numpy as np
from PIL import Image


OCR_SPACE_URL = "https://api.ocr.space/parse/image"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OCR_MODEL = "baidu/qianfan-ocr-fast:free"


class OCRError(Exception):
    pass


class OCRResult:
    def __init__(self, text: str, confidence: float, words: list[dict]):
        self.text = text
        self.confidence = confidence
        self.words = words


async def _call_ocr_space(image: np.ndarray, api_key: str, language: str = "eng") -> OCRResult:
    """Call the ocr.space API with a numpy image array."""
    pil_image = Image.fromarray(image)

    buf = BytesIO()
    pil_image.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    if buf.tell() > 1_000_000:
        pil_image.save(buf, format="JPEG", quality=70, optimize=True)
        buf.seek(0)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            OCR_SPACE_URL,
            headers={"apikey": api_key},
            files={"file": ("document.png", buf, "image/png")},
            data={
                "language": language,
                "isOverlayRequired": "true",
                "isCreateSearchablePdf": "false",
                "isSearchablePdfHideTextLayer": "false",
                "detectOrientation": "true",
                "scale": "true",
                "OCREngine": "2",
            },
        )
        response.raise_for_status()
        data = response.json()

    if data.get("IsErroredOnProcessing"):
        raise OCRError(f"ocr.space error: {data.get('ErrorMessage', 'unknown')}")

    parsed_results = data.get("ParsedResults")
    if not parsed_results:
        raise OCRError("ocr.space returned no results")

    all_text = ""
    all_confidences = []
    all_words = []

    for result in parsed_results:
        exit_code = result.get("FileParseExitCode")
        if exit_code != 1:
            err = result.get("ParsedText", "")
            raise OCRError(f"ocr.space parse error (code {exit_code}): {err}")

        all_text += result.get("ParsedText", "") + "\n"

        overlay = result.get("TextOverlay", {})
        for line in overlay.get("Lines", []):
            line_words = []
            for w in line.get("Words", []):
                line_words.append({
                    "text": w.get("WordText", ""),
                    "left": w.get("Left", 0),
                    "top": w.get("Top", 0),
                    "width": w.get("Width", 0),
                    "height": w.get("Height", 0),
                })
            all_words.append({"line_text": line.get("LineText", ""), "words": line_words})

        processing_time = data.get("ProcessingTimeInMilliseconds", 0)
        if processing_time > 0:
            all_confidences.append(0.85)

    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.7
    return OCRResult(text=all_text.strip(), confidence=avg_confidence, words=all_words)


async def _call_openrouter_vision(image: np.ndarray, api_key: str) -> OCRResult:
    """Call OpenRouter vision model as OCR fallback."""
    pil_image = Image.fromarray(image)
    buf = BytesIO()
    pil_image.save(buf, format="JPEG", quality=85, optimize=True)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    payload = {
        "model": DEFAULT_OCR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract all visible text from this Indian identity document image. "
                            "Return ONLY the extracted text exactly as it appears. "
                            "Do not add commentary or explanation."
                        ),
                    },
                ],
            }
        ],
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "KYC Document Processor",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    return OCRResult(text=content.strip(), confidence=0.80, words=[])


async def extract_text(
    image: np.ndarray,
    ocr_space_key: Optional[str] = None,
    openrouter_key: Optional[str] = None,
    language: str = "eng",
) -> OCRResult:
    """
    Extract text from a preprocessed image using ocr.space (primary) with
    OpenRouter vision model as fallback.
    """
    ocr_errors = []

    if ocr_space_key:
        try:
            return await _call_ocr_space(image, ocr_space_key, language)
        except Exception as e:
            ocr_errors.append(f"ocr.space: {e}")

    if openrouter_key:
        try:
            return await _call_openrouter_vision(image, openrouter_key)
        except Exception as e:
            ocr_errors.append(f"OpenRouter: {e}")

    error_detail = "; ".join(ocr_errors) if ocr_errors else "No OCR API keys configured"
    raise OCRError(f"OCR failed: {error_detail}")
