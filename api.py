"""FastAPI endpoint for KYC document extraction."""

import logging
import os
from io import BytesIO
from typing import Optional

import fitz
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from classifier import classify_document
from extractors import AadhaarExtractor, PanExtractor, VoterIdExtractor, DLExtractor, PassportExtractor
from masking import mask_fields, safe_log
from models import ExtractionResult, ErrorResponse
from ocr_client import extract_text as ocr_extract_text, OCRError
from preprocess import preprocess_image, estimate_scan_quality
from validators import validate_fields

load_dotenv()

OCR_SPACE_KEY = os.getenv("OCR_SPACE_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KYC Document Processing API",
    description="Extract structured data from Indian identity documents",
    version="1.0.0",
)

EXTRACTOR_MAP = {
    "aadhaar": AadhaarExtractor(),
    "pan": PanExtractor(),
    "voter_id": VoterIdExtractor(),
    "driving_licence": DLExtractor(),
    "passport": PassportExtractor(),
}


def _pdf_to_images(pdf_bytes: bytes) -> list[np.ndarray]:
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.array(img))
    doc.close()
    return images


async def _process_image(image: np.ndarray, language: str = "eng") -> ExtractionResult:
    quality_score, quality_warnings = estimate_scan_quality(image)
    extraction_warnings = []

    if quality_score < 0.5:
        extraction_warnings.append("low_scan_quality")

    processed = preprocess_image(image)

    try:
        ocr_result = await ocr_extract_text(
            processed,
            ocr_space_key=OCR_SPACE_KEY,
            openrouter_key=OPENROUTER_KEY,
            language=language,
        )
    except OCRError as e:
        raise HTTPException(status_code=502, detail=f"OCR service unavailable: {e}")

    text = safe_log(ocr_result.text)
    logger.info(f"Extracted text ({len(text)} chars)")

    doc_type, confidence = classify_document(text, image.shape)
    if not doc_type:
        return ExtractionResult(
            document_type=None,
            confidence=confidence,
            extraction_warnings=["unrecognized_document"],
        )

    extractor = EXTRACTOR_MAP.get(doc_type)
    if not extractor:
        return ExtractionResult(
            document_type=None,
            confidence=0.0,
            extraction_warnings=["unsupported_document_type"],
        )

    fields = extractor.extract(text)
    field_confidences = extractor.get_field_confidences(text, fields)

    validation_warnings = validate_fields(doc_type, fields)
    extraction_warnings.extend(validation_warnings)
    extraction_warnings.extend(quality_warnings)

    fields = mask_fields(fields, doc_type)
    masked = doc_type == "aadhaar"

    avg_field_conf = (
        sum(fc.confidence for fc in field_confidences if fc.confidence > 0) / max(1, sum(1 for fc in field_confidences if fc.confidence > 0))
        if field_confidences else 0.0
    )
    final_confidence = round((ocr_result.confidence * 0.4 + confidence * 0.3 + avg_field_conf * 0.3), 2)

    return ExtractionResult(
        document_type=doc_type,
        confidence=final_confidence,
        fields=fields,
        masked=masked,
        extraction_warnings=extraction_warnings,
        field_confidences=field_confidences,
    )


@app.post("/extract", response_model=ExtractionResult)
async def extract_document(file: UploadFile = File(...)):
    contents = await file.read()
    filename = (file.filename or "").lower()

    logger.info(f"Processing: {file.filename} ({len(contents)} bytes)")

    if filename.endswith(".pdf"):
        images = _pdf_to_images(contents)
        if not images:
            raise HTTPException(status_code=400, detail="PDF has no readable pages")
        result = await _process_image(images[0])
        if len(images) > 1 and result.document_type:
            logger.info(f"PDF has {len(images)} pages; used page 1 for extraction")
    else:
        try:
            image = Image.open(BytesIO(contents))
            image = np.array(image.convert("RGB"))
        except Exception:
            raise HTTPException(status_code=400, detail="Unsupported image format")
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        result = await _process_image(image)

    if result.document_type is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unable to identify document. Supported types: aadhaar, pan, voter_id, driving_licence, passport",
                "document_type": None,
            },
        )

    return result


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ocr_space_configured": bool(OCR_SPACE_KEY),
        "openrouter_configured": bool(OPENROUTER_KEY),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
