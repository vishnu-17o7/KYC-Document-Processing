"""FastAPI endpoint for KYC document extraction."""

import logging
import os
from io import BytesIO
from typing import Optional

import fitz
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from classifier import classify_document
from extractors import AadhaarExtractor, PanExtractor, VoterIdExtractor, DLExtractor, PassportExtractor
from llm_extractor import extract_with_vision, extract_from_text
from masking import mask_fields, safe_log
from models import ExtractionResult, ErrorResponse, FieldConfidence
from ocr_client import extract_text as ocr_extract_text, OCRError
from preprocess import preprocess_image, estimate_scan_quality
from validators import validate_fields

load_dotenv()

OCR_SPACE_KEY = os.getenv("OCR_SPACE_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
EXTRACTION_MODE = os.getenv("EXTRACTION_MODE", "hybrid")

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


async def _process_image(image: np.ndarray, raw_bytes: bytes, language: str = "eng") -> ExtractionResult:
    quality_score, quality_warnings = estimate_scan_quality(image)
    extraction_warnings = []

    if quality_score < 0.5:
        extraction_warnings.append("low_scan_quality")

    result = None
    ocr_text = None

    if EXTRACTION_MODE in ("hybrid", "vision") and OPENROUTER_KEY:
        result = await _try_vision_extraction(raw_bytes, extraction_warnings)

    if not result and EXTRACTION_MODE in ("hybrid", "llm") and OPENROUTER_KEY and OCR_SPACE_KEY:
        result, ocr_text = await _try_llm_text_extraction(image, extraction_warnings, language)

    if not result:
        result = await _try_regex_extraction(image, extraction_warnings, language, ocr_text)

    result.extraction_warnings.extend(quality_warnings)
    return result


async def _try_vision_extraction(raw_bytes: bytes, warnings: list[str]) -> Optional[ExtractionResult]:
    """Tier 1: Send image directly to a vision model."""
    logger.info("Tier 1: Attempting vision model extraction")
    try:
        data = await extract_with_vision(raw_bytes)
    except Exception as e:
        logger.warning(f"Vision extraction failed: {e}")
        return None

    doc_type = data.get("document_type")
    if not doc_type:
        return None

    fields = data.get("fields", {})
    llm_confidence = data.get("confidence", 0.7)

    fields = mask_fields(fields, doc_type)
    masked = doc_type == "aadhaar"

    validation_warnings = validate_fields(doc_type, fields)
    warnings.extend(data.get("extraction_warnings", []))
    warnings.extend(validation_warnings)

    logger.info(f"Vision extraction succeeded: {doc_type} (confidence: {llm_confidence})")
    return ExtractionResult(
        document_type=doc_type,
        confidence=round(llm_confidence, 2),
        fields=fields,
        masked=masked,
        extraction_warnings=list(warnings),
        field_confidences=_build_field_confidences(fields),
    )


async def _try_llm_text_extraction(image: np.ndarray, warnings: list[str], language: str) -> tuple[Optional[ExtractionResult], Optional[str]]:
    """Tier 2: OCR → classify → text LLM extract. Returns (result, ocr_text)."""
    logger.info("Tier 2: Attempting OCR + text LLM extraction")
    processed = preprocess_image(image)

    try:
        ocr_result = await ocr_extract_text(processed, ocr_space_key=OCR_SPACE_KEY, openrouter_key=OPENROUTER_KEY, language=language)
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return None, None

    text = safe_log(ocr_result.text)
    logger.info(f"OCR extracted {len(text)} chars")

    doc_type, cls_conf = classify_document(text, image.shape)
    if not doc_type:
        return None, text

    try:
        data = await extract_from_text(text, doc_type)
    except Exception as e:
        logger.warning(f"Text LLM extraction failed: {e}")
        return None, text

    fields = data.get("fields", {})
    llm_confidence = data.get("confidence", cls_conf)

    fields = mask_fields(fields, doc_type)
    masked = doc_type == "aadhaar"

    validation_warnings = validate_fields(doc_type, fields)
    warnings.extend(data.get("extraction_warnings", []))
    warnings.extend(validation_warnings)

    final_conf = round((ocr_result.confidence * 0.3 + llm_confidence * 0.7), 2)
    logger.info(f"Text LLM extraction succeeded: {doc_type} (confidence: {final_conf})")
    return ExtractionResult(
        document_type=doc_type,
        confidence=final_conf,
        fields=fields,
        masked=masked,
        extraction_warnings=list(warnings),
        field_confidences=_build_field_confidences(fields),
    ), text


async def _try_regex_extraction(image: np.ndarray, warnings: list[str], language: str, cached_ocr_text: Optional[str] = None) -> ExtractionResult:
    """Tier 3: OCR → classify → regex extract (legacy fallback)."""
    logger.info("Tier 3: Falling back to regex extraction")

    if cached_ocr_text:
        text = cached_ocr_text
        logger.info(f"Reusing OCR text from Tier 2 ({len(text)} chars)")
    else:
        processed = preprocess_image(image)
        try:
            ocr_result = await ocr_extract_text(processed, ocr_space_key=OCR_SPACE_KEY, openrouter_key=OPENROUTER_KEY, language=language)
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
    warnings.extend(validation_warnings)

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
        extraction_warnings=warnings,
        field_confidences=field_confidences,
    )


def _build_field_confidences(fields: dict) -> list[FieldConfidence]:
    confidences = []
    for key, value in fields.items():
        if value:
            confidences.append(FieldConfidence(field=key, confidence=0.85, raw_text=str(value)))
        else:
            confidences.append(FieldConfidence(field=key, confidence=0.0))
    return confidences


@app.post("/extract", response_model=ExtractionResult)
async def extract_document(file: UploadFile = File(...)):
    contents = await file.read()
    filename = (file.filename or "").lower()

    logger.info(f"Processing: {file.filename} ({len(contents)} bytes)")

    if filename.endswith(".pdf"):
        images = _pdf_to_images(contents)
        if not images:
            raise HTTPException(status_code=400, detail="PDF has no readable pages")
        result = await _process_image(images[0], contents)
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
        result = await _process_image(image, contents)

    if result.document_type is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unable to identify document. Supported types: aadhaar, pan, voter_id, driving_licence, passport",
                "document_type": None,
            },
        )

    return result


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KYC Document Processor</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #1a1a2e; min-height: 100vh; }
.header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 24px 0; text-align: center; }
.header h1 { font-size: 1.6rem; font-weight: 600; }
.header p { font-size: 0.85rem; opacity: 0.7; margin-top: 4px; }
.container { max-width: 720px; margin: 0 auto; padding: 24px 16px; }
.card { background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); padding: 24px; margin-bottom: 20px; }
.drop-zone { border: 2px dashed #cbd5e1; border-radius: 10px; padding: 36px 20px; text-align: center; cursor: pointer; transition: border-color 0.2s, background 0.2s; }
.drop-zone:hover, .drop-zone.active { border-color: #3b82f6; background: #f0f7ff; }
.drop-zone svg { width: 40px; height: 40px; color: #94a3b8; margin-bottom: 8px; }
.drop-zone .label { font-size: 0.9rem; color: #64748b; }
.drop-zone .hint { font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }
.drop-zone input { display: none; }
.file-name { margin-top: 12px; font-size: 0.85rem; color: #3b82f6; text-align: center; display: none; }
.btn { display: block; width: 100%; padding: 12px; background: #3b82f6; color: white; border: none; border-radius: 8px; font-size: 0.95rem; font-weight: 500; cursor: pointer; margin-top: 16px; transition: background 0.2s; }
.btn:hover { background: #2563eb; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.spinner { display: none; width: 20px; height: 20px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto; }
@keyframes spin { to { transform: rotate(360deg); } }
.error { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; padding: 12px 16px; border-radius: 8px; font-size: 0.85rem; margin-top: 16px; display: none; }
.result { display: none; }
.doc-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.badge-aadhaar { background: #dbeafe; color: #1e40af; }
.badge-pan { background: #fef3c7; color: #92400e; }
.badge-voter { background: #d1fae5; color: #065f46; }
.badge-dl { background: #ede9fe; color: #5b21b6; }
.badge-passport { background: #fce7f3; color: #9d174d; }
.conf-bar-wrap { display: flex; align-items: center; gap: 10px; margin: 12px 0; }
.conf-bar { flex: 1; height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
.conf-label { font-size: 0.8rem; color: #64748b; min-width: 40px; }
.field-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
.field-table td { padding: 8px 12px; font-size: 0.85rem; border-bottom: 1px solid #f1f5f9; }
.field-table td:first-child { color: #64748b; font-weight: 500; width: 140px; white-space: nowrap; }
.field-table td:last-child { color: #1a1a2e; word-break: break-all; }
.masked-field { background: #fffbeb; color: #92400e !important; border-radius: 4px; padding: 2px 6px; font-family: monospace; font-size: 0.82rem; }
.masked-icon { display: inline-block; width: 14px; height: 14px; margin-right: 4px; vertical-align: middle; }
.warnings-box { background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 12px 16px; margin-top: 12px; }
.warnings-box h4 { font-size: 0.8rem; color: #92400e; margin-bottom: 6px; }
.warnings-box ul { margin-left: 18px; font-size: 0.8rem; color: #a16207; }
.btn-small { display: inline-block; padding: 6px 14px; background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 0.78rem; cursor: pointer; margin-top: 12px; transition: all 0.2s; }
.btn-small:hover { background: #e2e8f0; }
.footer { text-align: center; padding: 16px; font-size: 0.75rem; color: #94a3b8; }
</style>
</head>
<body>

<div class="header">
  <h1>KYC Document Processor</h1>
  <p>Extract structured data from Indian identity documents</p>
</div>

<div class="container">
  <div class="card">
    <div class="drop-zone" id="dropZone">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
      <div class="label">Drop image or PDF here</div>
      <div class="hint">PNG, JPG, PDF &bull; Max 10MB</div>
      <input type="file" id="fileInput" accept="image/*,.pdf">
    </div>
    <div class="file-name" id="fileName"></div>
    <button class="btn" id="processBtn" disabled>
      <span id="btnText">Select a file to begin</span>
      <div class="spinner" id="spinner"></div>
    </button>
  </div>

  <div class="error" id="errorBox"></div>

  <div class="card result" id="resultCard">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
      <span class="doc-badge" id="docBadge"></span>
      <span style="font-size:0.8rem;color:#94a3b8;" id="maskedLabel"></span>
    </div>
    <div class="conf-bar-wrap">
      <span class="conf-label" id="confPct">0%</span>
      <div class="conf-bar"><div class="conf-bar-fill" id="confFill" style="width:0%"></div></div>
    </div>
    <table class="field-table" id="fieldTable"></table>
    <div class="warnings-box" id="warningsBox" style="display:none;"></div>
    <button class="btn-small" id="copyBtn">Copy JSON</button>
  </div>
</div>

<div class="footer">Supports Aadhaar, PAN, Voter ID, Driving Licence &amp; Passport</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const processBtn = document.getElementById('processBtn');
const btnText = document.getElementById('btnText');
const spinner = document.getElementById('spinner');
const errorBox = document.getElementById('errorBox');
const resultCard = document.getElementById('resultCard');
let selectedFile = null;

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('active'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('active'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('active');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) setFile(fileInput.files[0]); });

function setFile(file) {
  selectedFile = file;
  fileName.textContent = file.name + ' (' + formatSize(file.size) + ')';
  fileName.style.display = 'block';
  processBtn.disabled = false;
  btnText.textContent = 'Process Document';
  errorBox.style.display = 'none';
}

function formatSize(bytes) {
  return bytes < 1024*1024 ? (bytes/1024).toFixed(1)+' KB' : (bytes/(1024*1024)).toFixed(1)+' MB';
}

processBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  processBtn.disabled = true;
  btnText.style.display = 'none';
  spinner.style.display = 'block';
  errorBox.style.display = 'none';
  resultCard.style.display = 'none';

  const form = new FormData();
  form.append('file', selectedFile);

  try {
    const res = await fetch('/extract', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail?.error || data.detail || 'Processing failed');
    renderResult(data);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.style.display = 'block';
  } finally {
    processBtn.disabled = false;
    btnText.style.display = 'inline';
    spinner.style.display = 'none';
  }
});

function renderResult(data) {
  resultCard.style.display = 'block';

  const badge = document.getElementById('docBadge');
  const type = data.document_type || 'unknown';
  badge.textContent = type.replace(/_/g, ' ');
  badge.className = 'doc-badge badge-' + type;

  document.getElementById('maskedLabel').textContent = data.masked ? 'Aadhaar masked' : '';

  const pct = Math.round(data.confidence * 100);
  document.getElementById('confPct').textContent = pct + '%';
  const fill = document.getElementById('confFill');
  fill.style.width = pct + '%';
  fill.style.background = pct >= 80 ? '#22c55e' : pct >= 50 ? '#f59e0b' : '#ef4444';

  const table = document.getElementById('fieldTable');
  table.innerHTML = '';
  Object.entries(data.fields).forEach(([key, val]) => {
    if (val === null || val === undefined || val === '') return;
    const isMasked = data.masked && key === 'aadhaar_number';
    const display = isMasked
      ? '<span class="masked-field"><svg class="masked-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/><line x1="12" y1="15" x2="12" y2="19"/></svg>' + val + '</span>'
      : val;
    table.innerHTML += '<tr><td>' + key.replace(/_/g, ' ') + '</td><td>' + display + '</td></tr>';
  });

  const warningsBox = document.getElementById('warningsBox');
  if (data.extraction_warnings && data.extraction_warnings.length) {
    warningsBox.style.display = 'block';
    warningsBox.innerHTML = '<h4>Warnings</h4><ul>' +
      data.extraction_warnings.map(w => '<li>' + w.replace(/_/g, ' ') + '</li>').join('') +
      '</ul>';
  } else {
    warningsBox.style.display = 'none';
  }

  document.getElementById('copyBtn').onclick = () => {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
  };
}
</script>

</body>
</html>"""


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "extraction_mode": EXTRACTION_MODE,
        "ocr_space_configured": bool(OCR_SPACE_KEY),
        "openrouter_configured": bool(OPENROUTER_KEY),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
