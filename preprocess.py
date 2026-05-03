import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess a document image for OCR: deskew, contrast enhance, binarize."""
    if image is None or image.size == 0:
        raise ValueError("Empty image received")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary < 128))
    if len(coords) == 0:
        return binary

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return deskewed


def estimate_scan_quality(image: np.ndarray) -> tuple[float, list[str]]:
    """Return a quality score (0-1) and list of warnings about image quality."""
    warnings = []

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    h, w = gray.shape[:2]
    resolution = (h * w) / 1_000_000

    brightness = np.mean(gray)
    contrast = np.std(gray)

    score = 1.0
    if laplacian_var < 100:
        score -= 0.25
        warnings.append("low_image_sharpness")
    if laplacian_var < 50:
        score -= 0.15
        warnings.append("very_low_sharpness")
    if resolution < 0.5:
        score -= 0.15
        warnings.append("low_resolution")
    if brightness < 40:
        score -= 0.1
        warnings.append("too_dark")
    if brightness > 230:
        score -= 0.1
        warnings.append("too_bright")
    if contrast < 30:
        score -= 0.1
        warnings.append("low_contrast")

    score = max(0.0, min(1.0, score))
    return score, warnings
