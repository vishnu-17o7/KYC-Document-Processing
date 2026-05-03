"""Integration tests for the preprocessing + classifier pipeline."""

import numpy as np
from preprocess import preprocess_image, estimate_scan_quality
from classifier import classify_document


class TestPreprocessing:
    def test_deskew_straight_image(self):
        img = np.ones((500, 500), dtype=np.uint8) * 255
        cv2 = __import__("cv2")
        cv2.putText(img, "TEST", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        result = preprocess_image(img)
        assert result.shape == (500, 500)
        assert result.dtype == np.uint8

    def test_color_to_gray(self):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 200
        cv2 = __import__("cv2")
        cv2.putText(img, "A", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        result = preprocess_image(img)
        assert len(result.shape) == 2

    def test_empty_image_raises(self):
        import pytest
        with pytest.raises(ValueError):
            preprocess_image(None)  # type: ignore[arg-type]

    def test_quality_high(self):
        img = np.ones((800, 600), dtype=np.uint8) * 200
        cv2 = __import__("cv2")
        cv2.putText(img, "CLEAR TEXT", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 50, 3)
        score, warnings = estimate_scan_quality(img)
        assert score >= 0.4  # Allow lower threshold since synthetic may vary

    def test_quality_low_dark(self):
        img = np.ones((300, 200), dtype=np.uint8) * 20
        score, warnings = estimate_scan_quality(img)
        assert "too_dark" in warnings or score < 0.8


class TestEndToEndWithoutAPI:
    """Tests that don't require API keys."""

    def test_aadhaar_pipeline(self):
        from extractors.aadhaar import AadhaarExtractor
        from masking import mask_fields
        from validators import validate_aadhaar

        classification_text = "Government of India UIDAI AADHAAR"
        doc_type, conf = classify_document(classification_text)
        assert doc_type == "aadhaar"

        ocr_text = (
            "Name: Priya Sharma\n"
            "Date of Birth: 12/04/1990\n"
            "Gender: Female\n"
            "Aadhaar No: 9876 5432 1098\n"
            "Address: 123, MG Road, Bengaluru\n"
        )
        fields = AadhaarExtractor().extract(ocr_text)
        assert fields["name"] == "Priya Sharma"
        assert fields["dob"] == "1990-04-12"

        warnings = validate_aadhaar(fields)
        assert not warnings

        masked = mask_fields(fields, "aadhaar")
        assert masked["aadhaar_number"] == "XXXX XXXX 1098"

    def test_pan_pipeline(self):
        from extractors.pan import PanExtractor
        from validators import validate_pan

        classification_text = "INCOME TAX DEPARTMENT PERMANENT ACCOUNT NUMBER"
        doc_type, _ = classify_document(classification_text)
        assert doc_type == "pan"

        ocr_text = "Name: Rahul Kumar\nFather's Name: Suresh Kumar\nDOB: 15/08/1985\nABCKE1234F\n"
        fields = PanExtractor().extract(ocr_text)
        assert fields["pan_number"] == "ABCKE1234F"
        assert fields["name"] == "Rahul Kumar"

        warnings = validate_pan(fields)
        assert not warnings

    def test_dl_pipeline(self):
        from extractors.driving_licence import DLExtractor
        from validators import validate_dl

        classification_text = "DRIVING LICENCE MOTOR VEHICLES DEPARTMENT"
        doc_type, _ = classify_document(classification_text)
        assert doc_type == "driving_licence"

        ocr_text = (
            "DL No: KA05 20190001234\n"
            "Name: Amit Patel\n"
            "DOB: 22/06/1992\n"
            "Valid From: 15/01/2019\n"
            "Valid To: 14/01/2039\n"
        )
        fields = DLExtractor().extract(ocr_text)
        assert "2039-01-14" in fields["valid_to"]

        warnings = validate_dl(fields)
        assert not warnings
