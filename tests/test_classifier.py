"""Tests for document classifier."""

from classifier import classify_document


class TestClassifier:
    def test_aadhaar_keywords(self):
        text = "Government of India UIDAI AADHAAR card. Enrolment No. Name: Priya Sharma"
        doc_type, confidence = classify_document(text)
        assert doc_type == "aadhaar"
        assert confidence >= 0.75

    def test_pan_keywords(self):
        text = "INCOME TAX DEPARTMENT PERMANENT ACCOUNT NUMBER PAN CARD ABCDE1234F"
        doc_type, confidence = classify_document(text)
        assert doc_type == "pan"
        assert confidence >= 0.75

    def test_dl_keywords(self):
        text = "DRIVING LICENCE DL No: KA05 20190001234 Vehicle Class: MCWG, LMV"
        doc_type, confidence = classify_document(text)
        assert doc_type == "driving_licence"
        assert confidence >= 0.75

    def test_passport_mrz(self):
        text = "PASSPORT P<INDSHARMA<<PRIYA<<<<<<<<<<<<<<<<<<<<<< A1234567<<8IND9004120F2904150<<<<<<<<<<<<<<04"
        doc_type, confidence = classify_document(text)
        assert doc_type == "passport"

    def test_unknown_document(self):
        text = "This is just a random receipt from a grocery store"
        doc_type, confidence = classify_document(None)
        assert doc_type is None

    def test_unrecognized_content(self):
        text = "Random unrelated text without any document keywords"
        doc_type, _ = classify_document(text)
        assert doc_type is None
