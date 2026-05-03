"""Tests for masking module."""

from masking import mask_aadhaar, mask_fields, safe_log


class TestMaskAadhaar:
    def test_masks_first_8_digits(self):
        result = mask_aadhaar("9876 5432 1098")
        assert result == "XXXX XXXX 1098"

    def test_masks_running_digits(self):
        result = mask_aadhaar("123456789012")
        assert result == "XXXX XXXX 9012"

    def test_non_aadhaar_passes_through(self):
        assert mask_aadhaar("ABCDE1234F") == "ABCDE1234F"
        assert mask_aadhaar("Not a number") == "Not a number"

    def test_already_masked_remains(self):
        result = mask_aadhaar("XXXX XXXX 1098")
        assert result == "XXXX XXXX 1098"


class TestMaskFields:
    def test_aadhaar_masked(self):
        fields = {"name": "Priya", "aadhaar_number": "9876 5432 1098"}
        masked = mask_fields(fields, "aadhaar")
        assert masked["aadhaar_number"] == "XXXX XXXX 1098"
        assert masked["name"] == "Priya"

    def test_non_aadhaar_not_masked(self):
        fields = {"name": "Rahul", "pan_number": "ABCDE1234F"}
        masked = mask_fields(fields, "pan")
        assert masked["pan_number"] == "ABCDE1234F"
        assert "masked" not in masked.values()


class TestSafeLog:
    def test_scrubs_aadhaar(self):
        msg = "Processing document with Aadhaar: 1234 5678 9012"
        scrubbed = safe_log(msg)
        assert "XXXX XXXX 9012" in scrubbed
        assert "1234" not in scrubbed
        assert "5678" not in scrubbed

    def test_non_aadhaar_unchanged(self):
        msg = "Processing PAN: ABCDE1234F"
        assert safe_log(msg) == msg
