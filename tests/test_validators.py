"""Tests for validators."""

from validators import validate_pan, validate_dl, validate_aadhaar, validate_passport


class TestValidateAadhaar:
    def test_valid(self):
        assert not validate_aadhaar({"aadhaar_number": "123456789012", "dob": "1990-04-12"})

    def test_missing_number(self):
        warnings = validate_aadhaar({})
        assert any("missing" in w for w in warnings)

    def test_future_dob(self):
        warnings = validate_aadhaar({"dob": "2099-01-01"})
        assert "dob_in_future" in warnings


class TestValidatePan:
    def test_valid(self):
        fields = {"pan_number": "ABCKE1234F", "name": "Rahul Kumar", "dob": "1985-08-15"}
        assert not validate_pan(fields)

    def test_invalid_format(self):
        warnings = validate_pan({"pan_number": "12345ABCDE"})
        assert "invalid_pan_format" in warnings

    def test_4th_char_mismatch(self):
        fields = {"pan_number": "ABCDE1234F", "name": "Rahul Xyz"}
        warnings = validate_pan(fields)
        assert any("mismatch" in w for w in warnings)


class TestValidateDL:
    def test_valid(self):
        warnings = validate_dl({
            "dl_number": "KA0520190001234",
            "dob": "1992-06-22",
            "valid_to": "2039-01-14",
        })
        assert not warnings

    def test_expired(self):
        warnings = validate_dl({
            "dl_number": "KA05",
            "dob": "1992-06-22",
            "valid_to": "2020-01-01",
        })
        assert "dl_expired" in warnings

    def test_under_18(self):
        from datetime import datetime
        year = datetime.now().year - 16
        warnings = validate_dl({"dob": f"{year}-01-01"})
        assert "dl_holder_under_18" in warnings


class TestValidatePassport:
    def test_valid(self):
        warnings = validate_passport({
            "passport_number": "A1234567",
            "expiry_date": "2029-04-29",
            "mrz_line2": "A1234567<<8IND9004120F2904150<<<<<<<<<<<<<<04",
        })
        assert not warnings

    def test_expired(self):
        warnings = validate_passport({
            "passport_number": "A1234567",
            "expiry_date": "2020-01-01",
        })
        assert "passport_expired" in warnings

    def test_mrz_mismatch(self):
        warnings = validate_passport({
            "passport_number": "Z9999999",
            "expiry_date": "2029-04-29",
            "mrz_line2": "A1234567<<8IND9004120F2904150<<<<<<<<<<<<<<04",
        })
        assert "mrz_passport_mismatch" in warnings
