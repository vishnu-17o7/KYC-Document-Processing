"""Tests for field extractors."""

from extractors.aadhaar import AadhaarExtractor
from extractors.pan import PanExtractor
from extractors.driving_licence import DLExtractor
from extractors.voter_id import VoterIdExtractor
from extractors.passport import PassportExtractor


class TestAadhaarExtractor:
    def test_extract_basic(self):
        text = (
            "Government of India\n"
            "AADHAAR\n"
            "Name: Priya Sharma\n"
            "Date of Birth: 12/04/1990\n"
            "Gender: Female\n"
            "Address: 123, MG Road, Bengaluru\n"
            "Aadhaar No: 9876 5432 1098\n"
        )
        fields = AadhaarExtractor().extract(text)
        assert fields["name"] == "Priya Sharma"
        assert fields["dob"] == "1990-04-12"
        assert fields["gender"] == "Female"
        assert "9876 5432 1098" in fields["aadhaar_number"]
        assert "Bengaluru" in fields["address"]

    def test_aadhaar_number_format(self):
        text = "Aadhaar No: 1234 5678 9012"
        fields = AadhaarExtractor().extract(text)
        assert fields["aadhaar_number"] == "1234 5678 9012"

    def test_missing_fields(self):
        text = "Government of India"
        fields = AadhaarExtractor().extract(text)
        assert fields.get("name") is None
        assert fields.get("dob") is None


class TestPanExtractor:
    def test_extract_basic(self):
        text = (
            "INCOME TAX DEPARTMENT\n"
            "PERMANENT ACCOUNT NUMBER\n"
            "Name: Rahul Kumar\n"
            "Father's Name: Suresh Kumar\n"
            "Date of Birth: 15/08/1985\n"
            "ABCDE1234F\n"
        )
        fields = PanExtractor().extract(text)
        assert fields["name"] == "Rahul Kumar"
        assert fields["fathers_name"] == "Suresh Kumar"
        assert fields["dob"] == "1985-08-15"
        assert fields["pan_number"] == "ABCDE1234F"

    def test_invalid_pan_rejected(self):
        text = "Name: Test Person\n12345ABCDE\n"
        fields = PanExtractor().extract(text)
        assert fields.get("pan_number") is None

    def test_pan_alpha_numeric(self):
        text = "HWRTY1234Z Name: Amit Patel"
        fields = PanExtractor().extract(text)
        assert fields["pan_number"] == "HWRTY1234Z"


class TestDLExtractor:
    def test_extract_basic(self):
        text = (
            "DRIVING LICENCE\n"
            "DL No: KA05 20190001234\n"
            "Name: Amit Patel\n"
            "Date of Birth: 22/06/1992\n"
            "Valid From: 15/01/2019\n"
            "Valid To: 14/01/2039\n"
            "Vehicle Class: MCWG, LMV, TRANS\n"
        )
        fields = DLExtractor().extract(text)
        assert fields["name"] == "Amit Patel"
        assert fields["dob"] == "1992-06-22"
        assert "KA05 20190001234" in fields["dl_number"]
        assert fields["valid_from"] == "2019-01-15"
        assert fields["valid_to"] == "2039-01-14"
        assert "LMV" in fields["vehicle_classes"]


class TestVoterIdExtractor:
    def test_extract_voter_id(self):
        text = (
            "ELECTION COMMISSION OF INDIA\n"
            "Voter Identity Card\n"
            "EPIC No: ABC1234567\n"
            "Name: Vikram Singh\n"
            "Father's Name: Ranjit Singh\n"
            "Date of Birth: 10/03/1978\n"
        )
        fields = VoterIdExtractor().extract(text)
        assert fields["voter_id_number"] == "ABC1234567"
        assert fields["name"] == "Vikram Singh"
        assert fields["relative_name"] == "Ranjit Singh"
        assert fields["dob"] == "1978-03-10"


class TestPassportExtractor:
    def test_extract_mrz(self):
        text = (
            "PASSPORT\n"
            "Passport No: A1234567\n"
            "Nationality: IND\n"
            "Date of Expiry: 29/04/2029\n"
            "P<INDSHARMA<<PRIYA<<<<<<<<<<<<<<<<<<<<<<\n"
            "A1234567<<IND9004120F2904150<<<<<<<<<<<<<<04\n"
        )
        fields = PassportExtractor().extract(text)
        assert "SHARMA" in fields.get("name", "").upper()
        assert "PRIYA" in fields.get("name", "").upper()
        assert "A1234567" == fields.get("passport_number")
        assert fields.get("dob") == "1990-04-12"
        assert fields.get("nationality") == "IND"

    def test_picks_up_passport_number(self):
        text = (
            "PASSPORT\n"
            "Passport No: Z9876543\n"
            "P<INDSINGH<<RAHUL<<<<<<<<<<<<<<<<<<<<\n"
            "Z9876543<<8IND8503101M2506152<<<<<<<<<<<<<<00\n"
        )
        fields = PassportExtractor().extract(text)
        assert fields.get("passport_number") == "Z9876543"
