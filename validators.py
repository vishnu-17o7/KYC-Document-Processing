"""Document field validators."""

import re
from datetime import datetime, date


def validate_aadhaar(fields: dict) -> list[str]:
    warnings = []
    if fields.get("aadhaar_number"):
        num = fields["aadhaar_number"].replace(" ", "")
        if len(num) != 12 or not num.isdigit():
            warnings.append("invalid_aadhaar_format")
    else:
        warnings.append("missing_aadhaar_number")
    if fields.get("dob"):
        try:
            dob = datetime.strptime(fields["dob"], "%Y-%m-%d")
            if dob > datetime.now():
                warnings.append("dob_in_future")
        except ValueError:
            warnings.append("invalid_dob_format")
    return warnings


def validate_pan(fields: dict) -> list[str]:
    warnings = []
    pan = fields.get("pan_number", "")
    if not pan:
        warnings.append("missing_pan_number")
    elif not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", pan):
        warnings.append("invalid_pan_format")
    else:
        surname = ""
        if fields.get("name"):
            parts = fields["name"].split()
            if parts:
                surname = parts[-1]
        if surname and pan[3].upper() != surname[0].upper():
            warnings.append("pan_4th_char_mismatch_surname")
    if fields.get("dob"):
        try:
            dob = datetime.strptime(fields["dob"], "%Y-%m-%d")
            if dob > datetime.now():
                warnings.append("dob_in_future")
        except ValueError:
            warnings.append("invalid_dob_format")
    return warnings


def validate_voter_id(fields: dict) -> list[str]:
    warnings = []
    if fields.get("voter_id_number"):
        if not re.match(r"^[A-Z]{3}\d{7}$", fields["voter_id_number"]):
            warnings.append("invalid_voter_id_format")
    else:
        warnings.append("missing_voter_id_number")
    if fields.get("dob"):
        try:
            dob = datetime.strptime(fields["dob"], "%Y-%m-%d")
            if dob > datetime.now():
                warnings.append("dob_in_future")
        except ValueError:
            warnings.append("invalid_dob_format")
    return warnings


def validate_dl(fields: dict) -> list[str]:
    warnings = []
    if not fields.get("dl_number"):
        warnings.append("missing_dl_number")
    if fields.get("dob"):
        try:
            dob = datetime.strptime(fields["dob"], "%Y-%m-%d")
            age = (datetime.now() - dob).days / 365.25
            if age < 18:
                warnings.append("dl_holder_under_18")
        except ValueError:
            warnings.append("invalid_dob_format")
    if fields.get("valid_to"):
        try:
            expiry = datetime.strptime(fields["valid_to"], "%Y-%m-%d")
            if expiry < datetime.now():
                warnings.append("dl_expired")
        except ValueError:
            warnings.append("invalid_expiry_format")
    else:
        warnings.append("missing_expiry_date")
    return warnings


def validate_passport(fields: dict) -> list[str]:
    warnings = []
    if not fields.get("passport_number"):
        warnings.append("missing_passport_number")
    if fields.get("expiry_date"):
        try:
            expiry = datetime.strptime(fields["expiry_date"], "%Y-%m-%d")
            if expiry < datetime.now():
                warnings.append("passport_expired")
        except ValueError:
            warnings.append("invalid_expiry_format")
    if fields.get("mrz_line2") and fields.get("passport_number"):
        mrz_pp = fields["mrz_line2"][:9].replace("<", "")
        extracted_pp = fields["passport_number"]
        if mrz_pp and extracted_pp and mrz_pp != extracted_pp:
            warnings.append("mrz_passport_mismatch")
    if fields.get("mrz_line2") and fields.get("dob"):
        try:
            mrz_dob = fields["mrz_line2"][13:19]
            extracted_dob = datetime.strptime(fields["dob"], "%Y-%m-%d")
            if mrz_dob.isdigit():
                mrz_year = int(mrz_dob[:2])
                mrz_month = int(mrz_dob[2:4])
                mrz_day = int(mrz_dob[4:6])
                full_year = 1900 + mrz_year if mrz_year > 50 else 2000 + mrz_year
                if (full_year != extracted_dob.year or
                        mrz_month != extracted_dob.month or
                        mrz_day != extracted_dob.day):
                    warnings.append("mrz_dob_mismatch")
        except (ValueError, IndexError):
            pass
    return warnings


VALIDATOR_MAP = {
    "aadhaar": validate_aadhaar,
    "pan": validate_pan,
    "voter_id": validate_voter_id,
    "driving_licence": validate_dl,
    "passport": validate_passport,
}


def validate_fields(document_type: str, fields: dict) -> list[str]:
    validator = VALIDATOR_MAP.get(document_type)
    if not validator:
        return ["unknown_document_type"]
    return validator(fields)
