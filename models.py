from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    AADHAAR = "aadhaar"
    PAN = "pan"
    VOTER_ID = "voter_id"
    DRIVING_LICENCE = "driving_licence"
    PASSPORT = "passport"


class AadhaarFields(BaseModel):
    name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    aadhaar_number: Optional[str] = None


class PanFields(BaseModel):
    name: Optional[str] = None
    fathers_name: Optional[str] = None
    dob: Optional[str] = None
    pan_number: Optional[str] = None


class VoterIdFields(BaseModel):
    name: Optional[str] = None
    relative_name: Optional[str] = None
    dob: Optional[str] = None
    voter_id_number: Optional[str] = None
    address: Optional[str] = None
    constituency: Optional[str] = None


class DLFields(BaseModel):
    name: Optional[str] = None
    dob: Optional[str] = None
    dl_number: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    vehicle_classes: Optional[str] = None
    address: Optional[str] = None


class PassportFields(BaseModel):
    name: Optional[str] = None
    dob: Optional[str] = None
    passport_number: Optional[str] = None
    nationality: Optional[str] = None
    expiry_date: Optional[str] = None
    mrz_line1: Optional[str] = None
    mrz_line2: Optional[str] = None


ExtractedFields = AadhaarFields | PanFields | VoterIdFields | DLFields | PassportFields | dict


class FieldConfidence(BaseModel):
    field: str
    confidence: float
    raw_text: Optional[str] = None


class ExtractionResult(BaseModel):
    document_type: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    fields: dict = Field(default_factory=dict)
    masked: bool = False
    extraction_warnings: list[str] = Field(default_factory=list)
    field_confidences: list[FieldConfidence] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    document_type: Optional[str] = None
