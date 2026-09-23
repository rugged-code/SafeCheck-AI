from pydantic import BaseModel, field_validator
from typing import Optional

class AdverseEventChunk(BaseModel):
    report_id : str
    drug_name : str
    receive_date: Optional[str]

    serious : bool
    outcome : list[str]
    reactions : list[str]
    patient_age : Optional[str]
    patient_sex : Optional[str]
    has_narrative : bool

    chunk_text : str

    @field_validator("drug_name")
    @classmethod
    def normalize_drug_name(cls, v: str)->str:
        return v.strip().lower()

    @field_validator("chunk_text")
    @classmethod
    def chunk_must_not_be_empty(cls, v : str)->str:
        if not v or not v.strip():
            raise ValueError("chunk_text must not be empty - flattening failed")
        return v.strip()