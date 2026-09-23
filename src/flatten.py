import json
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from pydantic import ValidationError
from src.schema import AdverseEventChunk



SEX_MAP = {
    "0": "unknown",
    "1": "male",
    "2": "female",
}

OUTCOME_FIELDS = {
    "seriousnessdeath":             "death",
    "seriousnesshospitalization":   "hospitalization",
    "seriousnesslifethreatening":   "life-threatening",
    "seriousnessdisabling":         "disability",
    "seriousnesscongenitalanomali": "congenital anomaly",
    "seriousnessother":             "other serious outcome",
}

AGE_UNIT_MAP = {
    "800": "years",
    "801": "months",
    "802": "weeks",
    "803": "days",
    "804": "hours",
}

AGE_GROUP_MAP = {
    "1": "neonate",
    "2": "infant",
    "3": "child",
    "4": "adolescent",
    "5": "adult",
    "6": "elderly",
}



def extract_reactions(patient: dict) -> list[str]:
    return [
        rx.get("reactionmeddrapt", "").strip()
        for rx in patient.get("reaction", [])
        if rx.get("reactionmeddrapt")
    ]


def extract_outcomes(record: dict) -> list[str]:
    return [
        label
        for field, label in OUTCOME_FIELDS.items()
        if record.get(field) == "1"
    ]


def decode_sex(patient: dict) -> Optional[str]:
    code = patient.get("patientsex")
    return SEX_MAP.get(str(code)) if code else None


def decode_age(patient: dict) -> Optional[str]:
    """
    Uses patientonsetage (numeric, 84.2% present) first.
    Falls back to patientagegroup coded category.
    """
    onset_age  = patient.get("patientonsetage")
    onset_unit = patient.get("patientonsetageunit")

    if onset_age:
        unit = AGE_UNIT_MAP.get(str(onset_unit), "years")
        try:
            age_val = int(float(onset_age))
            return f"{age_val}-{unit}-old"
        except (ValueError, TypeError):
            return None

    code = patient.get("patientagegroup")
    return AGE_GROUP_MAP.get(str(code)) if code else None


def parse_date(raw_date: Optional[str]) -> Optional[str]:
    """Converts openFDA YYYYMMDD string to YYYY-MM-DD."""
    if raw_date and len(raw_date) == 8:
        return f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
    return raw_date



def is_target_drug_suspect(patient: dict, target_drug: str) -> bool:
    """
    Returns True only if target_drug has drugcharacterization == "1" (suspect).
    FIX: str(characterization) comparison — openFDA returns this field as int 1,
    not string "1", so direct == "1" always fails without the cast.
    """
    target = target_drug.upper().strip()

    for drug in patient.get("drug", []):
        name             = drug.get("medicinalproduct", "").upper().strip()
        characterization = drug.get("drugcharacterization", "")

        if target in name and str(characterization) == "1":
            return True

    return False



def build_chunk_text(
    report_id:    str,
    drug_name:    str,
    reactions:    list[str],
    outcomes:     list[str],
    age:          Optional[str],
    sex:          Optional[str],
    receive_date: Optional[str],
    is_serious:   bool,
) -> str:
    patient_parts = []
    if age:
        patient_parts.append(age)
    if sex and sex != "unknown":
        patient_parts.append(sex)
    patient_desc = " ".join(patient_parts) if patient_parts else "patient"

    reaction_str = ", ".join(reactions) if reactions else "unspecified reactions"
    serious_str  = "serious" if is_serious else "non-serious"
    outcome_str  = ", ".join(outcomes) if outcomes else "unspecified outcome"
    date_str     = f" Reported: {receive_date}." if receive_date else ""

    return (
        f"{patient_desc.capitalize()} patient reported {reaction_str} "
        f"following use of {drug_name.upper()}. "
        f"Outcome: {serious_str} — {outcome_str}.{date_str} "
        f"Report ID: {report_id}."
    )



def flatten_record(record: dict, target_drug: str) -> Optional[AdverseEventChunk]:
    """
    Flattens one raw openFDA record into one AdverseEventChunk.
    Returns None for records that should be dropped.
    Drop conditions:
      - No safetyreportid
      - Target drug is not the suspect in this report
      - No reactions present
    """
    report_id = record.get("safetyreportid")
    if not report_id:
        return None

    patient = record.get("patient", {})

    if not is_target_drug_suspect(patient, target_drug):
        return None

    reactions = extract_reactions(patient)
    if not reactions:
        return None

    outcomes     = extract_outcomes(record)
    age          = decode_age(patient)
    sex          = decode_sex(patient)
    receive_date = parse_date(record.get("receivedate"))
    is_serious   = record.get("serious") == "1"

    chunk_text = build_chunk_text(
        report_id    = report_id,
        drug_name    = target_drug,
        reactions    = reactions,
        outcomes     = outcomes,
        age          = age,
        sex          = sex,
        receive_date = receive_date,
        is_serious   = is_serious,
    )

    try:
        return AdverseEventChunk(
            report_id     = report_id,
            drug_name     = target_drug,
            receive_date  = receive_date,
            serious       = is_serious,
            outcome       = outcomes,    # FIX: singular — matches schema field name
            reactions     = reactions,
            patient_age   = age,
            patient_sex   = sex,
            has_narrative = False,
            chunk_text    = chunk_text,
        )
    except (ValueError, ValidationError) as e:   # FIX: Pydantic v2 raises ValidationError
        print(f"Skipping record {report_id}: {e}")
        return None



def flatten_all(records: list[dict], drug_name: str) -> list[AdverseEventChunk]:
    chunks  = []
    dropped = 0

    for record in tqdm(records, desc=f"Flattening '{drug_name}' records", unit="record"):
        chunk = flatten_record(record, drug_name)
        if chunk:
            chunks.append(chunk)
        else:
            dropped += 1

    print(f"\nFlattened : {len(chunks)} chunks")
    print(f"Dropped   : {dropped} records")
    if records:
        print(f"Drop rate : {dropped / len(records) * 100:.1f}%")

    return chunks



def save_chunks(
    chunks:    list[AdverseEventChunk],
    drug_name: str,
    out_dir:   str = "data/chunks",
) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    safe_name = drug_name.lower().replace(" ", "-")
    out_path  = Path(out_dir) / f"{safe_name}_chunks.json"

    with open(out_path, "w") as f:
        json.dump([c.model_dump() for c in chunks], f, indent=2)

    print(f"Saved → {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path



def inspect_chunks(chunks: list[AdverseEventChunk], sample_size: int = 5) -> None:
    total         = len(chunks)
    serious_count = sum(1 for c in chunks if c.serious)
    has_age       = sum(1 for c in chunks if c.patient_age)
    has_sex       = sum(1 for c in chunks if c.patient_sex)

    print(f"\n{'='*60}")
    print(f"CHUNK INSPECTION — {total} total chunks")
    print(f"{'='*60}")

    if total:
        print(f"  Serious events : {serious_count} ({serious_count/total*100:.1f}%)")
        print(f"  Has age        : {has_age} ({has_age/total*100:.1f}%)")
        print(f"  Has sex        : {has_sex} ({has_sex/total*100:.1f}%)")

    print(f"\nSAMPLE CHUNK TEXTS (what gets embedded):")
    print(f"{'─'*60}")

    for i, chunk in enumerate(chunks[:sample_size]):
        print(f"\n[Chunk {i+1}]")
        print(f"  report_id  : {chunk.report_id}")
        print(f"  serious    : {chunk.serious}")
        print(f"  outcomes   : {chunk.outcome}")
        print(f"  reactions  : {chunk.reactions[:4]}")
        print(f"  chunk_text :\n    {chunk.chunk_text}")


if __name__ == "__main__":
    drug     = "ibuprofen"
    raw_path = Path(f"data/raw/{drug}_raw.json")

    print(f"Loading raw data from {raw_path}...")
    with open(raw_path) as f:
        records = json.load(f)

    chunks = flatten_all(records, drug)
    save_chunks(chunks, drug)
    inspect_chunks(chunks, sample_size=5)