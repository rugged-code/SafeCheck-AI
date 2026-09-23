from pathlib import Path
from src.config import OPENFDA_API_KEY, OPENFDA_BASE_URL, MAX_REPORTS, BATCH_SIZE
import json
import time
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from tqdm import tqdm
import requests

@retry(
    wait= wait_exponential(multiplier=1, min=2, max=32),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
    reraise=True
)

def fetch_page(drug_name: str, skip: int, limit: int = 100)->dict:
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "limit": limit,
        "skip": skip
    }
    if OPENFDA_API_KEY:
        params["api_key"] = OPENFDA_API_KEY

    response = requests.get(OPENFDA_BASE_URL, params=params, timeout=15)

    if response.status_code == 404:
        return {}

    if response.status_code == 429:
        raise requests.exceptions.ConnectionError("Rate limited - 429")

    response.raise_for_status()
    return response.json()




def fetch_adverse_events(drug_name: str, max_reports : int = MAX_REPORTS) -> list[dict]:
    all_records = []
    first_page = fetch_page(drug_name, skip=0, limit=1)
    if not first_page:
        print(f"No records found for the drug name: {drug_name}")
        return []

    total_available = first_page["meta"]["results"]["total"]
    total_to_fetch  = min(total_available, max_reports)

    print(f"\nDrug : '{drug_name}' | Total available: {total_available:,} | Fetching : {total_to_fetch:,}")

    offset = range(0, total_to_fetch, BATCH_SIZE)

    for skip in tqdm(offset, desc=f"Fetching '{drug_name}'", unit="batch"):
        limit = min(total_to_fetch - skip, BATCH_SIZE)
        page = fetch_page(drug_name, skip=skip, limit=limit)

        if not page or "results" not in page:
            print(f"\nWarning! Empty page found at skip={skip}, stopping early.")
            break
        all_records.extend(page["results"])
        time.sleep(0.25)

    print(f"Fetched {len(all_records)} records for the drug: '{drug_name}'")
    return all_records


    

def save_raw(records: list[dict], drug_name : str, out_dir: str = "data/raw") -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok = True)
    safe_name = drug_name.lower().replace(" ","_")
    out_path = Path(out_dir)/f"{safe_name}_raw.json"

    with open(out_path, "w") as f:
        json.dump(records, f, indent = 2)

    print(f"Saved raw data in {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def inspect_raw(records: list[dict], sample_size: int = 5) -> None:
    """
    Print a structured inspection of raw records so you can see
    what fields are actually present vs absent before designing the schema.
    """
    print(f"\n{'='*60}")
    print(f"INSPECTION: {len(records)} records, showing {sample_size} samples")
    print(f"{'='*60}")

    
    field_stats = {
        "has_narrative":       0,
        "has_patient_age":     0,
        "has_patient_sex":     0,
        "has_reactions":       0,
        "has_seriousness":     0,
        "has_receivedate":     0,
        "multi_drug_report":   0,
    }

    for r in records:
        patient = r.get("patient", {})
        if r.get("narrativeincludeclinical"):                   field_stats["has_narrative"]     += 1
        if patient.get("patientagegroup"):                       field_stats["has_patient_age"]   += 1
        if patient.get("patientsex"):                            field_stats["has_patient_sex"]   += 1
        if patient.get("reaction"):                              field_stats["has_reactions"]     += 1
        if r.get("serious"):                                     field_stats["has_seriousness"]   += 1
        if r.get("receivedate"):                                 field_stats["has_receivedate"]   += 1
        if len(patient.get("drug", [])) > 1:                    field_stats["multi_drug_report"] += 1

    print("\nField presence across ALL records:")
    for field, count in field_stats.items():
        pct = (count / len(records)) * 100
        print(f"  {field:<25} {count:>5} / {len(records)}  ({pct:.1f}%)")

    
    print(f"\n{'─'*60}")
    print("SAMPLE RECORDS (raw):")
    print(f"{'─'*60}")
    for i, record in enumerate(records[:sample_size]):
        patient = record.get("patient", {})
        print(f"\n[Record {i+1}]")
        print(f"  safetyreportid    : {record.get('safetyreportid', 'MISSING')}")
        print(f"  receivedate       : {record.get('receivedate', 'MISSING')}")
        print(f"  serious           : {record.get('serious', 'MISSING')}")
        print(f"  seriousnessdeath  : {record.get('seriousnessdeath', 'MISSING')}")
        print(f"  seriousnesshospitalization: {record.get('seriousnesshospitalization', 'MISSING')}")
        print(f"  patientagegroup   : {patient.get('patientagegroup', 'MISSING')}")
        print(f"  patientsex        : {patient.get('patientsex', 'MISSING')}")

        reactions = [rx.get("reactionmeddrapt", "?") for rx in patient.get("reaction", [])]
        print(f"  reactions         : {reactions}")

        drugs = [d.get("medicinalproduct", "?") for d in patient.get("drug", [])]
        print(f"  drugs_in_report   : {drugs}")

        narrative = record.get("narrativeincludeclinical", None)
        if narrative:
            print(f"  narrative         : {narrative[:200]}...")
        else:
            print(f"  narrative         : *** ABSENT ***")



if __name__ == "__main__":

    drug = input("Enter drug name: ")   

    records = fetch_adverse_events(drug, max_reports=500)
    save_raw(records, drug)
    inspect_raw(records, sample_size=5)
