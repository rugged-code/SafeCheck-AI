import json 
from openai import OpenAI
from pydantic import BaseModel, field_validator
from src.config import GEMINI_API_KEY
from src.retriever import retrieve

client = OpenAI(base_url = "https://generativelanguage.googleapis.com/v1beta/openai/", api_key = GEMINI_API_KEY )
GEMINI_MODEL = "gemini-3.5-flash-lite"


class Finding(BaseModel):
    claim:      str
    report_ids: list[str]
    confidence: float

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_valid(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return round(v, 2)


class SafetyBrief(BaseModel):
    drug_name: str
    summary:   str
    findings:  list[Finding]

def build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks):
        block = (
            f"[Source {i+1}]\n"
            f"Report ID: {chunk['report_id']}\n"
            f"Text: {chunk['chunk_text']}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    return f"""You are a drug safety analyst.
A user has asked: "{query}"

Below are {len(chunks)} adverse event reports retrieved from the FDA database.
Use ONLY these reports to answer. Do not use any outside knowledge.

{context}

Return your response as a JSON object with this exact structure:
{{
  "drug_name": "<drug name>",
  "summary": "<2-3 sentence overall summary of what the reports show>",
  "findings": [
    {{
      "claim": "<one specific finding>",
      "report_ids": ["<report ID>", "<report ID>"],
      "confidence": <number between 0.0 and 1.0>
    }}
  ]
}}

RULES:
- Every claim MUST cite at least one report ID from the sources above
- Only use report IDs that actually appear in the sources above
- Do not invent report IDs
- Return only the JSON object, no extra text
"""


def call_gemini(prompt: str) ->str:
    response = client.chat.completions.create(
        model    = GEMINI_MODEL,
        messages = [
            {
                "role":    "system",
                "content": "You are a drug safety analyst. Always respond with valid JSON only."
            },
            {
                "role":    "user",
                "content": prompt
            }
        ],
        temperature = 0.1    # low temperature = more consistent, less creative output
    )   
    return response.choices[0].message.content

def parse_response(raw_text: str) -> SafetyBrief | None:
    # Strip markdown code fences if Gemini wraps the JSON
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        data  = json.loads(cleaned)
        brief = SafetyBrief(**data)
        return brief
    except Exception as e:
        print(f"Failed to parse Gemini response: {e}")
        print(f"Raw response was:\n{raw_text}")
        return None

def check_citations(brief: SafetyBrief, retrieved_chunks: list[dict]) -> dict:
    valid_ids    = {chunk["report_id"] for chunk in retrieved_chunks}
    all_cited    = []
    hallucinated = []

    for finding in brief.findings:
        for report_id in finding.report_ids:
            all_cited.append(report_id)
            if report_id not in valid_ids:
                hallucinated.append(report_id)

    return {
        "total_citations":    len(all_cited),
        "valid_citations":    len(all_cited) - len(hallucinated),
        "hallucinated_ids":   hallucinated,
        "hallucination_free": len(hallucinated) == 0
    }

def display_brief(brief: SafetyBrief, citation_report: dict) -> None:
    print("\n" + "=" * 60)
    print(f"SAFECHECK AI — SAFETY BRIEF")
    print(f"Drug: {brief.drug_name.upper()}")
    print("=" * 60)

    print(f"\nSUMMARY\n{brief.summary}")

    print(f"\nFINDINGS")
    print("-" * 60)
    for i, finding in enumerate(brief.findings):
        ids_str = ", ".join(finding.report_ids)
        print(f"\n{i+1}. {finding.claim}")
        print(f"   Report IDs : {ids_str}")
        print(f"   Confidence : {finding.confidence}")

    print(f"\nCITATION CHECK")
    print("-" * 60)
    print(f"  Total citations    : {citation_report['total_citations']}")
    print(f"  Valid citations    : {citation_report['valid_citations']}")
    print(f"  Hallucinated IDs   : {citation_report['hallucinated_ids'] or 'None'}")
    print(f"  Hallucination free : {citation_report['hallucination_free']}")
    print("=" * 60)


def generate(query: str, drug_name: str, serious_only: bool = False) -> dict | None:
    chunks = retrieve(query, drug_name, serious_only)

    if not chunks:
        print("No chunks retrieved — cannot generate brief")
        return None

    print("\nGenerating safety brief with Gemini...")
    prompt   = build_prompt(query, chunks)
    raw_text = call_gemini(prompt)

    brief = parse_response(raw_text)
    if not brief:
        return None

    citation_report = check_citations(brief, chunks)
    display_brief(brief, citation_report)

    return {
        "brief":            brief,
        "citation_report":  citation_report,
        "retrieved_chunks": chunks
    }


if __name__ == "__main__":
    generate(
        query        = "What are the most serious adverse reactions reported for ibuprofen?",
        drug_name    = "ibuprofen",
        serious_only = True
    )