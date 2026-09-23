import streamlit as st
from src.ingest import fetch_adverse_events, save_raw
from src.flatten import flatten_all, save_chunks
from src.embed_and_store import embed_and_store
from src.generate import generate
from qdrant_client import QdrantClient
from src.config import QDRANT_URL, QDRANT_API_KEY

# Page config 

st.set_page_config(
    page_title = "SafeCheck AI",
    page_icon  = "💊",
    layout     = "centered"
)

st.title("💊 SafeCheck AI")
st.caption("Drug adverse event analysis powered by FDA data + RAG")


#  Helper: check if drug is already indexed 

def is_drug_indexed(drug_name: str) -> bool:
    """
    Checks whether a drug already has vectors in Qdrant.
    If yes, we skip ingestion and go straight to retrieval.
    """
    try:
        client  = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        results = client.scroll(
            collection_name = "safecheck_adverse_events",
            scroll_filter   = {
                "must": [
                    {"key": "drug_name", "match": {"value": drug_name.lower()}}
                ]
            },
            limit        = 1,
            with_payload = False,
            with_vectors = False
        )
        return len(results[0]) > 0
    except Exception:
        return False


# Helper: run full ingestion pipeline for a new drug 

def run_ingestion_pipeline(drug_name: str, max_reports: int) -> int:
    """
    Runs ingest → flatten → embed → store for a drug not yet indexed.
    Returns number of chunks stored.
    """
    with st.status(f"Fetching {drug_name} reports from openFDA...", expanded=True) as status:

        # Stage 1: fetch
        st.write("📥 Fetching adverse event reports...")
        records = fetch_adverse_events(drug_name, max_reports=max_reports)
        save_raw(records, drug_name)
        st.write(f"✅ Fetched {len(records)} records")

        # Stage 2: flatten
        st.write("🔧 Flattening records into chunks...")
        chunks = flatten_all(records, drug_name)
        save_chunks(chunks, drug_name)
        st.write(f"✅ Created {len(chunks)} chunks")

        # Stage 4+5: embed and store
        st.write("🔢 Embedding and storing in Qdrant...")
        embed_and_store(drug_name)
        st.write(f"✅ Stored {len(chunks)} vectors in Qdrant")

        status.update(label="Data ready!", state="complete")

    return len(chunks)


# Sidebar settings 

with st.sidebar:
    st.header("⚙️ Settings")

    max_reports = st.slider(
        label   = "Max reports to fetch (new drugs only)",
        min_value = 100,
        max_value = 2000,
        value   = 500,
        step    = 100,
        help    = "Only used when fetching a drug for the first time"
    )

    serious_only = st.toggle(
        label = "Serious events only",
        value = True,
        help  = "Filter to reports marked as serious (hospitalization, death, etc.)"
    )

    st.divider()
    st.caption("SafeCheck AI uses FDA adverse event data. Not a substitute for medical advice.")


# Main input 

col1, col2 = st.columns([2, 1])

with col1:
    drug_name = st.text_input(
        label       = "Drug name",
        placeholder = "e.g. ibuprofen, metformin, aspirin",
    ).strip().lower()

with col2:
    st.write("")   # spacing
    st.write("")
    check_index = st.button("🔍 Check", use_container_width=True)

query = st.text_area(
    label       = "Your question",
    placeholder = "e.g. What are the most serious reactions reported for this drug?",
    height      = 100
)

run_button = st.button("Generate Safety Brief", type="primary", use_container_width=True)


# Check if drug is indexed ──────────────────────────────────────────────────
if check_index and drug_name:
    if is_drug_indexed(drug_name):
        st.success(f"✅ {drug_name.capitalize()} is already indexed — no fetching needed")
    else:
        st.warning(f"⚠️ {drug_name.capitalize()} not indexed yet — will fetch on first query")


# Run the full pipeline 

if run_button:
    if not drug_name:
        st.error("Please enter a drug name")
        st.stop()

    if not query:
        st.error("Please enter a question")
        st.stop()

    if not is_drug_indexed(drug_name):
        st.info(f"{drug_name.capitalize()} not found in index — fetching from openFDA now...")
        count = run_ingestion_pipeline(drug_name, max_reports)
        if count == 0:
            st.error("No chunks were created — drug may not exist in openFDA or all records were filtered out")
            st.stop()

    
    with st.spinner("Retrieving relevant reports and generating brief..."):
        result = generate(
            query        = query,
            drug_name    = drug_name,
            serious_only = serious_only
        )

    if not result:
        st.error("Failed to generate brief — check your API keys and try again")
        st.stop()

    brief           = result["brief"]
    citation_report = result["citation_report"]
    chunks          = result["retrieved_chunks"]

    #  Display safety brief 

    st.divider()
    st.subheader(f"Safety Brief — {brief.drug_name.upper()}")
    st.write(brief.summary)

    st.subheader("Findings")
    for i, finding in enumerate(brief.findings):
        with st.expander(f"{i+1}. {finding.claim}", expanded=True):
            st.write(f"**Report IDs:** {', '.join(finding.report_ids)}")
            st.write(f"**Confidence:** {finding.confidence}")

    #  Citation check 

    st.subheader("Citation Verification")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Citations", citation_report["total_citations"])
    with col2:
        st.metric("Valid Citations", citation_report["valid_citations"])
    with col3:
        halluc = len(citation_report["hallucinated_ids"])
        st.metric("Hallucinated", halluc, delta=None)

    if citation_report["hallucination_free"]:
        st.success("✅ All citations verified — no hallucinated report IDs")
    else:
        st.error(f"⚠️ Hallucinated IDs detected: {citation_report['hallucinated_ids']}")

    #  Retrieved chunks (expandable) 

    with st.expander("View retrieved source chunks"):
        for chunk in chunks:
            st.markdown(f"**Report ID:** `{chunk['report_id']}`")
            st.write(chunk["chunk_text"])
            st.write(f"Rerank score: `{chunk.get('rerank_score', 'N/A')}`")
            st.divider()