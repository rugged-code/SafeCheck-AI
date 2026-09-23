from dotenv import load_dotenv
import os

load_dotenv()

OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY")
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
JINA_API_KEY    = os.getenv("JINA_API_KEY")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

OPENFDA_BASE_URL = "https://api.fda.gov/drug/event.json"
MAX_REPORTS      = 1000   # Can raise later
BATCH_SIZE       = 100    # openFDA hard cap per request