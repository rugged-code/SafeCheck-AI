from tqdm import tqdm
import time
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt
import json
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance
from src.config import JINA_API_KEY, QDRANT_API_KEY, QDRANT_URL
import requests


COLLECTION_NAME = "safecheck_adverse_events"
BATCH_SIZE = 50
EMBEDDING_MODEL = "jina-embeddings-v3"
VECTOR_SIZE = 1024

def get_qdrant_client() ->QdrantClient:
    client = QdrantClient(url = QDRANT_URL, api_key=QDRANT_API_KEY, timeout = 30)

    print("Connecting to Qdrant cloud...")
    return client   

def create_collection_if_not_exists(client: QdrantClient) -> None:

    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME} already exists, skipping creation.")
        return 

    client.create_collection(
       collection_name=COLLECTION_NAME,
       vectors_config=VectorParams(
           size=VECTOR_SIZE,
           distance=Distance.COSINE
       )
    )   

    print(f"Created collection {COLLECTION_NAME}")
    

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=32),
    stop=stop_after_attempt(5), 
    reraise=True
)

def embed_batch(texts: list[str]) -> list[list[float]]:

    JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"
    response = requests.post(JINA_EMBEDDING_URL,
            headers={
                "Authorization" : f"Bearer {JINA_API_KEY}",
                "Content-Type" : "application/json"
            },
            json={
                "model" : EMBEDDING_MODEL,
                "input" : texts,
                "task" : "retrieval.passage"
            }
    )   
    response.raise_for_status()
    data = response.json()

    return [item["embedding"] for item in data["data"]]

def upsert_batch(client: QdrantClient, chunks: list[dict], vectors: list[list[float]]) ->None:
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point = PointStruct(
            id      = abs(hash(chunk["report_id"])) % (2**63),  # Qdrant needs integer ID
            vector  = vector,
            payload = {                        
                "report_id":   chunk["report_id"],
                "drug_name":   chunk["drug_name"],
                "serious":     chunk["serious"],
                "outcome":     chunk["outcome"],
                "reactions":   chunk["reactions"],
                "patient_age": chunk["patient_age"],
                "patient_sex": chunk["patient_sex"],
                "receive_date":chunk["receive_date"],
                "chunk_text":  chunk["chunk_text"],  
            }
        )
        points.append(point)
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def embed_and_store(drug_name: str) ->None:
    safe_name = drug_name.lower().replace(" " ,"-")
    chunks_path = f"data/chunks/{safe_name}_chunks.json"

    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path) as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    client = get_qdrant_client()
    create_collection_if_not_exists(client)

    total_batches = (len(chunks) + BATCH_SIZE - 1) //  BATCH_SIZE
    total_stored = 0

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding + storing", total=total_batches, unit = "batch"):
        batch = chunks[i: i + BATCH_SIZE]

        texts = [chunk["chunk_text"] for chunk in batch]

        vector = embed_batch(texts)
        upsert_batch(client, batch, vector)

        total_stored += len(batch)
        time.sleep(0.5)

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"\nDone. Total vector in '{COLLECTION_NAME}': {count}")

if __name__ == "__main__":
    embed_and_store(drug_name = "ibuprofen")