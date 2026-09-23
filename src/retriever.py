from tqdm import tqdm
import requests
from tenacity import retry, wait_exponential, stop_after_attempt
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, QueryRequest
from src.config import JINA_API_KEY, QDRANT_API_KEY, QDRANT_URL

COLLECTION_NAME = "safecheck_adverse_events"
EMBEDDING_MODEL = "jina-embeddings-v3"
RERANKER_MODEL = "jina-reranker-v2-base-multilingual"
top_k = 20
top_n = 5


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@retry(
    wait = wait_exponential(multiplier=1, min=2, max=32),
    stop=stop_after_attempt(5),
    reraise=True
)

def embed_query(query:str)->list[float]:

    response= requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-type" : "application/json"
        },
        json={
            "model" : EMBEDDING_MODEL,
            "input" : [query],
            "task"  : "retrieval.query"
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def search_qdrant(
    client:       QdrantClient,
    query_vector: list[float],
    drug_name:    str,
    serious_only: bool = False
) -> list[dict]:

    must_conditions = [
        FieldCondition(
            key="drug_name",
            match=MatchValue(value=drug_name.lower())
        )
    ]

    if serious_only:
        must_conditions.append(
            FieldCondition(
                key="serious",
                match=MatchValue(value=True)
            )
        )

    results = client.query_points(
        collection_name = COLLECTION_NAME,
        query           = query_vector,
        query_filter    = Filter(must=must_conditions),
        limit           = top_k,
        with_payload    = True
    )

    chunks = []
    for result in results.points:       # .points instead of iterating directly
        chunk = result.payload
        chunk["score"] = result.score
        chunks.append(chunk)

    return chunks

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=32),
    stop=stop_after_attempt(5),
    reraise=True
)

def rerank_chunks(query: str, chunks: list[dict])->list[dict]:

    if not chunks:
        return []

    documents = [chunk["chunk_text"] for chunk in chunks]

    response = requests.post("https://api.jina.ai/v1/rerank",
                headers= {
                    "Authorization" : f"Bearer {JINA_API_KEY}",
                    "Content-Type" : "application/json"
                },
                json = {
                    "model" : RERANKER_MODEL,
                    "query" : query,
                    "documents" : documents,
                    "top_n" : top_n
                },
                timeout = 30

            )
    response.raise_for_status()
    results = response.json()["results"]

    reranked = []

    for result in results:
        chunk = chunks[result["index"]].copy()
        chunk["rerank_score"] = result["relevance_score"]
        reranked.append(chunk)

    return reranked


def retrieve(query:str, drug_name: str, serious_only: bool=False)->list[dict]:
    print(f"\nQuery: {query}")
    print(f"Drug: {drug_name} | Serious only: {serious_only}")

    client = get_qdrant_client()
    query_vector = embed_query(query)

    print(f"Searching Qdrant fot top {top_k} chunks...")
    chunks = search_qdrant(client, query_vector, drug_name, serious_only)
    print(f"Found {len(chunks)} from Qdrant")

    if not chunks:
        print("No chunks found- try a different drug name")
        return[]

    print(f"Reranking to top {top_n}...")
    top_chunks = rerank_chunks(query, chunks)
 
    print(f"\nTop {len(top_chunks)} chunks after reranking:")
    for i, chunk in enumerate(top_chunks):
        print(f"[{i+1}] Report {chunk['report_id']}"
              f"| rerank score: {chunk['rerank_score']:.3f}"
              f"| {chunk['chunk_text'][:80]}...")

    return top_chunks


if __name__ == "__main__":
    results = retrieve(query = "serious gastrointestinal bleeding reaction from ibuprofen",
                      drug_name = "ibuprofen",
                      serious_only=True
                      )

    print(f"\nFinal retrieved chunks: {len(results)}")
    for chunk in results:
        print(f"\n  Report ID : {chunk['report_id']}")
        print(f"  Score     : {chunk['rerank_score']:.3f}")
        print(f"  Text      : {chunk['chunk_text']}")
    


