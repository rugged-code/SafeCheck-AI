import time
from src.retriever import get_qdrant_client, embed_query, search_qdrant, rerank_chunks
from src.generate import generate

# ── Ground Truth Test Set 
EVAL_DATASET = [
    {
        "query": "Stevens-Johnson syndrome skin reactions and blisters from ibuprofen",
        "drug_name": "ibuprofen",
        "serious_only": True,
        # Reports 10004208, 10010880, 10014471, 10033034 have SJS, blisters, skin exfoliation
        "expected_ids": ["10004208", "10010880", "10014471", "10033034"],
        "expected_claim_keywords": ["stevens-johnson", "blister", "skin"]
    },
    {
        "query": "Gastric ulcer perforation and gastrointestinal bleeding",
        "drug_name": "ibuprofen",
        "serious_only": True,
        # 10012655: Gastric ulcer perforation
        # 10025628: Gastric perforation
        # 10028598: Perforated ulcer
        # 10037054: Gastric fistula, Gastric ulcer
        # 10039609: Peptic ulcer perforation
        "expected_ids": ["10012655", "10025628", "10028598", "10037054", "10039609"],
        "expected_claim_keywords": ["ulcer", "perforation", "gastric"]
    },
    {
        "query": "Pulmonary haemorrhage and coughing up blood haemoptysis",
        "drug_name": "ibuprofen",
        "serious_only": True,
        # 10005044, 10006726, 10012463, 10012654, 10016161, 10019638
        "expected_ids": ["10005044", "10006726", "10012463", "10012654", "10016161", "10019638"],
        "expected_claim_keywords": ["pulmonary", "haemorrhage", "haemoptysis"]
    },
    {
        "query": "Acute renal failure and kidney impairment with drug interaction",
        "drug_name": "ibuprofen",
        "serious_only": True,
        # 10021222, 10027031, 10038999, 10039020, 10039295, 10039299
        "expected_ids": ["10021222", "10027031", "10038999", "10039020", "10039295", "10039299"],
        "expected_claim_keywords": ["renal", "kidney", "failure"]
    },
    {
        "query": "Severe angioedema swelling in infants and children",
        "drug_name": "ibuprofen",
        "serious_only": True,
        # Many reports: 10024924, 10024928, 10027857, 10027858, 10029384, etc.
        "expected_ids": ["10024924", "10024928", "10027857", "10027858", "10029384"],
        "expected_claim_keywords": ["angioedema", "swelling", "oedema"]
    },
    {
        "query": "Acute generalised exanthematous pustulosis rash and fever",
        "drug_name": "ibuprofen",
        "serious_only": True,
        # 10009321, 10026052, 10032214
        "expected_ids": ["10009321", "10026052", "10032214"],
        "expected_claim_keywords": ["pustulosis", "rash", "exanthematous"]
    },
    {
        "query": "Renal tubular acidosis and hypokalaemia from misuse or abuse",
        "drug_name": "ibuprofen",
        "serious_only": True,
        # 10008276, 10037251
        "expected_ids": ["10008276", "10037251"],
        "expected_claim_keywords": ["tubular acidosis", "hypokalaemia", "potassium"]
    },
    {
        "query": "Mild non-serious drug hypersensitivity skin reactions",
        "drug_name": "ibuprofen",
        "serious_only": False,
        # Non-serious: 10004874, 10015225, 10015719, 10016367, 10022660, 10024133, 10027529, 10031754, 10037783
        "expected_ids": ["10004874", "10015225", "10015719", "10016367", "10022660", "10024133", "10027529", "10031754", "10037783"],
        "expected_claim_keywords": ["hypersensitivity", "non-serious", "reaction"]
    }
]


# ── Metric helpers 

def compute_recall_precision(retrieved_ids: list[str], target_ids: list[str]):
    if not retrieved_ids or not target_ids:
        return 0.0, 0.0
    
    hits = 0
    for rid in retrieved_ids:
        if rid in target_ids:
            hits += 1

    precision = hits / len(retrieved_ids)
    recall = hits / len(target_ids)
    return recall, precision


# ── Run Evaluation 

def run_evaluation():
    print("\n" + "="*55)
    print("      SAFECHECK AI - RETRIEVAL & GENERATION EVAL      ")
    print("="*55)

    client = get_qdrant_client()
    
    pre_recalls = []
    pre_precisions = []
    post_recalls = []
    post_precisions = []
    
    hallucination_counts = []
    total_citations_made = []

    for idx, item in enumerate(EVAL_DATASET):
        q = item["query"]
        drug = item["drug_name"]
        serious = item["serious_only"]
        target_ids = item["expected_ids"]

        print(f"\n[{idx+1}/{len(EVAL_DATASET)}] Query: {q[:60]}...")

        vec = embed_query(q)
        qdrant_raw = search_qdrant(client, vec, drug, serious)
        pre_ids = [str(c["report_id"]).strip() for c in qdrant_raw]
        
        r_pre, p_pre = compute_recall_precision(pre_ids, target_ids)
        pre_recalls.append(r_pre)
        pre_precisions.append(p_pre)

        reranked = rerank_chunks(q, qdrant_raw)
        post_ids = [str(c["report_id"]).strip() for c in reranked]

        r_post, p_post = compute_recall_precision(post_ids, target_ids)
        post_recalls.append(r_post)
        post_precisions.append(p_post)

        dropped = set(pre_ids).intersection(set(target_ids)) - set(post_ids)
        drop_note = f"(Reranker dropped match: {dropped})" if dropped else ""

        print(f"  Pre-Rerank  (k=20): Recall={r_pre:.2f} | Precision={p_pre:.2f}")
        print(f"  Post-Rerank (k=5) : Recall={r_post:.2f} | Precision={p_post:.2f} {drop_note}")

        gen_res = generate(q, drug, serious)
        if gen_res and "citation_report" in gen_res:
            rep = gen_res["citation_report"]
            hallucinated = len(rep["hallucinated_ids"])
            total_cites = rep["total_citations"]

            hallucination_counts.append(hallucinated)
            total_citations_made.append(total_cites)
            print(f"  Generation citations: {total_cites} total, {hallucinated} hallucinated")
        else:
            print("  Generation skipped / empty output.")

        time.sleep(0.5)





    # ── Final Summary Table 
    print("\n" + "="*55)
    print("                   EVALUATION REPORT                  ")
    print("="*55)
    
    avg_pre_r = sum(pre_recalls) / len(pre_recalls) if pre_recalls else 0
    avg_pre_p = sum(pre_precisions) / len(pre_precisions) if pre_precisions else 0
    avg_post_r = sum(post_recalls) / len(post_recalls) if post_recalls else 0
    avg_post_p = sum(post_precisions) / len(post_precisions) if post_precisions else 0

    print(f"Total Test Cases Evaluated : {len(EVAL_DATASET)}")
    print("\nRetrieval Metrics:")
    print(f"  Pre-Rerank  Mean Recall@20    : {avg_pre_r:.3f}")
    print(f"  Pre-Rerank  Mean Precision@20 : {avg_pre_p:.3f}")
    print(f"  Post-Rerank Mean Recall@5     : {avg_post_r:.3f}")
    print(f"  Post-Rerank Mean Precision@5  : {avg_post_p:.3f}")

    total_h = sum(hallucination_counts)
    total_c = sum(total_citations_made)
    print("\nCitation & Hallucination Guard:")
    print(f"  Total Citations Generated     : {total_c}")
    print(f"  Hallucinated Report IDs       : {total_h}")
    hallucination_rate = (total_h / total_c * 100) if total_c > 0 else 0.0
    print(f"  Hallucination Rate            : {hallucination_rate:.1f}%")
    print("="*55 + "\n")


if __name__ == "__main__":
    run_evaluation()