"""
Benchmark retrieval quality: gte-large-en-v1.5 vs all-MiniLM-L6-v2
Runs all dummy prompts through both models and compares:
  - Top-1 score
  - Top-3 average score
  - Whether the retrieved source looks relevant
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sentence_transformers import SentenceTransformer
import faiss, numpy as np

QUERIES = [
    "What is the daily transfer limit for mobile banking at NUST Bank?",
    "How do I reset my MPIN if I forgot it?",
    "Can I use the NUST mobile banking app while I am traveling abroad?",
    "What documents do I need to open a NUST savings account?",
    "How do I add a new beneficiary for funds transfer?",
    "What is the NUST Bank Home Remittance service?",
    "Are there any charges for using the RAAST transfer system?",
    "How can I activate international transactions on my NUST debit card?",
    "What utility bills can I pay through the NUST mobile app?",
    "How do I enable biometric fingerprint login on the app?",
    "Does NUST Bank offer any car financing schemes?",
    "What is the minimum deposit required for a NUST savings account?",
    "How can I contact NUST Bank customer support?",
    "What is the difference between a current and savings account at NUST Bank?",
    "How do I report a lost or stolen NUST Bank debit card?",
]

METADATA_PATH  = "data/vector_store/bank_metadata.json"
INDEX_PATH     = "data/vector_store/bank_faiss_index.bin"

# ── Load rebuilt index (gte-large) ────────────────────────────────────────────
print("\n" + "="*70)
print(f"  Loading GTE-Large index from  {INDEX_PATH}")
print("="*70)

with open(METADATA_PATH) as f:
    metadata = json.load(f)

index = faiss.read_index(INDEX_PATH)
print(f"  Index vectors : {index.ntotal}")
print(f"  Dimension     : {index.d}")
print(f"  Metadata rows : {len(metadata)}")

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device        : {device}\n")

gte_model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", device=device, trust_remote_code=True)
mini_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def retrieve(model, index, metadata, query, top_k=5, dim_expected=None):
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    if dim_expected and vec.shape[1] != dim_expected:
        return []
    dists, idxs = index.search(vec, k=top_k)
    results = []
    for rank, (idx, dist) in enumerate(zip(idxs[0], dists[0])):
        score = round(float(1 - dist / 2), 4)
        chunk = metadata[idx]
        results.append({
            "rank": rank + 1,
            "score": score,
            "question": chunk.get("question", "")[:80],
            "source": chunk.get("source", "?"),
        })
    return results

# Build a separate FLAT index for MiniLM (384-dim) from scratch using same metadata
print("Building MiniLM reference index for comparison...")
t0 = time.perf_counter()
texts = [f"{c.get('question','')} {c.get('answer','')}" for c in metadata]
mini_vecs = mini_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64, show_progress_bar=True)
mini_index = faiss.IndexFlatIP(384)
mini_index.add(mini_vecs.astype(np.float32))
mini_build_s = time.perf_counter() - t0
print(f"MiniLM index built in {mini_build_s:.1f}s\n")

# ── Run benchmark ──────────────────────────────────────────────────────────────
print("="*70)
print(f"{'QUERY':<50} {'GTE-Top1':>9} {'Mini-Top1':>9} {'DELTA':>7}")
print("="*70)

gte_scores, mini_scores = [], []

for q in QUERIES:
    gte_res  = retrieve(gte_model,  index,      metadata, q, dim_expected=1024)
    mini_res = retrieve(mini_model, mini_index, metadata, q, dim_expected=384)

    gte_top1  = gte_res[0]["score"]  if gte_res  else 0.0
    mini_top1 = mini_res[0]["score"] if mini_res else 0.0
    delta     = gte_top1 - mini_top1

    gte_scores.append(gte_top1)
    mini_scores.append(mini_top1)

    label = "+" if delta > 0 else ""
    print(f"{q[:50]:<50} {gte_top1:>9.4f} {mini_top1:>9.4f} {label}{delta:>+.4f}")

print("="*70)
print(f"{'AVERAGE':<50} {sum(gte_scores)/len(gte_scores):>9.4f} {sum(mini_scores)/len(mini_scores):>9.4f} {(sum(gte_scores)-sum(mini_scores))/len(gte_scores):>+.4f}")
print("="*70)
print(f"\nGTE top-1 beat MiniLM on {sum(g>m for g,m in zip(gte_scores,mini_scores))}/{len(QUERIES)} queries")
print(f"Average improvement: +{(sum(gte_scores)-sum(mini_scores))/len(gte_scores):.4f} cosine similarity\n")

# ── Show top hits for first 3 queries ─────────────────────────────────────────
print("\n── Detailed top-3 results for first 3 queries ──────────────────────────")
for q in QUERIES[:3]:
    print(f"\nQuery: {q}")
    gte_res  = retrieve(gte_model,  index,      metadata, q, dim_expected=1024)
    mini_res = retrieve(mini_model, mini_index, metadata, q, dim_expected=384)
    print(f"  {'GTE-Large':<40} | {'MiniLM':<40}")
    print(f"  {'-'*40} | {'-'*40}")
    for i in range(3):
        g = gte_res[i]  if i < len(gte_res)  else {"score":0,"question":"—"}
        m = mini_res[i] if i < len(mini_res) else {"score":0,"question":"—"}
        print(f"  [{g['score']:.3f}] {g['question'][:36]:<36} | [{m['score']:.3f}] {m['question'][:36]:<36}")
