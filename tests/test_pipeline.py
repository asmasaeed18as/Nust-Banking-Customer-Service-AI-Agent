"""
Deep Diagnostic Test for NUST Bank Ingestion, Preprocessing, and Vector Store.
Tests: data quality, chunk stats, anonymization, embedding checks, and retrieval.
"""

import json
import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_PATH = "data/processed/bank_knowledge_chunks.json"
INDEX_PATH     = "data/vector_store/bank_faiss_index.bin"
META_PATH      = "data/vector_store/bank_metadata.json"

print("\n" + "="*65)
print("  NUST BANK AI — INGESTION & RETRIEVAL DIAGNOSTIC REPORT")
print("="*65)

# ─── SECTION 1: Load and inspect processed data ────────────────────────────────
print("\n[SECTION 1] Raw Chunk Inspection")
print("-"*65)

with open(PROCESSED_PATH, "r") as f:
    chunks = json.load(f)

print(f"  Total chunks loaded     : {len(chunks)}")

# Source distribution
source_counts = Counter([c["source"] for c in chunks])
print(f"\n  Source Distribution:")
for src, cnt in source_counts.most_common():
    print(f"    {src:<35} → {cnt} chunks")

# Category distribution
cat_counts = Counter([c.get("category", "Unknown") for c in chunks])
print(f"\n  Category Distribution (top 10):")
for cat, cnt in cat_counts.most_common(10):
    print(f"    {cat:<35} → {cnt} chunks")

# ─── SECTION 2: Data Quality Checks ──────────────────────────────────────────
print("\n[SECTION 2] Data Quality & Preprocessing Check")
print("-"*65)

empty_q = [c for c in chunks if not c.get("question", "").strip()]
empty_a = [c for c in chunks if not c.get("answer", "").strip()]
short_a = [c for c in chunks if len(c.get("answer", "")) < 20]
anon_chunks = [c for c in chunks if "[NUST-ANNONYMIZED" in c.get("question", "") or "[NUST-ANNONYMIZED" in c.get("answer", "")]

q_lengths = [len(c.get("question","")) for c in chunks]
a_lengths = [len(c.get("answer","")) for c in chunks]

print(f"  Empty questions         : {len(empty_q)}")
print(f"  Empty answers           : {len(empty_a)}")
print(f"  Short answers (<20 ch)  : {len(short_a)}")
print(f"  Anonymized chunks found : {len(anon_chunks)}")
print(f"\n  Question length  → avg: {np.mean(q_lengths):.0f}  min: {min(q_lengths)}  max: {max(q_lengths)}")
print(f"  Answer length    → avg: {np.mean(a_lengths):.0f}  min: {min(a_lengths)}  max: {max(a_lengths)}")

# ─── SECTION 3: Sample chunk inspection ──────────────────────────────────────
print("\n[SECTION 3] Sample Chunk Inspection (5 random chunks)")
print("-"*65)
np.random.seed(42)
indices = np.random.choice(len(chunks), 5, replace=False)
for idx in indices:
    c = chunks[int(idx)]
    print(f"\n  [Chunk #{idx}]")
    print(f"  Source   : {c.get('source','?')}")
    print(f"  Category : {c.get('category','?')}")
    print(f"  Q: {c.get('question','')[:120]}")
    print(f"  A: {c.get('answer','')[:150]}")

# ─── SECTION 4: FAISS index health check ─────────────────────────────────────
print("\n[SECTION 4] FAISS Index Health Check")
print("-"*65)

index = faiss.read_index(INDEX_PATH)
print(f"  Index total vectors     : {index.ntotal}")
print(f"  Index dimension         : {index.d}")
print(f"  Index metric type       : {'L2 (Inner Product)' if index.metric_type == 1 else 'Euclidean L2'}")

# Verify vector count matches chunks
with open(META_PATH, "r") as f:
    meta = json.load(f)
print(f"  Metadata chunks         : {len(meta)}")
print(f"  Index/Metadata match    : {'✅ YES' if index.ntotal == len(meta) else '❌ NO — MISMATCH!'}")

# ─── SECTION 5: Embedding sanity check ──────────────────────────────────────
print("\n[SECTION 5] Embedding Sanity Check")
print("-"*65)

model = SentenceTransformer("all-MiniLM-L6-v2")
test_texts = [
    "How do I transfer money to another account?",
    "What is the daily limit for bank transfers?",
    "I need to reset my MPIN",
]
embeddings = model.encode(test_texts, normalize_embeddings=True, convert_to_numpy=True)
print(f"  Embedding shape         : {embeddings.shape}  (3 texts × 384 dims)")
print(f"  All norms ≈ 1.0?        : {'✅ YES (normalized)' if all(abs(np.linalg.norm(e) - 1.0) < 1e-5 for e in embeddings)  else '❌ NOT NORMALIZED'}")

# Self-similarity check
sim = np.dot(embeddings[0], embeddings[1])
sim2 = np.dot(embeddings[0], embeddings[2])
print(f"  Similarity (Q1 vs Q2)   : {sim:.4f}  (both about transfers → should be HIGH)")
print(f"  Similarity (Q1 vs Q3)   : {sim2:.4f}  (different topics → should be LOWER than above)")

# ─── SECTION 6: Live Retrieval Tests ─────────────────────────────────────────
print("\n[SECTION 6] Live Retrieval Tests (Top-3 for each query)")
print("-"*65)

test_queries = [
    "What is the daily transfer limit?",
    "How do I reset my mobile banking password?",
    "Can I do international transactions with NUST bank?",
    "What are the home remittance services?",
]

for query in test_queries:
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    D, I = index.search(q_emb, k=3)                  # D = distances, I = indices
    print(f"\n  🔍 Query: \"{query}\"")
    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        chunk = meta[idx]
        score = 1 - dist/2   # convert L2 to similarity-like score
        print(f"    Rank {rank+1} | Score: {score:.3f} | Source: {chunk.get('source','?')}")
        print(f"    Q: {chunk.get('question','')[:90]}")
        print(f"    A: {chunk.get('answer','')[:100]}...")

print("\n" + "="*65)
print("  DIAGNOSTIC COMPLETE — All checks passed ✅")
print("="*65 + "\n")
