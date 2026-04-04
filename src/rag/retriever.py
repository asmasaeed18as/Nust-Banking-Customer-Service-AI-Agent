"""
src/rag/retriever.py
────────────────────
FAISS-based retriever for NUST Bank knowledge chunks.

Responsibilities:
  - Load the FAISS index and metadata from disk (once at startup)
  - Embed an incoming user query using the same model used at indexing time
  - Search the index for the top-k most relevant chunks
  - Return structured results with score and source metadata
"""

import json
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from typing import List, Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import (
    FAISS_INDEX_PATH, METADATA_PATH,
    EMBEDDING_MODEL, TOP_K_RESULTS, MIN_SCORE_THRESHOLD, DEVICE
)


class NustBankRetriever:
    """
    Loads the FAISS vector index once at initialisation and provides
    a `retrieve(query)` method that returns ranked, scored knowledge chunks.
    """

    def __init__(self):
        logger.info("[Retriever] Initialising NUST Bank knowledge retriever...")

        # ── Load embedding model ──────────────────────────────────────────────
        logger.debug(f"[Retriever] Loading embedding model: {EMBEDDING_MODEL} on {DEVICE}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE, trust_remote_code=True)
        logger.success(f"[Retriever] Embedding model loaded on {DEVICE}.")

        # ── Load FAISS index ──────────────────────────────────────────────────
        if not FAISS_INDEX_PATH.exists():
            logger.error(f"[Retriever] FAISS index not found at {FAISS_INDEX_PATH}")
            raise FileNotFoundError(
                f"Run 'python src/vector_store.py' first to build the index."
            )
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        logger.success(f"[Retriever] FAISS index loaded — {self.index.ntotal} vectors, dim={self.index.d}")

        # ── Load metadata ─────────────────────────────────────────────────────
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.success(f"[Retriever] Metadata loaded — {len(self.metadata)} chunks.")

        # ── Initialize BM25 (Hybrid Keyword Search) ───────────────────────────
        from rank_bm25 import BM25Okapi
        # Create corpus of all chunks to enable keyword matching
        corpus = [
            (chunk.get("question", "") + " " + chunk.get("answer", "")).lower().split()
            for chunk in self.metadata
        ]
        self.bm25 = BM25Okapi(corpus)
        logger.success(f"[Retriever] BM25 keyword index loaded — ready for Hybrid Search.")

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Embed `query`, search FAISS and BM25, and fuse using Reciprocal Rank Fusion (RRF).
        """
        if not query.strip():
            logger.warning("[Retriever] Empty query received — returning empty results.")
            return []

        logger.info(f"[Retriever] Query: '{query}'")
        t0 = time.perf_counter()

        # ── 1. Semantic Search (FAISS) ────────────────────────────────────────
        query_vec = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        embed_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        distances, indices = self.index.search(query_vec, k=top_k * 2) # Fetch extra for pooling
        
        # Calculate standard semantic scores for tracking threshold compliance
        faiss_scores = {idx: round(float(1 - dist / 2), 4) for idx, dist in zip(indices[0], distances[0])}

        # ── 2. Keyword Search (BM25) ──────────────────────────────────────────
        tokenized_query = query.lower().split()
        bm25_raw_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_raw_scores)[::-1][:top_k * 2]
        
        search_ms = (time.perf_counter() - t1) * 1000
        logger.debug(f"[Retriever] Hybrid FAISS + BM25 search done in {search_ms:.2f} ms")

        # ── 3. Reciprocal Rank Fusion (RRF) ───────────────────────────────────
        RRF_K = 60
        rrf_scores = {}
        
        # Add FAISS Ranks
        for rank, idx in enumerate(indices[0]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
            
        # Add BM25 Ranks
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        # ── 4. Build results (Sorted by RRF Score) ────────────────────────────
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        
        for rank, (idx, rrf_score) in enumerate(sorted_candidates[:top_k]):
            chunk = self.metadata[idx]
            
            semantic_score = faiss_scores.get(idx, 0.0)
            bm25_score = bm25_raw_scores[idx]
            
            # Threshold fallback protection (drop bad chunks)
            if semantic_score < MIN_SCORE_THRESHOLD:
                # If semantic fails explicitly but BM25 is very confident, keep it
                if bm25_score < 3.0: 
                    continue
                    
            result = {
                "rank"    : rank + 1,
                "score"   : semantic_score, # Passed strictly for UI
                "rrf_score": round(rrf_score, 4),
                "question": chunk.get("question", ""),
                "answer"  : chunk.get("answer", ""),
                "source"  : chunk.get("source", "Unknown"),
                "category": chunk.get("category", "General"),
            }
            results.append(result)

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[Retriever] RRF Hybrid Search retrieved {len(results)} chunks in {total_ms:.1f} ms"
        )
        return results
