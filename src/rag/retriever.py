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

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Embed `query`, search FAISS, and return enriched chunk dictionaries.

        Applies a two-pass threshold:
          Pass 1 — MIN_SCORE_THRESHOLD (strict): preferred results
          Pass 2 — FALLBACK_THRESHOLD (soft): if pass 1 returns nothing,
                   retry rather than immediately returning empty

        Each returned dict contains:
          - question  : the matched document question
          - answer    : the matched document answer
          - source    : the source file / sheet
          - category  : the topic category
          - score     : cosine-like similarity (1 = perfect, 0 = unrelated)
          - rank      : 1-based retrieval rank
        """
        if not query.strip():
            logger.warning("[Retriever] Empty query received — returning empty results.")
            return []

        logger.info(f"[Retriever] Query: '{query}'")
        t0 = time.perf_counter()

        # ── Embed the query ───────────────────────────────────────────────────
        query_vec = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        embed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"[Retriever] Query embedding done in {embed_ms:.1f} ms")

        # ── FAISS search ──────────────────────────────────────────────────────
        t1 = time.perf_counter()
        distances, indices = self.index.search(query_vec, k=top_k)
        search_ms = (time.perf_counter() - t1) * 1000
        logger.debug(f"[Retriever] FAISS search done in {search_ms:.2f} ms")

        # ── Build results with threshold fallback ─────────────────────────────
        FALLBACK_THRESHOLD = 0.20   # softer threshold for second pass

        for threshold in (MIN_SCORE_THRESHOLD, FALLBACK_THRESHOLD):
            results = []
            for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                score = round(float(1 - dist / 2), 4)
                chunk = self.metadata[idx]

                if score < threshold:
                    logger.debug(
                        f"[Retriever] Rank {rank+1} dropped — score {score:.3f} "
                        f"below threshold {threshold}"
                    )
                    continue

                result = {
                    "rank"    : rank + 1,
                    "score"   : score,
                    "question": chunk.get("question", ""),
                    "answer"  : chunk.get("answer", ""),
                    "source"  : chunk.get("source", "Unknown"),
                    "category": chunk.get("category", "General"),
                }
                results.append(result)

                logger.debug(
                    f"[Retriever] Rank {rank+1} | Score: {score:.3f} | "
                    f"Src: {chunk.get('source','?')} | "
                    f"Q: {chunk.get('question','')[:60]}..."
                )

            if results:
                if threshold == FALLBACK_THRESHOLD and threshold < MIN_SCORE_THRESHOLD:
                    logger.warning(
                        f"[Retriever] Used fallback threshold {threshold} — "
                        f"no chunks passed strict threshold {MIN_SCORE_THRESHOLD}"
                    )
                break  # got results, stop

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[Retriever] Retrieved {len(results)} chunks in {total_ms:.1f} ms "
            f"(embed: {embed_ms:.1f} ms, search: {search_ms:.2f} ms)"
        )
        return results
