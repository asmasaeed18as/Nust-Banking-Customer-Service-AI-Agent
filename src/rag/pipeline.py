"""
src/rag/pipeline.py
────────────────────
Main RAG orchestration pipeline for the NUST Bank AI Agent.

Full flow for every query:
  ┌─────────────────────────────────────────┐
  │  User Query                             │
  │         ↓                              │
  │  [GuardRail.check_input]               │  ← block jailbreaks / off-topic
  │         ↓                              │
  │  [Retriever.retrieve]                  │  ← FAISS top-k search
  │         ↓                              │
  │  [PromptBuilder.build_prompt]          │  ← format context + question
  │         ↓                              │
  │  [LLMHandler.generate]                 │  ← call Qwen / local model
  │         ↓                              │
  │  [GuardRail.check_output]              │  ← redact forbidden patterns
  │         ↓                              │
  │  Structured Response Dict              │
  └─────────────────────────────────────────┘
"""

import time
from loguru import logger
from typing import Dict, List

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.logging_config import setup_logger
from src.rag.retriever import NustBankRetriever
from src.rag.prompt_builder import build_prompt
from src.rag.llm_handler import LLMHandler
from src.guardrails.guard import GuardRail
from config.settings import TOP_K_RESULTS


class NustBankRAGPipeline:
    """
    Single entry-point for the entire RAG pipeline.
    Initialise once, call `answer(query)` repeatedly.
    """

    def __init__(self):
        setup_logger("logs/pipeline.log")
        logger.info("=" * 60)
        logger.info("[Pipeline] Starting NUST Bank RAG Pipeline initialisation")
        logger.info("=" * 60)

        self.retriever = NustBankRetriever()
        self.llm       = LLMHandler()
        self.guard     = GuardRail()

        logger.success("[Pipeline] All components ready. Pipeline is live.")

    def answer(self, query: str) -> Dict:
        """
        End-to-end pipeline: query → guardrail → retrieval → prompt → LLM → response.

        Returns a structured dict:
        {
            "query"         : str,
            "answer"        : str,
            "sources"       : List[dict],   # retrieved chunks (with scores)
            "context_used"  : bool,
            "blocked"       : bool,         # True if guardrail rejected query
            "latency_ms"    : float,        # total wall-clock time in ms
        }
        """
        total_start = time.perf_counter()
        logger.info(f"[Pipeline] ── NEW QUERY ──────────────────────────────")
        logger.info(f"[Pipeline] Query received: '{query}'")

        # ── Step 1: Input Guardrail ───────────────────────────────────────────
        is_safe, guard_message = self.guard.check_input(query)
        if not is_safe:
            logger.warning(f"[Pipeline] Query BLOCKED by guardrail.")
            return {
                "query"       : query,
                "answer"      : guard_message,
                "sources"     : [],
                "context_used": False,
                "blocked"     : True,
                "latency_ms"  : round((time.perf_counter() - total_start) * 1000, 1),
            }

        # ── Step 2: Retrieval ─────────────────────────────────────────────────
        chunks = self.retriever.retrieve(query, top_k=TOP_K_RESULTS)
        logger.info(f"[Pipeline] Retrieved {len(chunks)} relevant chunks.")
        for c in chunks:
            logger.debug(
                f"[Pipeline]   Rank {c['rank']} | Score {c['score']} | "
                f"{c['source']} | Q: {c['question'][:55]}..."
            )

        # ── Step 3: Prompt Construction ───────────────────────────────────────
        prompt = build_prompt(query, chunks)
        logger.info(f"[Pipeline] Prompt built. Context used: {prompt['context_used']}")

        # ── Step 4: LLM Generation ────────────────────────────────────────────
        raw_answer = self.llm.generate(
            system=prompt["system"],
            user=prompt["user"]
        )

        # ── Step 5: Output Guardrail ──────────────────────────────────────────
        clean_answer = self.guard.check_output(raw_answer)

        total_ms = round((time.perf_counter() - total_start) * 1000, 1)
        logger.success(f"[Pipeline] Query answered in {total_ms} ms.")
        logger.info(f"[Pipeline] Answer preview: {clean_answer[:120]}...")

        return {
            "query"       : query,
            "answer"      : clean_answer,
            "sources"     : chunks,
            "context_used": prompt["context_used"],
            "blocked"     : False,
            "latency_ms"  : total_ms,
        }
