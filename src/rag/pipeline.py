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
  │  [History augmentation]                │  ← prepend last N turns to query
  │         ↓                              │
  │  [Retriever.retrieve]                  │  ← FAISS top-k search
  │         ↓                              │
  │  [PromptBuilder.build_prompt]          │  ← format context + history + question
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

# ── Conversation memory settings ──────────────────────────────────────────────
MAX_HISTORY_TURNS  = 4    # number of past (user, bot) pairs to remember
MAX_SESSIONS       = 500  # cap to avoid unbounded memory growth


class NustBankRAGPipeline:
    """
    Single entry-point for the entire RAG pipeline.
    Initialise once, call `answer(query, session_id)` repeatedly.

    Conversation history is stored per session_id so that multiple
    browser tabs / users each maintain independent context.
    """

    def __init__(self):
        setup_logger("logs/pipeline.log")
        logger.info("=" * 60)
        logger.info("[Pipeline] Starting NUST Bank RAG Pipeline initialisation")
        logger.info("=" * 60)

        self.retriever = NustBankRetriever()
        self.llm       = LLMHandler()
        self.guard     = GuardRail()

        # session_id → list of {"query": str, "answer": str}
        self.sessions: Dict[str, List[Dict]] = {}

        logger.success("[Pipeline] All components ready. Pipeline is live.")

    # ── Session helpers ───────────────────────────────────────────────────────

    def _get_history(self, session_id: str) -> List[Dict]:
        return self.sessions.get(session_id, [])

    def _save_turn(self, session_id: str, query: str, answer: str):
        if session_id not in self.sessions:
            # Evict oldest session if cap reached
            if len(self.sessions) >= MAX_SESSIONS:
                oldest = next(iter(self.sessions))
                del self.sessions[oldest]
                logger.debug(f"[Pipeline] Evicted oldest session: {oldest}")
            self.sessions[session_id] = []

        self.sessions[session_id].append({"query": query, "answer": answer})

        # Keep only latest N turns
        if len(self.sessions[session_id]) > MAX_HISTORY_TURNS:
            self.sessions[session_id] = self.sessions[session_id][-MAX_HISTORY_TURNS:]

    def clear_session(self, session_id: str):
        """Wipe conversation history for a given session (called on Clear Chat)."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"[Pipeline] Session cleared: {session_id}")

    # ── Main answer method ────────────────────────────────────────────────────

    def answer(self, query: str, session_id: str = "default", user_context: dict = None) -> Dict:
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
        history = self._get_history(session_id)

        logger.info(f"[Pipeline] ── NEW QUERY ──────────────────────────────")
        logger.info(f"[Pipeline] Session: '{session_id}' | History turns: {len(history)} | Profile: {'yes' if user_context else 'no'}")
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

        # ── Step 2: Augment query with history for better retrieval ───────────
        # Only prepend the last 1-2 User queries to resolve coreference (it/that/those)
        # We avoid prepending the assistant's previous "I don't know" answers to prevent contamination.
        augmented_query = query
        if history:
            recent_user_queries = [h['query'] for h in history[-2:]]
            history_context = " ".join(recent_user_queries)
            augmented_query = f"{history_context} {query}"
            logger.debug(f"[Pipeline] Augmented query for retrieval: '{augmented_query}'")

        # ── Step 3: Retrieval ─────────────────────────────────────────────────
        chunks = self.retriever.retrieve(augmented_query, top_k=TOP_K_RESULTS)
        logger.info(f"[Pipeline] Retrieved {len(chunks)} relevant chunks.")
        for c in chunks:
            logger.info(f"[Pipeline]   - Chunk {c['rank']} [{c['score']}]: Q: {c['question'][:60]}... | A: {c['answer'][:60]}...")

        # ── Step 4: Prompt Construction (with history injected) ───────────────
        prompt = build_prompt(query, chunks, history=history, user_context=user_context)
        logger.info(f"[Pipeline] Prompt built. Context used: {prompt['context_used']}")

        # ── Step 5: LLM Generation ────────────────────────────────────────────
        raw_answer = self.llm.generate(
            system=prompt["system"],
            user=prompt["user"]
        )

        # ── Step 6: Output Guardrail ──────────────────────────────────────────
        clean_answer = self.guard.check_output(raw_answer)

        # ── Step 7: Save turn to history ──────────────────────────────────────
        self._save_turn(session_id, query, clean_answer)

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
