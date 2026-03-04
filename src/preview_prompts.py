"""
src/preview_prompts.py
───────────────────────
Preview what the RAG pipeline sends to the LLM — WITHOUT making any API calls.

For each query in dummy_prompts.txt, this script will:
  1. Run FAISS retrieval
  2. Build the full prompt (system + user message)
  3. Print everything the LLM would receive

No HF_API_TOKEN required. Safe to run at any time.

Usage:
    venv\Scripts\python.exe src/preview_prompts.py
"""

import sys, os
sys.path.insert(0, os.getcwd())

from src.rag.retriever import NustBankRetriever
from src.rag.prompt_builder import build_prompt
from src.guardrails.guard import GuardRail


PROMPTS_FILE = "dummy_prompts.txt"
SEPARATOR    = "=" * 70


def load_prompts():
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Strip leading numbering like "1. " if present
    prompts = []
    for line in lines:
        if line[0].isdigit() and ". " in line:
            prompts.append(line.split(". ", 1)[1])
        else:
            prompts.append(line)
    return prompts


def preview():
    print("\n" + SEPARATOR)
    print("  NUST BANK RAG — PROMPT PREVIEW (No LLM API called)")
    print(SEPARATOR)

    retriever = NustBankRetriever()
    guard     = GuardRail()
    queries   = load_prompts()

    print(f"\n  Loaded {len(queries)} queries from {PROMPTS_FILE}\n")

    for i, query in enumerate(queries, start=1):
        print(f"\n{SEPARATOR}")
        print(f"  QUERY {i:02d}: {query}")
        print(SEPARATOR)

        # ── Guardrail check ───────────────────────────────────────────────────
        is_safe, guard_msg = guard.check_input(query)
        if not is_safe:
            print(f"  ⛔ BLOCKED by GuardRail: {guard_msg}")
            continue

        # ── Retrieval ─────────────────────────────────────────────────────────
        chunks = retriever.retrieve(query)

        print(f"\n  📦 RETRIEVED CHUNKS ({len(chunks)} results):")
        for c in chunks:
            print(f"    ┌─ Rank {c['rank']} | Score: {c['score']:.4f} | Source: {c['source']}")
            print(f"    │  Q: {c['question'][:80]}")
            print(f"    └─ A: {c['answer'][:100]}...")

        # ── Prompt Construction ───────────────────────────────────────────────
        prompt = build_prompt(query, chunks)

        print(f"\n  📨 FULL PROMPT SENT TO LLM:")
        print(f"\n  ── [SYSTEM PROMPT] ({'─'*47})")
        for line in prompt["system"].split("\n"):
            print(f"  │  {line}")

        print(f"\n  ── [USER MESSAGE] ({'─'*48})")
        for line in prompt["user"].split("\n"):
            print(f"  │  {line}")

        print(f"\n  {'─'*66}")
        print(f"  Total system chars : {len(prompt['system'])}")
        print(f"  Total user chars   : {len(prompt['user'])}")
        print(f"  Context used       : {prompt['context_used']}")
        print(f"  LLM would now generate a response based on the above ☝️")

    print(f"\n\n{SEPARATOR}")
    print(f"  Preview complete — {len(queries)} prompts shown.")
    print(f"  To generate real answers, set HF_API_TOKEN in .env and run:")
    print(f"    venv\\Scripts\\python.exe src\\test_rag.py")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    preview()
