"""
src/rag/prompt_builder.py
─────────────────────────
Builds the complete LLM prompt from retrieved context chunks
and optional conversation history.

Design decisions:
  - System prompt is banking-specific and instructs the model to be helpful,
    honest, and to redirect out-of-domain questions gracefully.
  - Conversation history is injected BEFORE context so the model understands
    what "it", "that", "its rates" etc. refer to.
  - Context chunks are numbered so the model can reference them.
  - Question is appended last so it is in the model's most recent attention.
"""

from typing import List, Dict, Optional
from loguru import logger

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import BANK_NAME, BOT_NAME, SUPPORT_EMAIL, SUPPORT_UAN


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt (injected before every conversation)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are {BOT_NAME}, a professional, friendly, and knowledgeable \
customer service representative for {BANK_NAME}.

Your responsibilities:
1. Answer customer questions ONLY using the provided context sections below.
2. If the answer is clearly in the context, give a complete, helpful response.
3. If the context does NOT contain enough information, politely say so and \
direct the customer to contact support:
   - Email: {SUPPORT_EMAIL}
   - UAN: {SUPPORT_UAN}
4. Keep answers concise but complete. Use bullet points for multi-step answers.
5. NEVER fabricate account numbers, interest rates, or policy details.
6. NEVER discuss topics unrelated to banking or {BANK_NAME} services.
7. Always maintain a warm, professional, and reassuring tone.
8. If asked who you are, say you are the {BOT_NAME} AI assistant.
9. Use the conversation history to resolve pronouns like "it", "its", "that", \
"this" — they refer to whatever was discussed most recently.

IMPORTANT: Only answer from the CONTEXT provided. Do not use external knowledge."""


def build_prompt(
    query: str,
    context_chunks: List[Dict],
    history: Optional[List[Dict]] = None,
    user_context: Optional[Dict] = None,
) -> dict:
    """
    Constructs a chat-style prompt dictionary with:
      - system  : banking-specific instructions
      - user    : conversation history + retrieved context + customer question

    Args:
        query          : The current user question.
        context_chunks : Retrieved knowledge chunks from the vector store.
        history        : List of past {"query": str, "answer": str} dicts.

    Returns a dict with 'system', 'user', and 'context_used' keys.
    """

    history = history or []

    # ── Format user profile (optional) ───────────────────────────────────────
    profile_block = ""
    if user_context:
        lines = []
        label_map = {
            "customer_type"      : "Customer type",
            "employment"         : "Employment",
            "age_group"          : "Age group",
            "existing_products"  : "Existing products",
            "interests"          : "Areas of interest",
        }
        for key, label in label_map.items():
            val = user_context.get(key)
            if val:
                if isinstance(val, list):
                    lines.append(f"  - {label}: {', '.join(val)}")
                else:
                    lines.append(f"  - {label}: {val}")
        if lines:
            profile_block = (
                "=== Customer Profile ===\n"
                + "\n".join(lines)
                + "\n========================\n\n"
            )
            logger.debug(f"[PromptBuilder] Injected user profile with {len(lines)} fields.")

    # ── Format conversation history ───────────────────────────────────────────
    history_block = ""
    if history:
        lines = []
        for turn in history:
            lines.append(f"Customer: {turn['query']}")
            # Truncate long answers in history to keep prompt size manageable
            answer_preview = turn['answer'][:300] + "..." if len(turn['answer']) > 300 else turn['answer']
            lines.append(f"Assistant: {answer_preview}")
        history_block = (
            "=== Conversation History (most recent) ===\n"
            + "\n".join(lines)
            + "\n==========================================\n\n"
        )
        logger.debug(f"[PromptBuilder] Injected {len(history)} history turns into prompt.")

    # ── Format context chunks ─────────────────────────────────────────────────
    if not context_chunks:
        logger.warning("[PromptBuilder] No context chunks — building fallback prompt.")
        user_content = (
            f"{profile_block}"
            f"{history_block}"
            f"A customer asked: {query}\n\n"
            "No relevant context was found in the knowledge base. "
            "Please provide a helpful response directing them to contact support."
        )
        return {
            "system"      : SYSTEM_PROMPT,
            "user"        : user_content,
            "context_used": False,
        }

    context_lines = []
    for i, chunk in enumerate(context_chunks, start=1):
        block = (
            f"[Context {i}] (Source: {chunk['source']} | Score: {chunk['score']})\n"
            f"Q: {chunk['question']}\n"
            f"A: {chunk['answer']}"
        )
        context_lines.append(block)

    context_text = "\n\n".join(context_lines)

    user_content = (
        f"{profile_block}"
        f"{history_block}"
        f"Here is relevant information from the {BANK_NAME} knowledge base:\n\n"
        f"{context_text}\n\n"
        f"---\n"
        f"Customer Question: {query}\n\n"
        f"Please answer the customer's question using the context above."
    )

    logger.debug(
        f"[PromptBuilder] Built prompt with {len(context_chunks)} context chunks, "
        f"{len(history)} history turns. "
        f"Total user content length: {len(user_content)} chars."
    )

    return {
        "system"      : SYSTEM_PROMPT,
        "user"        : user_content,
        "context_used": True,
    }
