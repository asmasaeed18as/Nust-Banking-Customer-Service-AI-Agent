"""
src/rag/prompt_builder.py
─────────────────────────
Builds the complete LLM prompt from retrieved context chunks.

Design decisions:
  - System prompt is banking-specific and instructs the model to be helpful,
    honest, and to redirect out-of-domain questions gracefully.
  - Context chunks are numbered so the model can reference them.
  - Question is appended last so it is in the model's most recent attention.
"""

from typing import List, Dict
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

IMPORTANT: Only answer from the CONTEXT provided. Do not use external knowledge."""


def build_prompt(query: str, context_chunks: List[Dict]) -> dict:
    """
    Constructs a chat-style prompt dictionary with:
      - system  : banking-specific instructions
      - user    : retrieved context + customer question

    Returns a dict with 'system' and 'user' keys, and a 'context_used' flag.
    """

    if not context_chunks:
        logger.warning("[PromptBuilder] No context chunks — building fallback prompt.")
        user_content = (
            f"A customer asked: {query}\n\n"
            "No relevant context was found in the knowledge base. "
            "Please provide a helpful response directing them to contact support."
        )
        return {
            "system"      : SYSTEM_PROMPT,
            "user"        : user_content,
            "context_used": False,
        }

    # ── Format each retrieved chunk ───────────────────────────────────────────
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
        f"Here is relevant information from the {BANK_NAME} knowledge base:\n\n"
        f"{context_text}\n\n"
        f"---\n"
        f"Customer Question: {query}\n\n"
        f"Please answer the customer's question using the context above."
    )

    logger.debug(
        f"[PromptBuilder] Built prompt with {len(context_chunks)} context chunks. "
        f"Total user content length: {len(user_content)} chars."
    )

    return {
        "system"      : SYSTEM_PROMPT,
        "user"        : user_content,
        "context_used": True,
    }
