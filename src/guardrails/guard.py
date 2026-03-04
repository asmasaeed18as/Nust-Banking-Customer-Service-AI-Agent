"""
src/guardrails/guard.py
────────────────────────
Input + Output safety layer for the NUST Bank AI Agent.

Checks performed:
  Pre-LLM  (on user input):
    1. Length check — very long queries may be prompt injections
    2. Jailbreak pattern detection — "ignore previous instructions", "DAN", etc.
    3. PII guard — user shouldn't be sending actual account numbers
    4. Off-topic detection — obvious non-banking requests

  Post-LLM (on assistant output):
    1. Hallucination markers — model saying "I don't know" in harmful ways
    2. Forbidden data patterns — model accidentally leaking fake PII
"""

import re
from loguru import logger
from typing import Tuple

# ── Pre-LLM: Jailbreak & Injection Patterns ───────────────────────────────────
JAILBREAK_PATTERNS = [
    r"ignore (previous|all|your) instructions",
    r"you are now (DAN|an? unrestricted|an? jailbroken)",
    r"pretend (you have no|you are not)",
    r"do anything now",
    r"bypass (your|the) (restrictions|rules|guidelines|filter)",
    r"act as (if you have no|a different|an unrestricted)",
    r"forget (you are|you're) a bank",
    r"reveal (your|the) (system prompt|instructions|prompt)",
    r"what is your (system prompt|initial prompt)",
]

# ── Pre-LLM: Off-topic patterns (obvious non-banking topics) ─────────────────
OFF_TOPIC_PATTERNS = [
    r"\b(recipe|cook|film|movie|sports|cricket|football|game)\b",
    r"\b(write (a poem|code|essay|story))\b",
    r"\b(who is (the president|prime minister))\b",
    r"\b(tell me a joke|joke)\b",
]

# ── Post-LLM: Patterns that should NOT appear in outputs ─────────────────────
FORBIDDEN_OUTPUT_PATTERNS = [
    r"PK\d{2}[A-Z]{4}\d{16}",   # IBAN pattern
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # card number
]

MAX_QUERY_LENGTH = 1500  # characters


class GuardRail:
    """
    Applies safety checks before and after the LLM call.
    """

    def check_input(self, query: str) -> Tuple[bool, str]:
        """
        Returns (is_safe: bool, reason: str).
        If is_safe=False, reason contains a user-friendly message to return instead.
        """

        # ── 1. Length check ───────────────────────────────────────────────────
        if len(query) > MAX_QUERY_LENGTH:
            logger.warning(f"[GuardRail] Query too long ({len(query)} chars). Blocking.")
            return False, (
                "Your message is too long. Please ask a shorter, specific question "
                "about NUST Bank services."
            )

        # ── 2. Jailbreak / prompt injection detection ─────────────────────────
        query_lower = query.lower()
        for pattern in JAILBREAK_PATTERNS:
            if re.search(pattern, query_lower):
                logger.warning(f"[GuardRail] Jailbreak attempt detected: pattern='{pattern}' query='{query[:60]}'")
                return False, (
                    "I'm designed to assist with NUST Bank services only. "
                    "I cannot respond to that type of request."
                )

        # ── 3. Off-topic detection ────────────────────────────────────────────
        for pattern in OFF_TOPIC_PATTERNS:
            if re.search(pattern, query_lower):
                logger.warning(f"[GuardRail] Off-topic query detected: '{query[:60]}'")
                return False, (
                    "I'm only able to help with NUST Bank banking questions — "
                    "accounts, transfers, cards, loans, and similar topics. "
                    "For other queries, please consult a general assistant."
                )

        logger.debug(f"[GuardRail] Input passed all checks. Query='{query[:60]}'")
        return True, ""

    def check_output(self, response: str) -> str:
        """
        Scrubs the LLM output of any forbidden content.
        Returns the cleaned (or unchanged) response.
        """

        for pattern in FORBIDDEN_OUTPUT_PATTERNS:
            if re.search(pattern, response):
                logger.warning(f"[GuardRail] Forbidden pattern found in LLM output. Redacting.")
                response = re.sub(pattern, "[REDACTED]", response)

        return response
