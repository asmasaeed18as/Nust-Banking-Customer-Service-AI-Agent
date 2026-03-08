"""
Global Configuration for NUST Bank AI Agent.
All tuneable parameters live here — no hardcoded values in other files.
"""

import os
from pathlib import Path

# ── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent

# ── Data Paths ───────────────────────────────────────────────────────────────
RAW_DATA_DIR        = ROOT_DIR / "Bank Knowledge"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "bank_knowledge_chunks.json"
VECTOR_STORE_DIR    = ROOT_DIR / "data" / "vector_store"
FAISS_INDEX_PATH    = VECTOR_STORE_DIR / "bank_faiss_index.bin"
METADATA_PATH       = VECTOR_STORE_DIR / "bank_metadata.json"
LOGS_DIR            = ROOT_DIR / "logs"

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL     = "Alibaba-NLP/gte-large-en-v1.5"  # 1024-dim, MTEB 65.4, English retrieval
EMBEDDING_DIMENSION = 1024

# ── Retrieval (RAG) ───────────────────────────────────────────────────────────
TOP_K_RESULTS       = 3       # Number of FAISS chunks returned per query
MIN_SCORE_THRESHOLD = 0.35    # Raised from 0.30 — gte-large scores are tighter/higher

# ── LLM Configuration ─────────────────────────────────────────────────────────
# Primary: Qwen2.5-3B-Instruct via HuggingFace Inference API (free, no GPU)
# Fallback: Local model loading if HF API is unavailable
LLM_PROVIDER        = "hf_api"           # "hf_api" | "local"
HF_MODEL_ID         = "Qwen/Qwen2.5-3B-Instruct"
HF_API_TOKEN        = os.getenv("HF_API_TOKEN", "")  # reads from .env

# Local model settings (used if LLM_PROVIDER = "local")
LOCAL_MODEL_PATH    = HF_MODEL_ID        # will be downloaded by HF
MAX_NEW_TOKENS      = 512
TEMPERATURE         = 0.3                # Low temp → more factual / less creative
TOP_P               = 0.9
REPETITION_PENALTY  = 1.15

# ── Bank Identity ─────────────────────────────────────────────────────────────
BANK_NAME           = "NUST Bank"
BOT_NAME            = "NUST Bank AI Assistant"
SUPPORT_EMAIL       = "support@NUSTbank.com.pk"
SUPPORT_UAN         = "+92 (51) 111 000 494"
