"""
backend/main.py
────────────────
FastAPI backend for the NUST Bank AI Agent.

Endpoints:
  GET  /api/health        → server + pipeline status
  POST /api/chat          → main RAG query endpoint
  POST /api/upload        → upload new document (real-time KB update)
  GET  /api/sources       → list all indexed knowledge sources
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.logging_config import setup_logger
from src.rag.pipeline import NustBankRAGPipeline
from src.ingest import NustBankIngestor
from src.vector_store import NustBankIndexer

setup_logger("logs/api.log")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NUST Bank AI Customer Service API",
    description="RAG-powered banking assistant using Qwen 2.5-3B",
    version="1.0.0",
)

# ── CORS — allow React dev server ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pipeline (loaded once at startup) ────────────────────────────────────────
pipeline: NustBankRAGPipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    logger.info("[API] Starting up NUST Bank API server...")
    pipeline = NustBankRAGPipeline()
    logger.success("[API] Pipeline ready. Server is live.")


# ── Request / Response Models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    top_k: int = 3

class SourceChunk(BaseModel):
    rank: int
    score: float
    question: str
    answer: str
    source: str
    category: str

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list
    context_used: bool
    blocked: bool
    latency_ms: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {
        "status"    : "online",
        "pipeline"  : "ready" if pipeline else "not initialized",
        "model"     : os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"),
        "timestamp" : time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

    logger.info(f"[API /chat] Received query: '{request.query}'")

    result = pipeline.answer(request.query)
    return ChatResponse(**result)


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a new JSON or CSV knowledge document, ingest it,
    update the FAISS index so answers are available immediately.
    """
    if not file.filename.endswith((".json", ".csv", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only .json, .csv, or .txt files are supported."
        )

    upload_dir = ROOT / "Bank Knowledge"
    dest = upload_dir / file.filename
    logger.info(f"[API /upload] Receiving file: {file.filename}")

    # Save file
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.success(f"[API /upload] Saved to {dest}")

    # Re-ingest and re-index
    try:
        ingestor = NustBankIngestor(str(upload_dir))
        ingestor.ingest_json()
        ingestor.ingest_excel()
        ingestor.save_processed()

        indexer = NustBankIndexer()
        indexer.load_processed_data(str(ROOT / "data" / "processed" / "bank_knowledge_chunks.json"))
        indexer.create_index()
        indexer.save_index()

        # Reload pipeline retriever in-memory
        global pipeline
        pipeline = NustBankRAGPipeline()

        logger.success(f"[API /upload] Re-indexing complete after uploading {file.filename}")
        return {"status": "success", "message": f"'{file.filename}' ingested and index updated."}
    except Exception as e:
        logger.error(f"[API /upload] Re-indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.get("/api/sources")
def get_sources():
    """Return a summary of all indexed knowledge sources."""
    meta_path = ROOT / "data" / "vector_store" / "bank_metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Index not built yet.")

    with open(meta_path) as f:
        meta = json.load(f)

    from collections import Counter
    sources = Counter([m.get("source", "Unknown") for m in meta])
    return {
        "total_chunks": len(meta),
        "sources": [{"name": k, "count": v} for k, v in sources.most_common()],
    }
