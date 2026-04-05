# NUST Bank AI Customer Service Agent
 Retrieval-Augmented Generation (RAG) system for banking customer service. The agent answers customer queries using a structured knowledge base of FAQs, product sheets, and policy documents, generating context-aware responses via a large language model.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Breakdown](#component-breakdown)
3. [RAG Pipeline — Step by Step](#rag-pipeline--step-by-step)
4. [Technology Stack and Design Rationale](#technology-stack-and-design-rationale)
5. [Features](#features)
6. [Data Pipeline](#data-pipeline)
7. [Configuration](#configuration)
8. [Running the System](#running-the-system)
9. [Project Structure](#project-structure)

---

## Architecture Overview

```
Customer Query (Frontend)
        |
        v
[Input Guardrail]  -- jailbreak / off-topic filter
        |
        v
[Query Augmentation] -- prepend recent history for coreference resolution
        |
        v
[Embedding Model]  -- Alibaba-NLP/gte-large-en-v1.5 (1024-dim)
        |
        v
[FAISS Index]  -- cosine similarity search, two-pass threshold
        |
        v
[Prompt Builder]  -- profile block + history + context + question
        |
        v
[LLM]  -- Qwen/Qwen2.5-3B-Instruct via HuggingFace Inference API
        |
        v
[Output Guardrail]  -- PII redaction (IBAN, card, CNIC)
        |
        v
Response (Frontend)
```

The system is fully modular: each stage is an independent Python class, making it straightforward to swap components (e.g., replace FAISS with Qdrant, or the LLM with a locally hosted model).

---

## Component Breakdown

### 1. Data Ingestion (`src/ingest.py`)

Reads raw knowledge from two formats:
- **JSON** (`data/raw/*.json`): structured FAQ arrays with `category → questions → answer` hierarchy.
- **Excel** (`data/raw/*.xlsx`): product sheets for individual account types and loan products.

Each document is flattened into chunks of the form:
```json
{
  "question": "...",
  "answer": "...",
  "source": "Excel - NMF",
  "category": "NMF"
}
```

All chunks are saved to `data/processed/bank_knowledge_chunks.json` (1,760 chunks across the current knowledge base).

### 2. Vector Store (`src/vector_store.py`)

Embeds all chunks and builds a FAISS index.

- Each `(question + answer)` pair is embedded as a single vector.
- Both fields are concatenated before embedding so cosine similarity captures semantic meaning of the Q-A pair, not just the question alone.
- The index is saved as `data/vector_store/bank_faiss_index.bin` with a parallel `bank_metadata.json` that maps FAISS indices back to the original chunk fields.

### 3. Retriever (`src/rag/retriever.py`)

At query time, the user's (augmented) query is embedded and compared against the FAISS index using inner product (equivalent to cosine similarity because all vectors are L2-normalised).

Two-pass threshold logic:
- **Pass 1 — strict (0.35)**: returns results confidently above the threshold.
- **Pass 2 — fallback (0.20)**: if nothing passes the strict threshold, retries with a softer cut-off before concluding no context is available.

This prevents the system from immediately deflecting with "contact support" on valid but semantically unusual queries.

### 4. Prompt Builder (`src/rag/prompt_builder.py`)

Assembles the full prompt in this order:

```
[System Prompt]
[Customer Profile Block]  (if user has set a profile)
[Conversation History Block]  (last 4 turns)
[Retrieved Context Chunks]  (ranked by similarity score)
[Customer Question]
```

The order is intentional: the LLM reads profile and history before context, so it has the framing needed to interpret coreferences ("what are its rates?" → resolves to the last product discussed) and tailor the response to the customer type.

### 5. LLM Handler (`src/rag/llm_handler.py`)

Calls the HuggingFace Inference API with Qwen2.5-3B-Instruct. The prompt is sent as a structured chat message (system + user).

Parameters:
- `temperature: 0.3` — low to prioritize factual accuracy over creativity
- `max_new_tokens: 512` — sufficient for complete multi-step banking answers
- `repetition_penalty: 1.15` — prevents the model from looping on phrases
- `top_p: 0.9` — nucleus sampling for natural phrasing

### 6. Conversation Memory (`src/rag/pipeline.py`)

Each browser session gets a unique UUID (`session_id`) generated on page load. The pipeline maintains a server-side dict mapping session IDs to conversation history.

Design decisions:
- **Max turns stored: 4** — sufficient for multi-step queries without inflating prompt size.
- **Retrieval augmentation**: the last 2 turns (not all 4) are prepended to the raw retrieval query. This resolves coreferences at the retrieval stage, so the right chunks are fetched for follow-up questions. The full 4-turn history still appears in the prompt for response generation.
- **LRU eviction**: sessions are evicted oldest-first when the count exceeds 500.
- **Explicit clear**: clicking "Clear Chat" calls `POST /api/chat/clear` which removes the server-side history for that session.

### 7. Guardrails (`src/guardrails/guard.py`)

Two enforcement points:

**Input guardrail** (pre-LLM):
- 16 jailbreak detection patterns (regex): covers prompt injection, roleplay attacks, hypothetical framing, DAN-style instructions, override attempts, and system prompt extraction.
- Off-topic filter: rejects queries clearly unrelated to banking (sports, politics, general coding, etc.).
- Maximum query length: 1,500 characters.

**Output guardrail** (post-LLM):
- IBAN number pattern masking (`PK\d{2}[A-Z]{4}\d{16}`).
- Credit/debit card number masking (16-digit sequences).
- CNIC masking (Pakistani national ID format `XXXXX-XXXXXXX-X`).

Regex-based guardrails were chosen over a secondary LLM classifier to keep per-query latency deterministic. The patterns cover the realistic threat surface for a banking chatbot without the overhead of an additional model call.

### 8. User Profile Personalization

Users can optionally provide a profile (customer type, employment, age group, existing products, areas of interest). This is stored in React state client-side and serialized as a compact dict sent with each request when any field is set.

The profile is injected as the first block in the LLM prompt. This allows the model to frame answers appropriately — for example, a salaried individual asking about loans gets different product recommendations than a business owner, even if both queries are identical.

No profile data is persisted server-side; it is scoped to the browser session.

### 9. Real-Time Knowledge Updates (`/api/upload`)

New documents can be uploaded via the UI. The backend:
1. Saves the file to `data/raw/`.
2. Re-runs the ingestor over all documents.
3. Re-embeds all chunks and rebuilds the FAISS index.
4. Reloads the pipeline in memory.

The AI can answer questions from the new document immediately without restarting the server.

---

## RAG Pipeline — Step by Step

When a user sends a query, the following happens in sequence:

| Step | Component | Purpose |
|------|-----------|---------|
| 1 | `GuardRail.check_input()` | Block jailbreaks and off-topic queries |
| 2 | History augmentation | Prepend last 2 turns to query for retrieval coreference resolution |
| 3 | `NustBankRetriever.retrieve()` | Embed augmented query, search FAISS, apply threshold |
| 4 | `build_prompt()` | Assemble profile + history + context + question |
| 5 | `LLMHandler.generate()` | Call Qwen2.5-3B via HF Inference API |
| 6 | `GuardRail.check_output()` | Redact any PII patterns in the response |
| 7 | `pipeline._save_turn()` | Append exchange to session history |
| 8 | Return structured response | `{answer, sources, blocked, latency_ms}` |

---

## Technology Stack and Design Rationale

### Embedding Model: `Alibaba-NLP/gte-large-en-v1.5`

| Metric | Value |
|--------|-------|
| MTEB (retrieval subset) | 65.4 |
| Previous model (all-MiniLM-L6-v2) | 56.3 |
| Architecture | BERT-Large, 335M parameters |
| Embedding dimension | 1024 |

**Why this model:**
- **English-optimized**: The entire knowledge base is in English. Multilingual models (e.g., BGE-M3) consume parameter capacity on non-English linguistic features that provide no benefit here.
- **Accuracy over speed**: For a banking customer service context, retrieval precision matters — a wrong document returned means a wrong or misleading answer. The +9.1 MTEB improvement over MiniLM directly improves answer quality.
- **Size constraint**: At 335M parameters it is substantially lighter than BGE-M3 (570M) and Stella-1.5B (1.5B), making it viable for CPU inference.
- **Rejected alternatives**:
  - *all-MiniLM-L6-v2*: Fast but 56.3 MTEB — too much accuracy sacrifice for a domain requiring precise retrieval of rates and limits.
  - *BAAI/bge-m3*: Higher ambiguity due to multilingual training; also slower and larger for no gain on an English-only dataset.
  - *dunzhang/stella_en_1.5B_v5*: Better accuracy but 1.5B parameters exceed practical CPU latency limits.

### LLM: `Qwen/Qwen2.5-3B-Instruct`

**Why this model:**
- Accessed via HuggingFace Inference API: no local GPU required.
- At 3B parameters, it is the smallest model that reliably follows complex structured prompt formats and produces well-formatted banking answers.
- The Qwen 2.5 series is strong on factual response generation and adheres well to system prompt constraints.
- Instruction-tuned: crucial for enforcing the "only answer from context" directive in the system prompt.

### Vector Store: FAISS (`IndexFlatIP`)

**Why FAISS:**
- At ~1,760 chunks, exhaustive search is sub-millisecond. There is no need for approximate nearest-neighbour methods (HNSW, IVF) which trade recall for speed.
- `IndexFlatIP` on L2-normalized vectors gives exact cosine similarity — appropriate for semantic search where recall matters more than raw throughput.
- Serverless: no background process required, unlike Qdrant or Weaviate. Suitable for single-machine academic deployment.
- **If scaling to production**: Qdrant or Weaviate would be appropriate given their support for metadata filtering, distributed deployment, and incremental index updates.

### Backend: FastAPI

- Async by default, minimal boilerplate.
- Automatic OpenAPI docs at `/docs` for easy testing.
- Pydantic validation on all request/response models.

### Frontend: React + Vite

- Vite for fast HMR during development.
- No UI framework (no Material UI, Ant Design) — custom CSS with a consistent design system for full control over styling.
- `lucide-react` for icons: single stroke-width design language, tree-shakeable (only imported icons are bundled).

---

## Features

| Feature | Description |
|---------|-------------|
| RAG retrieval | FAISS-based dense retrieval with cosine similarity |
| Two-pass threshold | Fallback retrieval at relaxed threshold before deflecting to support |
| Multi-turn memory | Per-session conversation history (4 turns) with retrieval augmentation |
| User profile | Optional customer context injected into every prompt |
| Guardrails | Jailbreak detection (16 patterns), off-topic filter, PII output masking |
| Real-time updates | Upload JSON/CSV/TXT documents and query them immediately |
| Markdown rendering | Bullet lists, numbered steps, bold/italic rendered in UI |
| Response metadata | Per-response latency display and source document citations with confidence scores |
| Copy response | Clipboard copy on any bot response |
| Session management | UUID-scoped conversation state, explicit clear resets both frontend and backend |

---

## Data Pipeline

```
data/raw/*.json   ─┐
data/raw/*.xlsx   ─┤─► NustBankIngestor ─► bank_knowledge_chunks.json
                          │
                          └─► NustBankIndexer ─► bank_faiss_index.bin
                                                  bank_metadata.json
```

To rebuild the index after updating the knowledge base:

```bash
python src/vector_store.py
```

---

## Configuration

All parameters are centralised in `config/settings.py`:

```python
EMBEDDING_MODEL     = "Alibaba-NLP/gte-large-en-v1.5"
EMBEDDING_DIMENSION = 1024
TOP_K_RESULTS       = 5       # chunks retrieved per query
MIN_SCORE_THRESHOLD = 0.35    # strict threshold (fallback: 0.20)
LLM_PROVIDER        = "hf_api"
HF_MODEL_ID         = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS      = 512
TEMPERATURE         = 0.3
REPETITION_PENALTY  = 1.15
```

Environment variables (`.env`):

```
HF_API_TOKEN=your_huggingface_token
```

---

## Running the System

### Prerequisites

- Python 3.10+
- Node.js 18+
- HuggingFace account with API token

### Backend

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\Activate.ps1         # Windows
source venv/bin/activate           # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Build the vector index (first run only, or after KB updates)
python src/vector_store.py

# Start the API server
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Serves on http://localhost:5173
```

---

## Project Structure

```
.
├── backend/                 # FastAPI server implementation
├── config/                  # Global system settings
├── data/
│   ├── processed/           # Chunked dataset
│   ├── raw/                 # Original docs (Excel, JSON)
│   └── vector_store/        # FAISS index + metadata
├── docs/                    # Technical reports and specifications
├── frontend/                # React dashboard + styles
├── logs/                    # Standardized system logs
├── models/                  # Local model weights / adapters
├── scripts/                 # Data prep & training utilities (Fine-tuning, Cooking)
├── src/                     # Core RAG Application logic
│   ├── guardrails/          # Safety and PII logic
│   ├── ingest.py            # Ingestion logic
│   ├── rag/                 # RAG pipeline components
│   └── vector_store.py      # Embedding logic
├── tests/                   # Automated Pipeline & Guardrail tests
├── README.md
├── requirements.txt
└── .env
```

