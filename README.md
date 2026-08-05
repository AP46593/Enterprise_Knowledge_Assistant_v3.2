# RAG Chat Assistant

A document Q&A Chat Assistant powered by Retrieval-Augmented Generation (RAG). Uses a hybrid retrieval system (semantic + keyword search) with an MCP (Model Context Protocol) server/client architecture, PII redaction, automated evaluation via RAGAS, and full observability tracing.

---

## Architecture

```
Streamlit Chat UI (Client - .venv)
    ↕ MCP Protocol (Streamable HTTP on localhost:8000)
MCP Server (FastMCP - .mcpvenv)
    ├── Tools: filesystem, doc_loader, chunker, ingest, retriever
    ├── Agents: RAG Agent, Summarizer, PII Redactor, Evaluator
    ├── Storage: ChromaDB (vector) + BM25 (keyword) + Registry
    └── External: Ollama (LLM + Embeddings), Opik (Observability)
```

## Project Structure

```
L3June26_Assignment/
├── MCP_Stack/                  # MCP Server (runs in .mcpvenv)
│   ├── agents/
│   │   ├── rag_agent.py           # LangGraph RAG agent (retrieve → generate)
│   │   ├── summarizer_agent.py    # Iterative document summarization with caching
│   │   ├── pii_redactor.py        # Regex + optional LLM-based PII detection
│   │   └── evaluator_agent.py     # RAGAS evaluation + ground-truth generator
│   ├── tools/
│   │   ├── doc_loader.py          # Multi-format document loading (PDF/DOCX/TXT/CSV/XLSX/XML/images)
│   │   ├── chunker.py             # Semantic chunking with metadata
│   │   ├── ingest.py              # Ingestion pipeline + document registry
│   │   ├── retriever.py           # Hybrid search (ChromaDB + BM25 + reranking)
│   │   └── filesystem.py          # Sandboxed file browsing
│   ├── mcp_server.py              # FastMCP server entry point
│   ├── config.py                  # Server configuration
│   ├── .env.example               # Server secrets template
│   ├── requirements_mcp.txt       # Server dependencies
│   ├── knowledge_source/          # Drop documents here for ingestion
│   ├── knowledge_base/            # ChromaDB + BM25 index + registry.json (auto-generated)
│   ├── Server_Logs/               # Per-session JSONL logs
│   └── cache/                     # Summarizer cache (by content hash)
├── tests/                     # All tests
│   ├── test_property_*.py         # Property-based tests (Hypothesis)
│   ├── test_unit_*.py             # Unit tests
│   └── test_integration_*.py      # Integration tests
├── Client_Logs/               # Client JSONL logs
├── streamlit_app.py           # Streamlit chat UI (runs in .venv)
├── config.py                  # Client configuration
├── .env.example               # Client secrets template
├── requirements.txt           # Client dependencies
├── test_tools.py              # Manual test stub for tools/agents
└── README.md
```

---

## Prerequisites

| Dependency | Purpose |
|------------|---------|
| **Python 3.12** | Runtime (RAGAS has compatibility issues with 3.14) |
| **[uv](https://docs.astral.sh/uv/getting-started/installation/)** | Package manager (replaces pip) |
| **[Ollama](https://ollama.ai/)** | Local/cloud LLM serving |
| **Tesseract OCR** *(optional)* | Primary OCR for images; if unavailable, falls back to `gemma4:31b-cloud` vision model |

---

## Setup

### 1. Pull Required Ollama Models

```bash
# Chat model (cloud-hosted, no local GPU needed)
ollama pull gpt-oss:120b-cloud 

# Embedding model
ollama pull nomic-embed-text

# Vision model (OCR fallback — cloud-hosted, no local GPU needed)
ollama pull gemma4:31b-cloud
```

### 2. Create Virtual Environments

**MCP Server (.mcpvenv):**

```bash
uv venv .mcpvenv --python 3.12

# Windows
.mcpvenv\Scripts\activate

# Linux/macOS
source .mcpvenv/bin/activate

uv pip install -r MCP_Stack/requirements_mcp.txt
```

**Streamlit Client (.venv):**

```bash
uv venv .venv --python 3.12

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

uv pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy templates
cp .env.example .env
cp MCP_Stack/.env.example MCP_Stack/.env
```

Edit each `.env` file with your actual values:

**Client `.env`:**
```dotenv
OLLAMA_BASE_URL=http://localhost:11434
ORCHESTRATOR_MODEL=gpt-oss:120b-cloud
MCP_SERVER_URL=http://localhost:8000/mcp
ENABLE_OPIK_TRACING=false
OPIK_API_KEY=<your-key>
OPIK_WORKSPACE=<your-workspace>
OPIK_PROJECT_NAME=rag-chat-assistant
```

**Server `MCP_Stack/.env`:**
```dotenv
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=gpt-oss:120b-cloud
EMBEDDING_MODEL=nomic-embed-text
VISION_MODEL=gemma4:31b-cloud
CHUNK_SIZE=2000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5
SEMANTIC_WEIGHT=0.7
PII_USE_LLM=false
ENABLE_RAGAS_EVAL=false
MCP_SERVER_PORT=8000
```

### 4. Add Documents to Knowledge Source

Place your documents (PDF, DOCX, TXT, CSV, XLSX, XML, or images) into:

```
MCP_Stack/knowledge_source/
```

These will be automatically ingested when the MCP server starts.

---

## Running the Application

### Step 1: Start the MCP Server

Open a terminal and activate the server environment:

```bash
# Windows
.mcpvenv\Scripts\activate

# Linux/macOS
source .mcpvenv/bin/activate

# Start the server
python -m MCP_Stack.mcp_server
```

On startup, the server will:
1. Inject SSL certificates (truststore)
2. Load existing knowledge base from disk
3. Scan `knowledge_source/` and ingest any new or modified documents
4. Skip unchanged documents (based on content hash)
5. Register all tools and agents
6. Serve MCP protocol on `http://localhost:8000/mcp`

> **Note:** Documents added to `knowledge_source/` while the server is running will NOT be auto-detected. Restart the server to ingest new files.

### Step 2: Start the Streamlit Client

Open a **separate terminal** and activate the client environment:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# Start the UI
streamlit run streamlit_app.py
```

The chat UI will open in your browser (typically at `http://localhost:8501`).

---

## Usage

### Asking Questions

Type your question in the chat input. The RAG agent will:
1. Search the knowledge base using hybrid retrieval (semantic + keyword)
2. Generate an answer with citations to source documents
3. Display RAGAS evaluation scores (if enabled)

### Document Management

Through the chat interface you can:
- **Browse files** — list and inspect documents in `knowledge_source/`
- **Ingest manually** — force re-ingest of a specific file or all files
- **List documents** — see all ingested documents with metadata
- **Delete documents** — remove a document from the knowledge base
- **Summarize** — get a concise summary of a long document

### Ground-Truth Test Data Generation

Generate evaluation test data from your documents:
1. Provide a document name from `knowledge_source/`
2. The system generates question-answer pairs with context passages
3. Output is saved as JSON for use with RAGAS evaluation (faithfulness, answer relevancy, context precision, context recall)

---

## Configuration Reference

### Server Configuration (`MCP_Stack/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `DEFAULT_MODEL` | `gpt-oss:120b-cloud` | Chat model for answer generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for vector search |
| `VISION_MODEL` | `gemma4:31b-cloud` | Cloud vision model (OCR fallback) |
| `MAX_TOKENS` | `2048` | Max tokens for generated responses |
| `TEMPERATURE` | `0.7` | LLM temperature |
| `CHUNK_SIZE` | `2000` | Characters per chunk (~500 tokens) |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |
| `SEMANTIC_WEIGHT` | `0.7` | Semantic vs keyword balance (0.7 = 70% semantic) |
| `PII_USE_LLM` | `false` | Enable LLM-based PII detection (slower, catches more) |
| `ENABLE_RAGAS_EVAL` | `false` | Auto-evaluate responses with RAGAS |
| `MCP_SERVER_PORT` | `8000` | Server port |

### Client Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `ORCHESTRATOR_MODEL` | `gpt-oss:120b-cloud` | Model for client-side orchestration |
| `MCP_SERVER_URL` | `http://localhost:8000/mcp` | MCP server endpoint |
| `ENABLE_OPIK_TRACING` | `false` | Enable Opik observability tracing |

---

## Running Tests

```bash
# Activate the server environment (has all dependencies)
# Windows
.mcpvenv\Scripts\activate

# Linux/macOS
source .mcpvenv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run only property-based tests
python -m pytest tests/test_property_*.py -v

# Run only unit tests
python -m pytest tests/test_unit_*.py -v

# Run a specific test file
python -m pytest tests/test_unit_chunker.py -v
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Protocol | MCP over Streamable HTTP | Standardized tool/agent interface; single endpoint |
| Agent Framework | LangGraph | Stateful graph workflows with conditional routing |
| Vector Store | ChromaDB (persistent) | Local file-based; no external service needed |
| Keyword Search | rank-bm25 (BM25Okapi) | Lightweight in-process; complements semantic search |
| Embedding | nomic-embed-text via Ollama | Dedicated embedding model; local inference |
| OCR | Tesseract → gemma4:31b-cloud fallback | Tesseract is fast; cloud vision is available everywhere |
| Observability | Opik (by Comet) | Native LangChain callback integration |
| Evaluation | RAGAS | Standard RAG evaluation framework |
| SSL | truststore | Corporate proxy support via Windows cert store |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SSL errors behind corporate proxy | Ensure `truststore` is installed and imported first in entry points |
| Ollama connection refused | Verify Ollama is running: `ollama list` |
| Empty OCR results | Install Tesseract, or ensure `gemma4:31b-cloud` is available via `ollama pull gemma4:31b-cloud` |
| MCP connection timeout | Check that the server is running on the configured port (default 8000) |
| Documents not appearing after adding | Restart the MCP server — ingestion only happens at startup |
| RAGAS scores not showing | Set `ENABLE_RAGAS_EVAL=true` in `MCP_Stack/.env` |

---

## Supported Document Formats

| Format | Extensions | Method |
|--------|-----------|--------|
| PDF | `.pdf` | pypdf + pdfplumber fallback |
| Word | `.docx` | python-docx |
| Plain Text | `.txt` | Direct read with encoding detection |
| CSV | `.csv` | pandas |
| Excel | `.xlsx` | openpyxl via pandas |
| XML | `.xml` | xml.etree + lxml fallback |
| Images | `.png`, `.jpg`, `.jpeg`, `.tiff` | Tesseract OCR → gemma4:31b-cloud fallback |
