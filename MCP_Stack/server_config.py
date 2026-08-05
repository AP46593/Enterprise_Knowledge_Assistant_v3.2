"""
MCP Stack Server Configuration.

Centralizes all non-secret configuration values for the MCP server, including
model settings, chunking parameters, retrieval tuning, and feature flags.

Priority: Values defined in this file take precedence. The .env file is loaded
only to provide secrets (API keys) or to override values NOT set here.
To change a setting, edit this file directly.

This module is imported by virtually all other modules in the MCP_Stack package.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# Environment Setup
# =============================================================================

# Base directory for all relative path resolution (MCP_Stack/)
BASE_DIR = Path(__file__).parent

# Load .env file — only provides values for keys NOT already set in os.environ.
# Since we set config values below AFTER this call, config file always wins.
load_dotenv(BASE_DIR / ".env", override=False)

# =============================================================================
# Base Paths
# =============================================================================

# Directory containing raw source documents to be ingested
KNOWLEDGE_SOURCE_DIR = str(BASE_DIR / "knowledge_source")

# Directory containing the processed knowledge base (ChromaDB, BM25 index, registry)
KNOWLEDGE_BASE_DIR = str(BASE_DIR / "knowledge_base")

# Directory for structured JSONL server logs
SERVER_LOGS_DIR = str(BASE_DIR / "Server_Logs")

# Directory for cached summaries and other computed artifacts
CACHE_DIR = str(BASE_DIR / "cache")

# =============================================================================
# Ollama Model Configuration
# =============================================================================

# Base URL for the Ollama API server
OLLAMA_BASE_URL = "http://localhost:11434"

# Default chat/generation model used for RAG, evaluation, summarization
DEFAULT_MODEL = "gpt-oss:120b-cloud"

# Embedding model for semantic search (must produce consistent vector dimensions)
EMBEDDING_MODEL = "nomic-embed-text"

# Vision model for OCR fallback when pytesseract is unavailable
# use llava:13b - CPU/GPU heavy or 
# use llava - relatively lighter model
# Keep blank - to skip OCR/images from knowledge_source
VISION_MODEL = "llava"

# =============================================================================
# Generation Parameters
# =============================================================================

# Maximum number of tokens to generate per LLM call
MAX_TOKENS = 2048

# Sampling temperature: higher = more creative, lower = more deterministic
TEMPERATURE = 0.7

# =============================================================================
# Chunking Parameters
# =============================================================================

# Maximum character count per text chunk
CHUNK_SIZE = 2000

# Character overlap between consecutive chunks for context continuity
CHUNK_OVERLAP = 200

# =============================================================================
# Retrieval Parameters
# =============================================================================

# Number of top results to return from hybrid search
RETRIEVAL_TOP_K = 5

# Weight for semantic similarity vs BM25 keyword score (0.0–1.0)
# semantic_weight=0.7 means 70% semantic + 30% BM25
SEMANTIC_WEIGHT = 0.7

# =============================================================================
# Feature Flags
# =============================================================================

# Use LLM-based PII detection in addition to regex patterns
PII_USE_LLM = True

# Auto-evaluate RAG responses with RAGAS-style metrics (LLM-as-judge)
ENABLE_RAGAS_EVAL = True

# =============================================================================
# Ground-Truth Generation
# =============================================================================

# Minimum number of Q&A pairs to generate per document
GROUND_TRUTH_MIN_QUESTIONS = 3

# Maximum number of Q&A pairs to generate per document
GROUND_TRUTH_MAX_QUESTIONS = 10

# =============================================================================
# MCP Server Network Settings
# =============================================================================

# Port the MCP server listens on for Streamable HTTP connections
MCP_SERVER_PORT = 8000

# URL path endpoint for MCP protocol communication
MCP_SERVER_ENDPOINT = "/mcp"

# =============================================================================
# Secrets (loaded from .env if not set above — API keys, tokens, etc.)
# =============================================================================

# Opik observability (server-side tracing)
OPIK_API_KEY = os.getenv("OPIK_API_KEY", "")
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE", "")
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME", "rag-chat-assistant")
