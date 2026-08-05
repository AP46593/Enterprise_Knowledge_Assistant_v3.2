"""
Client Configuration for the RAG Chat Assistant.

Centralizes non-secret configuration values for the Streamlit chat client.

Priority: Values defined in this file take precedence. The .env file is loaded
only to provide secrets (API keys) or to override values NOT set here.
To change a setting, edit this file directly.

This module is imported by streamlit_app.py to configure the client's
connection to the MCP server and enable observability features.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# =============================================================================
# Environment Setup
# =============================================================================

# Load .env from project root — only provides values for keys NOT already set.
load_dotenv(Path(__file__).parent / ".env", override=False)

# =============================================================================
# Ollama Configuration
# =============================================================================

# Base URL for the Ollama API (used by the orchestrator model)
OLLAMA_BASE_URL = "http://localhost:11434"

# Model used by the client-side orchestrator for routing/planning
ORCHESTRATOR_MODEL = "gpt-oss:120b-cloud"

# =============================================================================
# MCP Server Connection
# =============================================================================

# Full URL (including path) to the MCP server's Streamable HTTP endpoint
MCP_SERVER_URL = "http://localhost:8000/mcp"

# =============================================================================
# Observability
# =============================================================================

# Enable Opik tracing for LLM call observability and debugging
ENABLE_OPIK_TRACING = True

# =============================================================================
# Secrets (loaded from .env — API keys, tokens, etc.)
# =============================================================================

OPIK_API_KEY = os.getenv("OPIK_API_KEY", "")
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE", "")
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME", "rag-chat-assistant")
