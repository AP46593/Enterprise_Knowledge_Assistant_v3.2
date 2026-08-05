# Command List — RAG Chat Assistant

Run these commands in sequence from the project root: `L3June26_Assignment/`

## 1. Prerequisites — Pull Ollama Models

```bash
ollama pull gpt-oss:120b-cloud          # Chat/reasoning model
ollama pull nomic-embed-text            # Embedding model for vector search
ollama pull llava:13b                   # Vision model for image OCR fallback
ollama pull llava                       # smaller model for OCR
```

## 2. MCP Server Environment Setup

```bash
uv venv .mcpvenv --python 3.12          # Create server virtual environment
.mcpvenv\Scripts\activate               # Activate it (Windows)
uv pip install -r MCP_Stack/requirements_mcp.txt   # Install server dependencies
```

## 3. Streamlit Client Environment Setup

```bash
uv venv .venv --python 3.12             # Create client virtual environment
.venv\Scripts\activate                  # Activate it (Windows)
uv pip install -r requirements.txt      # Install client dependencies
```

## 4. Configure Environment Variables

```bash
copy .env.example .env                  # Create client env file from template
copy MCP_Stack\.env.example MCP_Stack\.env   # Create server env file from template
```

Edit both `.env` files with your actual API keys and settings.

## 5. Add Documents

Place documents (PDF, DOCX, TXT, CSV, XLSX, XML, PNG, JPG) into `MCP_Stack/knowledge_source/`.

## 6. Start the MCP Server

```bash
.mcpvenv\Scripts\activate               # Activate server environment
python -m MCP_Stack.mcp_server          # Start server (ingests docs on startup)
```

## 7. Start the Streamlit Client (separate terminal)

```bash
.venv\Scripts\activate                  # Activate client environment
streamlit run streamlit_app.py          # Launch chat UI at http://localhost:8501
```

## 8. Run Tests (optional)

```bash
.mcpvenv\Scripts\activate               # Tests use server environment
python -m pytest tests/ -v              # Run all tests
python -m pytest tests/test_property_*.py -v   # Property-based tests only
python -m pytest tests/test_unit_*.py -v       # Unit tests only
```

## 9. Freeze Dependencies (optional, for reproducibility)

```bash
uv pip freeze > requirements.lock                    # Client lockfile
uv pip freeze > MCP_Stack/requirements_mcp.lock      # Server lockfile
```
