"""
Streamlit Chat UI for the Knowledge Chat Assistant.

Provides a conversational chat interface that communicates with the MCP server
to answer questions from the knowledge base with citations and evaluation scores.

The client:
- Injects truststore SSL certificates as its first operation
- Connects to the MCP Server via Streamable HTTP transport
- Maintains session state including conversation history
- Displays citations and source references in expandable sections
- Shows RAGAS evaluation scores when available
- Logs all operations as structured JSONL to Client_Logs/

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
"""

# FIRST OPERATION: Inject truststore SSL certificates so all subsequent
# HTTPS connections trust the system certificate store (required for corporate proxies).
import truststore

truststore.inject_into_ssl()

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from the project-root .env file
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=str(_env_path))

from client_config import ENABLE_OPIK_TRACING, MCP_SERVER_URL, ORCHESTRATOR_MODEL


# =============================================================================
# Structured JSONL Logging
# =============================================================================

CLIENT_LOGS_DIR = str(Path(__file__).parent / "Client_Logs")


class JSONLHandler(logging.Handler):
    """
    Logging handler that writes structured JSONL to Client_Logs/.

    Each log record is serialized as a single JSON line with timestamp,
    level, logger name, message, and optional exception traceback.
    A new log file is created per session (timestamped filename).
    """

    def __init__(self, log_dir: str):
        """
        Initialize the JSONL handler.

        Args:
            log_dir: Directory path where session log files will be written.
        """
        super().__init__()
        os.makedirs(log_dir, exist_ok=True)
        session_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_path = Path(log_dir) / f"session_{session_ts}.jsonl"
        self._file = open(self.log_path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        """
        Write a single log record as a JSON line.

        Args:
            record: The log record to serialize and write.
        """
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            # Include exception traceback if present
            if record.exc_info and record.exc_info[1]:
                entry["exception"] = traceback.format_exception(*record.exc_info)
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()
        except Exception:
            pass  # Logging must never crash the application

    def close(self) -> None:
        """Close the underlying file handle and release resources."""
        try:
            self._file.close()
        except Exception:
            pass
        super().close()


def _setup_logging() -> logging.Logger:
    """
    Configure the application logger with JSONL file and console handlers.

    Ensures handlers are only attached once (idempotent across Streamlit reruns).

    Returns:
        Configured logging.Logger instance for the streamlit_app module.
    """
    logger = logging.getLogger("streamlit_app")
    if logger.handlers:
        return logger  # Already configured (Streamlit reruns)

    logger.setLevel(logging.INFO)

    # Console handler for development visibility
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(console)

    # JSONL file handler for structured persistent logs
    jsonl_handler = JSONLHandler(CLIENT_LOGS_DIR)
    jsonl_handler.setLevel(logging.DEBUG)
    logger.addHandler(jsonl_handler)

    return logger


logger = _setup_logging()


# =============================================================================
# MCP Client Connection
# =============================================================================


async def _get_mcp_client():
    """
    Create and return an MCP client session connected to the server.

    Establishes a Streamable HTTP transport connection and initializes
    the MCP protocol handshake.

    Returns:
        An initialized ClientSession ready for tool calls.
    """
    from mcp.client.streamable_http import streamable_http_client
    from mcp import ClientSession

    read_stream, write_stream = await streamable_http_client(MCP_SERVER_URL).__aenter__()
    session = ClientSession(read_stream, write_stream)
    await session.__aenter__()
    await session.initialize()
    return session


async def _call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """
    Call an MCP tool on the server and return the parsed result.

    Opens a fresh connection per call (stateless), invokes the named tool
    with the provided arguments, and parses the response content blocks.

    Args:
        tool_name: Name of the MCP tool to invoke (e.g., "ask_question").
        arguments: Dictionary of arguments to pass to the tool.

    Returns:
        Parsed JSON response dict, or an error dict if parsing fails.
    """
    from mcp.client.streamable_http import streamable_http_client
    from mcp import ClientSession

    async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

            # Parse the result content blocks (MCP returns a list of content blocks)
            if result.content:
                for content_block in result.content:
                    if hasattr(content_block, "text"):
                        try:
                            return json.loads(content_block.text)
                        except json.JSONDecodeError:
                            # Non-JSON text response — wrap it as a simple answer
                            return {"success": True, "answer": content_block.text}
            return {"success": False, "error": "No response content received"}


def call_mcp_tool_sync(tool_name: str, arguments: dict) -> dict:
    """
    Synchronous wrapper for calling MCP tools from Streamlit's sync context.

    Handles the event loop complexities of calling async code from within
    Streamlit, which may already have a running event loop. Falls back through
    multiple strategies: thread pool executor, existing loop, or asyncio.run().

    Args:
        tool_name: Name of the MCP tool to invoke.
        arguments: Dictionary of arguments to pass to the tool.

    Returns:
        Parsed response dict from the MCP tool, or error dict on failure.
    """
    import asyncio

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Streamlit runs its own event loop — must use a separate thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, _call_mcp_tool(tool_name, arguments)
                    )
                    return future.result(timeout=120)
            else:
                return loop.run_until_complete(_call_mcp_tool(tool_name, arguments))
        except RuntimeError:
            # No event loop exists — safe to use asyncio.run()
            return asyncio.run(_call_mcp_tool(tool_name, arguments))
    except Exception as e:
        logger.error("MCP tool call failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


# =============================================================================
# Session State Management
# =============================================================================


def init_session_state() -> None:
    """
    Initialize Streamlit session state for conversation history.

    Ensures the 'messages' list and 'mcp_connected' flag exist in
    st.session_state. Safe to call on every rerun (idempotent).
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mcp_connected" not in st.session_state:
        st.session_state.mcp_connected = False


def add_message(role: str, content: str, metadata: dict = None) -> None:
    """
    Add a message to the session state conversation history.

    Args:
        role: The message role — "user" or "assistant".
        content: The text content of the message.
        metadata: Optional metadata dict (citations, scores, model info).
    """
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        message["metadata"] = metadata
    st.session_state.messages.append(message)


def get_messages() -> list:
    """
    Return the full conversation history from session state.

    Returns:
        List of message dicts with keys: role, content, timestamp, metadata.
    """
    return st.session_state.messages


# =============================================================================
# Citation Display
# =============================================================================


def render_citations(citations: list) -> None:
    """
    Render source document names in a compact expandable section.

    Shows only the unique document names that contributed to the response.

    Args:
        citations: List of citation dicts with keys: source_path,
                   document_title, chunk_index, relevant_passage.
    """
    if not citations:
        return

    # Get unique document names
    doc_names = list(dict.fromkeys(
        citation.get("document_title", "Unknown") for citation in citations
    ))

    with st.expander(f"📚 Sources ({len(doc_names)} documents)", expanded=False):
        for doc_name in doc_names:
            st.markdown(f"• **{doc_name}**")


# =============================================================================
# Evaluation Score Display
# =============================================================================


def render_evaluation_scores(scores: dict) -> None:
    """
    Display RAGAS evaluation scores in the Streamlit sidebar.

    Color-codes scores with emoji indicators:
        🟢 >= 0.7 (good), 🟡 >= 0.4 (acceptable), 🔴 < 0.4 (poor)

    Args:
        scores: Dictionary with metric names as keys and float scores as values.
    """
    if not scores:
        return

    with st.sidebar:
        st.subheader("📊 Response Quality (RAGAS)")

        metrics = [
            ("Faithfulness", scores.get("faithfulness")),
            ("Answer Relevancy", scores.get("answer_relevancy")),
            ("Context Precision", scores.get("context_precision")),
            ("Context Recall", scores.get("context_recall")),
        ]

        for metric_name, value in metrics:
            if value is not None:
                # Color code: green > 0.7, yellow > 0.4, red otherwise
                if value >= 0.7:
                    color = "🟢"
                elif value >= 0.4:
                    color = "🟡"
                else:
                    color = "🔴"
                st.metric(
                    label=f"{color} {metric_name}",
                    value=f"{value:.2f}",
                )


# =============================================================================
# Message Handling
# =============================================================================

# Common greetings and casual messages that don't need knowledge base lookup
GREETING_PATTERNS = {
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "howdy", "greetings", "what's up", "whats up", "sup", "yo",
    "how are you", "how's it going", "how are things",
    "thanks", "thank you", "bye", "goodbye", "see you",
}


def is_greeting(text: str) -> bool:
    """
    Check if user input is a casual greeting/chat message.

    Args:
        text: The user's input text.

    Returns:
        True if the message matches a known greeting pattern.
    """
    cleaned = text.strip().lower().rstrip("!?.,:;")
    return cleaned in GREETING_PATTERNS


def get_greeting_response(text: str) -> str:
    """
    Generate a friendly response to greetings while directing user to the knowledge base.

    Args:
        text: The user's greeting message.

    Returns:
        A friendly response string.
    """
    cleaned = text.strip().lower().rstrip("!?.,:;")

    if cleaned in {"thanks", "thank you"}:
        return (
            "You're welcome! Let me know if you have any other questions "
            "about the documents in the knowledge base. 📚"
        )
    elif cleaned in {"bye", "goodbye", "see you"}:
        return "Goodbye! Feel free to come back anytime you need help with the knowledge base. 👋"
    else:
        return (
            "Hello! 👋 I'm your Knowledge Chat Assistant. I can help you answer questions "
            "based on the documents in the knowledge base.\n\n"
            "Try asking me something like:\n"
            "- *What are the main modules in the platform?*\n"
            "- *Summarize the hiring assistant workflow*\n"
            "- *What scoring criteria are used?*\n\n"
            "💡 **Note:** I can only answer questions from the ingested documents."
        )


def send_question(user_input: str) -> dict:
    """
    Send user input via MCP tool call (ask_question) and return the response.

    Handles connection errors gracefully and returns a user-friendly error
    message if the MCP server is unreachable.

    Args:
        user_input: The user's natural language question.

    Returns:
        Response dict with keys: success, answer, citations, evaluation_scores.
    """
    logger.info("Sending question: %s", user_input[:100])

    try:
        response = call_mcp_tool_sync("ask_question", {"question": user_input})
        logger.info(
            "Received response: success=%s, has_citations=%s",
            response.get("success"),
            bool(response.get("citations")),
        )
        return response
    except Exception as e:
        logger.error("Error sending question: %s", e, exc_info=True)
        return {
            "success": False,
            "error": f"Failed to communicate with MCP server: {str(e)}",
            "answer": "I'm sorry, I couldn't connect to the knowledge base server. "
            "Please ensure the MCP server is running.",
        }


# =============================================================================
# Document List Helper
# =============================================================================


def _get_ingested_documents() -> list[str]:
    """
    Fetch the list of ingested documents from the MCP server.

    Calls the 'list_documents' MCP tool to get document paths.
    Returns a cached list to avoid repeated calls on each Streamlit rerun.

    Returns:
        List of document relative file paths currently in the knowledge base.
    """
    if "ingested_docs" in st.session_state:
        return st.session_state.ingested_docs

    try:
        response = call_mcp_tool_sync("list_documents", {})
        if response.get("success") and response.get("documents"):
            docs = [
                doc.get("file_path", doc.get("file_name", "Unknown"))
                for doc in response["documents"]
            ]
            st.session_state.ingested_docs = docs
            return docs
    except Exception as e:
        logger.debug("Could not fetch document list: %s", e)

    st.session_state.ingested_docs = []
    return []


def _render_doc_tree(doc_paths: list[str]) -> None:
    """
    Render documents in a folder-tree structure in the sidebar.

    Groups documents by their folder path and displays them hierarchically
    with folder icons for directories and document icons for files.

    Args:
        doc_paths: List of relative file paths (e.g., "Legacy docs/file.docx").
    """
    from collections import defaultdict

    # Build tree structure: folder -> list of filenames
    tree = defaultdict(list)
    root_files = []

    for path in sorted(doc_paths):
        # Normalize separators
        path = path.replace("\\", "/")
        # Strip any leading absolute or relative path up to and including "knowledge_source/"
        if "knowledge_source/" in path:
            path = path.split("knowledge_source/", 1)[1]
        elif "knowledge_source\\" in path:
            path = path.split("knowledge_source\\", 1)[1]

        parts = path.rsplit("/", 1)
        if len(parts) == 2:
            folder, filename = parts
            tree[folder].append(filename)
        else:
            root_files.append(parts[0])

    # Render root-level files first
    for f in root_files:
        st.caption(f"📄 {f}")

    # Render folders with their contents
    for folder in sorted(tree.keys()):
        st.markdown(f"**📁 {folder}**")
        for filename in sorted(tree[folder]):
            st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;📄 {filename}")


# =============================================================================
# Page Configuration and Layout
# =============================================================================


def main():
    """Main Streamlit application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Enterprise Knowledge Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("🤖 Enterprise Knowledge Assistant")
        st.markdown("---")
        st.markdown("**Knowledge Base Q&A**")
        st.caption(
            "Ask questions about documents in the knowledge base. "
            "Answers include citations to source documents."
        )
        st.markdown("---")

        # Ingested documents list (expandable, folder-structured)
        with st.expander("📄 Ingested Documents", expanded=False):
            docs = _get_ingested_documents()
            if docs:
                _render_doc_tree(docs)
            else:
                st.caption("_No documents ingested yet._")

        st.markdown("---")

        # Connection info
        st.subheader("⚙️ Configuration")
        st.caption(f"Server: `{MCP_SERVER_URL}`")
        st.caption(f"Model: `{ORCHESTRATOR_MODEL}`")
        st.caption(f"Opik Tracing: `{'Enabled' if ENABLE_OPIK_TRACING else 'Disabled'}`")

        st.markdown("---")

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main chat area
    st.title("💬 Document Q&A")

    # Display conversation history
    for message in get_messages():
        role = message["role"]
        content = message["content"]
        metadata = message.get("metadata", {})

        with st.chat_message(role):
            st.markdown(content)

            # Display citations for assistant messages
            if role == "assistant" and metadata.get("citations"):
                render_citations(metadata["citations"])

            # Display evaluation scores for assistant messages
            if role == "assistant" and metadata.get("evaluation_scores"):
                render_evaluation_scores(metadata["evaluation_scores"])

    # Chat input
    if user_input := st.chat_input("Ask a question about your documents..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add user message to history
        add_message("user", user_input)

        # Check if it's a casual greeting (no need to hit the knowledge base)
        if is_greeting(user_input):
            greeting_response = get_greeting_response(user_input)
            with st.chat_message("assistant"):
                st.markdown(greeting_response)
            add_message("assistant", greeting_response)
        else:
            # Send question to knowledge base and display response
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    response = send_question(user_input)

                if response.get("success"):
                    answer = response.get("answer", "No answer received.")
                    citations = response.get("citations", [])
                    evaluation_scores = response.get("evaluation_scores")

                    st.markdown(answer)

                    # Render citations
                    if citations:
                        render_citations(citations)

                    # Render evaluation scores in sidebar
                    if evaluation_scores:
                        render_evaluation_scores(evaluation_scores)

                    # Add assistant message to history with metadata
                    add_message(
                        "assistant",
                        answer,
                        metadata={
                            "citations": citations,
                            "evaluation_scores": evaluation_scores,
                            "model": response.get("metadata", {}).get("model"),
                        },
                    )
                else:
                    error_msg = response.get(
                        "answer",
                        response.get(
                            "error",
                            "An unexpected error occurred. Please try again.",
                        ),
                    )
                    st.error(error_msg)

                    # Add error response to history
                    add_message(
                        "assistant",
                        error_msg,
                        metadata={"error": response.get("error")},
                    )

        logger.info(
            "Conversation turn complete. Total messages: %d",
            len(get_messages()),
        )


if __name__ == "__main__":
    main()
