"""
Standalone script to generate ground-truth JSON for RAGAS evaluation.

This script creates question-answer-context datasets from source documents
that can later be used by run_ragas_eval.py to measure RAG pipeline quality.

Run this BEFORE starting the MCP server to pre-generate test data.

Usage:
    # Generate for a single document:
    python generate_ground_truth.py "Legacy docs/XLUTS1_documentation_v17.docx"

    # Generate for all documents in knowledge_source:
    python generate_ground_truth.py --all

Requirements:
    - Ollama must be running with the configured chat model
    - Run from the L3June26_Assignment directory
    - Activate the .mcpvenv virtual environment first

Output:
    - JSON files written to MCP_Stack/ground_truth_docs/
    - Each file contains question/answer/contexts entries for one document
"""

# Inject truststore SSL certificates before any network imports
import truststore
truststore.inject_into_ssl()

import sys
import os

# Ensure the project root is in the path so MCP_Stack package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from MCP_Stack.agents.evaluator_agent import generate_ground_truth
from MCP_Stack.server_config import KNOWLEDGE_SOURCE_DIR

# =============================================================================
# Constants
# =============================================================================

# File extensions supported by the document loader pipeline
SUPPORTED_EXTENSIONS = {"pdf", "docx", "txt", "csv", "xlsx", "xml", "png", "jpg", "jpeg", "tiff"}


# =============================================================================
# Helper Functions
# =============================================================================


def find_all_documents() -> list[str]:
    """
    Find all supported documents in the knowledge_source directory recursively.

    Walks the knowledge_source tree and returns relative paths for all files
    with extensions in SUPPORTED_EXTENSIONS.

    Returns:
        Sorted list of relative file paths (relative to knowledge_source root).
    """
    documents = []
    base = Path(KNOWLEDGE_SOURCE_DIR)
    for file_path in base.rglob("*"):
        if file_path.is_file() and file_path.suffix.lstrip(".").lower() in SUPPORTED_EXTENSIONS:
            # Store path relative to knowledge_source for portability
            rel_path = str(file_path.relative_to(base))
            documents.append(rel_path)
    return sorted(documents)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """
    Parse CLI arguments and generate ground-truth data for specified documents.

    Supports two modes:
        - Single document: pass the document's relative path as an argument
        - All documents: pass --all to process every supported file

    Prints progress and a final summary of successes/failures.
    """
    # Display usage if no arguments provided
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python generate_ground_truth.py "Legacy docs/XLUTS1_documentation_v17.docx"')
        print("  python generate_ground_truth.py --all")
        print()
        print("Available documents:")
        for doc in find_all_documents():
            print(f"  - {doc}")
        sys.exit(1)

    # Determine which documents to process
    if sys.argv[1] == "--all":
        documents = find_all_documents()
        if not documents:
            print("No supported documents found in knowledge_source/")
            sys.exit(1)
        print(f"Generating ground-truth for {len(documents)} documents...\n")
    else:
        documents = [sys.argv[1]]

    # Process each document and collect results
    results = []
    for doc_name in documents:
        print(f"Processing: {doc_name}")
        result = generate_ground_truth(doc_name)
        if result.success:
            print(f"  ✓ Generated {result.entry_count} Q&A pairs → {result.output_path}")
        else:
            print(f"  ✗ Failed: {result.error}")
        results.append(result)
        print()

    # Print final summary
    success_count = sum(1 for r in results if r.success)
    print(f"Done: {success_count}/{len(results)} documents processed successfully.")
    if success_count > 0:
        print(f"Output location: MCP_Stack/ground_truth_docs/")


if __name__ == "__main__":
    main()
