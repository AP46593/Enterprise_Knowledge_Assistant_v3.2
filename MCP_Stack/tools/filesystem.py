"""
Filesystem Tools Module.

Provides sandboxed file browsing within the knowledge_source directory.
All paths are resolved and validated against the knowledge_source root.
Path traversal attempts (e.g., ../) are rejected with an access denied error.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from MCP_Stack.server_config import KNOWLEDGE_SOURCE_DIR

logger = logging.getLogger(__name__)

# Resolve the knowledge_source root to an absolute path for sandbox validation
_KNOWLEDGE_SOURCE_ROOT = Path(KNOWLEDGE_SOURCE_DIR).resolve()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FileEntry:
    """
    A single file or directory entry in a listing.

    Attributes:
        name: Name of the file or directory.
        is_directory: True if entry is a directory, False for files.
        size: File size in bytes (0 for directories).
        file_type: Lowercase file extension without dot (empty for directories).
    """

    name: str
    is_directory: bool
    size: int = 0
    file_type: str = ""


@dataclass
class DirectoryListing:
    """
    Result from listing a directory.

    Attributes:
        path: The requested path (as provided by the caller).
        entries: List of FileEntry objects in the directory.
        success: Whether the listing operation succeeded.
        error: Error message if the operation failed (None on success).
    """

    path: str
    entries: list[FileEntry] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


@dataclass
class FileContent:
    """
    Result from reading a file.

    Attributes:
        file_path: The requested file path.
        content: Text content of the file (empty string on failure).
        success: Whether the read operation succeeded.
        error: Error message if the operation failed (None on success).
    """

    file_path: str
    content: str = ""
    success: bool = True
    error: Optional[str] = None


@dataclass
class FileMatch:
    """
    A file matching a search query.

    Attributes:
        file_path: Absolute path to the matched file.
        file_name: Filename with extension.
        relative_path: Path relative to knowledge_source root.
    """

    file_path: str
    file_name: str
    relative_path: str


@dataclass
class FileInfo:
    """
    Metadata about a file.

    Attributes:
        file_path: The requested file path.
        file_name: Filename with extension.
        file_type: Lowercase extension without dot.
        size: File size in bytes.
        modified_date: Last modification timestamp.
        success: Whether the operation succeeded.
        error: Error message if the operation failed (None on success).
    """

    file_path: str
    file_name: str
    file_type: str
    size: int = 0
    modified_date: Optional[datetime] = None
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# Security: Path Validation
# =============================================================================


class AccessDeniedError(Exception):
    """Raised when a path resolves outside the knowledge_source sandbox."""

    pass


def _validate_path(path: str) -> Path:
    """
    Validate that a path resolves within the knowledge_source directory.

    Resolves the path to an absolute path and checks that it is within
    or equal to the knowledge_source root. Rejects path traversal attempts.

    Args:
        path: The path string to validate.

    Returns:
        The resolved absolute Path object.

    Raises:
        AccessDeniedError: If the path resolves outside knowledge_source.
    """
    # Handle relative paths by joining with knowledge_source root
    target = Path(path)
    if not target.is_absolute():
        target = _KNOWLEDGE_SOURCE_ROOT / target

    # Resolve to canonical absolute path (resolves .., symlinks, etc.)
    resolved = target.resolve()

    # Check that the resolved path is within or equal to the sandbox root
    try:
        resolved.relative_to(_KNOWLEDGE_SOURCE_ROOT)
    except ValueError:
        raise AccessDeniedError(
            f"Access denied: path '{path}' resolves outside the knowledge_source directory"
        )

    return resolved


# =============================================================================
# Public API
# =============================================================================


def list_directory(path: str = "") -> DirectoryListing:
    """
    List files and subdirectories at the given path within knowledge_source.

    Args:
        path: Relative path within knowledge_source. Empty string or "."
              lists the root of knowledge_source.

    Returns:
        DirectoryListing with entries for files and subdirectories,
        or an error if the path is invalid or outside the sandbox.
    """
    try:
        resolved = _validate_path(path)
    except AccessDeniedError as e:
        logger.warning(f"Access denied for list_directory: {path}")
        return DirectoryListing(path=path, success=False, error=str(e))

    if not resolved.exists():
        return DirectoryListing(
            path=path,
            success=False,
            error=f"Directory not found: '{path}'",
        )

    if not resolved.is_dir():
        return DirectoryListing(
            path=path,
            success=False,
            error=f"Path is not a directory: '{path}'",
        )

    entries = []
    try:
        for item in sorted(resolved.iterdir()):
            entry = FileEntry(
                name=item.name,
                is_directory=item.is_dir(),
                size=item.stat().st_size if item.is_file() else 0,
                file_type=item.suffix.lstrip(".").lower() if item.is_file() else "",
            )
            entries.append(entry)
    except PermissionError as e:
        logger.error(f"Permission error listing directory '{path}': {e}")
        return DirectoryListing(
            path=path,
            success=False,
            error=f"Permission denied reading directory: '{path}'",
        )
    except OSError as e:
        logger.error(f"OS error listing directory '{path}': {e}")
        return DirectoryListing(
            path=path,
            success=False,
            error=f"Error reading directory: {e}",
        )

    return DirectoryListing(path=path, entries=entries, success=True)


def read_file(file_path: str) -> FileContent:
    """
    Read the raw content of a file within knowledge_source.

    Args:
        file_path: Relative or absolute path to the file.

    Returns:
        FileContent with the file's text content, or an error
        if the path is invalid, outside the sandbox, or unreadable.
    """
    try:
        resolved = _validate_path(file_path)
    except AccessDeniedError as e:
        logger.warning(f"Access denied for read_file: {file_path}")
        return FileContent(file_path=file_path, success=False, error=str(e))

    if not resolved.exists():
        return FileContent(
            file_path=file_path,
            success=False,
            error=f"File not found: '{file_path}'",
        )

    if not resolved.is_file():
        return FileContent(
            file_path=file_path,
            success=False,
            error=f"Path is not a file: '{file_path}'",
        )

    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to latin-1 for binary-ish text files
        try:
            content = resolved.read_text(encoding="latin-1")
        except Exception as e:
            return FileContent(
                file_path=file_path,
                success=False,
                error=f"Unable to read file: {e}",
            )
    except PermissionError as e:
        logger.error(f"Permission error reading file '{file_path}': {e}")
        return FileContent(
            file_path=file_path,
            success=False,
            error=f"Permission denied: '{file_path}'",
        )
    except OSError as e:
        logger.error(f"OS error reading file '{file_path}': {e}")
        return FileContent(
            file_path=file_path,
            success=False,
            error=f"Error reading file: {e}",
        )

    return FileContent(file_path=file_path, content=content, success=True)


def search_files(query: str) -> list[FileMatch]:
    """
    Search for files by name/pattern within knowledge_source using glob.

    Args:
        query: A glob pattern (e.g., "*.pdf", "report*", "**/*.txt").
              If the query doesn't contain glob characters, it is treated
              as a substring search of filenames.

    Returns:
        List of FileMatch objects for files matching the query.
    """
    matches = []

    # Determine if the query is a glob pattern or a plain substring
    is_glob = any(c in query for c in ("*", "?", "[", "]"))

    if is_glob:
        # Use rglob for recursive patterns or glob for simple ones
        if "**" in query:
            pattern = query
        else:
            # Make the pattern recursive by default
            pattern = f"**/{query}"

        try:
            for match_path in _KNOWLEDGE_SOURCE_ROOT.glob(pattern):
                if match_path.is_file():
                    relative = match_path.relative_to(_KNOWLEDGE_SOURCE_ROOT)
                    matches.append(
                        FileMatch(
                            file_path=str(match_path),
                            file_name=match_path.name,
                            relative_path=str(relative),
                        )
                    )
        except (OSError, ValueError) as e:
            logger.error(f"Error during glob search '{query}': {e}")
    else:
        # Substring search: find files whose name contains the query
        query_lower = query.lower()
        try:
            for match_path in _KNOWLEDGE_SOURCE_ROOT.rglob("*"):
                if match_path.is_file() and query_lower in match_path.name.lower():
                    relative = match_path.relative_to(_KNOWLEDGE_SOURCE_ROOT)
                    matches.append(
                        FileMatch(
                            file_path=str(match_path),
                            file_name=match_path.name,
                            relative_path=str(relative),
                        )
                    )
        except (OSError, ValueError) as e:
            logger.error(f"Error during substring search '{query}': {e}")

    # Sort by relative path for deterministic output
    matches.sort(key=lambda m: m.relative_path)
    return matches


def get_file_info(file_path: str) -> FileInfo:
    """
    Get metadata about a file within knowledge_source.

    Args:
        file_path: Relative or absolute path to the file.

    Returns:
        FileInfo with size, modification date, and file type, or an error
        if the path is invalid or outside the sandbox.
    """
    try:
        resolved = _validate_path(file_path)
    except AccessDeniedError as e:
        logger.warning(f"Access denied for get_file_info: {file_path}")
        return FileInfo(
            file_path=file_path,
            file_name="",
            file_type="",
            success=False,
            error=str(e),
        )

    if not resolved.exists():
        return FileInfo(
            file_path=file_path,
            file_name="",
            file_type="",
            success=False,
            error=f"File not found: '{file_path}'",
        )

    if not resolved.is_file():
        return FileInfo(
            file_path=file_path,
            file_name=resolved.name,
            file_type="directory" if resolved.is_dir() else "",
            success=False,
            error=f"Path is not a file: '{file_path}'",
        )

    try:
        stat = resolved.stat()
        modified_date = datetime.fromtimestamp(stat.st_mtime)
        size = stat.st_size
    except OSError as e:
        logger.error(f"Error getting file info for '{file_path}': {e}")
        return FileInfo(
            file_path=file_path,
            file_name=resolved.name,
            file_type="",
            success=False,
            error=f"Error getting file info: {e}",
        )

    return FileInfo(
        file_path=file_path,
        file_name=resolved.name,
        file_type=resolved.suffix.lstrip(".").lower(),
        size=size,
        modified_date=modified_date,
        success=True,
    )
