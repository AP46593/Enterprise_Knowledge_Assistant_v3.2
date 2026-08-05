"""
PII Redactor Agent.

Detects and redacts personally identifiable information from text using
regex patterns and optional LLM-based detection via ChatOllama.

Supported PII categories:
- Email addresses
- Phone numbers (multiple formats)
- Social Security Numbers (XXX-XX-XXXX)
- Credit card numbers (with Luhn validation)
- Names (pattern-based and optional LLM detection)
"""

import logging
import re
from dataclasses import dataclass, field

from MCP_Stack.server_config import DEFAULT_MODEL, OLLAMA_BASE_URL, PII_USE_LLM

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RedactionResult:
    """
    Result from PII redaction.

    Attributes:
        redacted_text: The input text with PII replaced by placeholders.
        entity_counts: Count of detected entities per category (email, phone, etc.).
        total_redacted: Total number of PII entities redacted across all categories.
    """

    redacted_text: str
    entity_counts: dict[str, int] = field(default_factory=dict)
    total_redacted: int = 0


# =============================================================================
# Regex Patterns
# =============================================================================

# Email: standard RFC-5322 simplified pattern
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# Phone numbers: multiple common US formats
# Matches: (123) 456-7890, 123-456-7890, 123.456.7890, +1 123-456-7890,
#          +1(123)456-7890
PHONE_PATTERN = re.compile(
    r"(?<!\d)"  # Negative lookbehind: not preceded by a digit
    r"(?:"
    r"\+?1[\s.\-]?)?"  # Optional US country code (+1)
    r"(?:"
    r"\(\d{3}\)[\s.\-]?\d{3}[\s.\-]?\d{4}"  # (123) 456-7890
    r"|"
    r"\d{3}[\s.\-]\d{3}[\s.\-]\d{4}"  # 123-456-7890 or 123.456.7890
    r")"
    r"(?!\d)"  # Negative lookahead: not followed by a digit
)

# SSN: XXX-XX-XXXX format with validation rules
# Excludes invalid ranges: 000, 666, 9xx in area; 00 in group; 0000 in serial
SSN_PATTERN = re.compile(
    r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
)

# Credit card numbers: 13-19 digits, optionally separated by spaces or dashes
# Covers Visa, MasterCard, Amex, Discover, etc.
CC_PATTERN = re.compile(
    r"\b(?:\d[ \-]?){12,18}\d\b"
)

# Name patterns: common honorific title + capitalized name sequences
# This is a heuristic pattern; LLM detection improves coverage for untitled names.
NAME_PATTERN = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
)


# =============================================================================
# Luhn Validation
# =============================================================================


def _luhn_check(number: str) -> bool:
    """
    Validate a number string using the Luhn algorithm (mod-10 checksum).

    Used to distinguish actual credit card numbers from random digit sequences.

    Args:
        number: String potentially containing a credit card number (digits,
                spaces, and dashes allowed).

    Returns:
        True if the number passes the Luhn checksum and has 13-19 digits.
    """
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    # Luhn algorithm: double every second digit from the right
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# =============================================================================
# Regex-Based Redaction Functions
# =============================================================================


def _redact_emails(text: str) -> tuple[str, int]:
    """
    Replace email addresses with [EMAIL_REDACTED] placeholder.

    Args:
        text: Input text to scan.

    Returns:
        Tuple of (modified text, count of emails redacted).
    """
    matches = EMAIL_PATTERN.findall(text)
    count = len(matches)
    if count > 0:
        text = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
    return text, count


def _redact_phones(text: str) -> tuple[str, int]:
    """
    Replace phone numbers with [PHONE_REDACTED] placeholder.

    Args:
        text: Input text to scan.

    Returns:
        Tuple of (modified text, count of phone numbers redacted).
    """
    matches = PHONE_PATTERN.findall(text)
    count = len(matches)
    if count > 0:
        text = PHONE_PATTERN.sub("[PHONE_REDACTED]", text)
    return text, count


def _redact_ssns(text: str) -> tuple[str, int]:
    """
    Replace Social Security Numbers with [SSN_REDACTED] placeholder.

    Args:
        text: Input text to scan.

    Returns:
        Tuple of (modified text, count of SSNs redacted).
    """
    matches = SSN_PATTERN.findall(text)
    count = len(matches)
    if count > 0:
        text = SSN_PATTERN.sub("[SSN_REDACTED]", text)
    return text, count


def _redact_credit_cards(text: str) -> tuple[str, int]:
    """
    Replace credit card numbers with [CC_REDACTED] placeholder after Luhn validation.

    Only redacts sequences that pass the Luhn checksum to reduce false positives.
    Processes matches in reverse order to preserve string indices.

    Args:
        text: Input text to scan.

    Returns:
        Tuple of (modified text, count of credit cards redacted).
    """
    count = 0
    matches = list(CC_PATTERN.finditer(text))
    # Process in reverse to preserve indices
    for match in reversed(matches):
        raw = match.group()
        if _luhn_check(raw):
            text = text[: match.start()] + "[CC_REDACTED]" + text[match.end() :]
            count += 1
    return text, count


def _redact_names(text: str) -> tuple[str, int]:
    """
    Replace name patterns (honorific title + name) with [NAME_REDACTED] placeholder.

    Detects patterns like "Mr. John Smith", "Dr. Jane Doe", etc.

    Args:
        text: Input text to scan.

    Returns:
        Tuple of (modified text, count of names redacted).
    """
    matches = NAME_PATTERN.findall(text)
    count = len(matches)
    if count > 0:
        text = NAME_PATTERN.sub("[NAME_REDACTED]", text)
    return text, count


# =============================================================================
# LLM-Based Detection
# =============================================================================


def _redact_with_llm(text: str) -> tuple[str, int]:
    """
    Use ChatOllama to detect additional PII entities beyond regex capabilities.

    The LLM is prompted to identify person names that were not caught by the
    regex-based NAME_PATTERN (e.g., names without honorific titles). The LLM
    returns a JSON list of name strings which are then replaced in the text.

    Args:
        text: Input text (may already contain some [*_REDACTED] placeholders).

    Returns:
        Tuple of (modified text, count of additional names redacted by LLM).
    """
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=DEFAULT_MODEL,
            temperature=0.0,
        )

        prompt = (
            "You are a PII detection assistant. Analyze the following text and "
            "identify any person names that are NOT already redacted (not inside "
            "square bracket placeholders like [NAME_REDACTED]). "
            "Return ONLY a JSON list of the exact name strings found. "
            "If no names are found, return an empty list [].\n\n"
            f"Text:\n{text}\n\n"
            "Response (JSON list only):"
        )

        response = llm.invoke(prompt)
        content = response.content.strip()

        # Parse the JSON list of names from LLM response
        import json

        # Handle potential markdown code block wrapping
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        names = json.loads(content)
        if not isinstance(names, list):
            return text, 0

        count = 0
        for name in names:
            if isinstance(name, str) and name.strip() and name in text:
                text = text.replace(name, "[NAME_REDACTED]")
                count += 1

        return text, count

    except Exception as e:
        logger.warning("LLM-based PII detection failed, falling back to regex only: %s", e)
        return text, 0


# =============================================================================
# Public API
# =============================================================================


def redact(text: str, use_llm: bool = False) -> RedactionResult:
    """
    Detect and replace PII with redaction placeholders.

    Applies regex-based detection for emails, phone numbers, SSNs, and
    credit card numbers. Optionally invokes LLM for additional name/entity
    detection beyond what regex patterns can capture.

    Args:
        text: Input document text.
        use_llm: Whether to use LLM-based detection in addition to regex.
                 Defaults to False; overridden by PII_USE_LLM config if not
                 explicitly set.

    Returns:
        RedactionResult with redacted text, entity counts per category,
        and total redacted count.
    """
    if not text:
        return RedactionResult(redacted_text=text, entity_counts={}, total_redacted=0)

    entity_counts: dict[str, int] = {}

    # Apply redaction in order: SSNs first (to avoid partial matches with phones),
    # then credit cards, emails, phones, and finally names.
    text, ssn_count = _redact_ssns(text)
    if ssn_count > 0:
        entity_counts["ssn"] = ssn_count

    text, cc_count = _redact_credit_cards(text)
    if cc_count > 0:
        entity_counts["credit_card"] = cc_count

    text, email_count = _redact_emails(text)
    if email_count > 0:
        entity_counts["email"] = email_count

    text, phone_count = _redact_phones(text)
    if phone_count > 0:
        entity_counts["phone"] = phone_count

    text, name_count = _redact_names(text)
    if name_count > 0:
        entity_counts["name"] = name_count

    # Optional LLM-based detection for additional entities
    resolve_use_llm = use_llm or PII_USE_LLM
    if resolve_use_llm:
        text, llm_name_count = _redact_with_llm(text)
        if llm_name_count > 0:
            entity_counts["name"] = entity_counts.get("name", 0) + llm_name_count

    total_redacted = sum(entity_counts.values())

    # Log redaction summary
    if total_redacted > 0:
        logger.info(
            "PII redaction complete: %d entities redacted. Breakdown: %s",
            total_redacted,
            entity_counts,
        )
    else:
        logger.debug("PII redaction complete: no entities detected.")

    return RedactionResult(
        redacted_text=text,
        entity_counts=entity_counts,
        total_redacted=total_redacted,
    )
