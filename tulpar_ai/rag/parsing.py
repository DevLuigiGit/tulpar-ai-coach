"""Backward-compatible imports for the document processing package."""

from parsing.cleaner import normalize_text
from parsing.parser import parse_document

__all__ = ["normalize_text", "parse_document"]
