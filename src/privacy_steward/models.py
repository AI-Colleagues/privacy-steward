"""Data models for privacy-steward."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EntitySpan:
    """A detected PII entity span within a piece of text."""

    start: int
    end: int
    entity_type: str
    score: float
    word: str
