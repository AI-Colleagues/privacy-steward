"""Output file writer."""

from __future__ import annotations
from pathlib import Path


def write_text(dest: Path, text: str) -> None:
    """Write *text* to *dest* as UTF-8, creating parent directories."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
