"""Tests for the text writer."""

from __future__ import annotations
from pathlib import Path
from privacy_steward.writer import write_text


def test_write_text_creates_parent_directories(tmp_path: Path) -> None:
    dest = tmp_path / "nested" / "file.txt"

    write_text(dest, "hello")

    assert dest.read_text(encoding="utf-8") == "hello"
