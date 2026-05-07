"""Unit tests for the path resolver."""

from __future__ import annotations
import os
from pathlib import Path
import pytest
from privacy_steward.resolver import output_root, resolve


# ---------------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------------


def test_resolve_file_default_output(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text("hello")
    pairs = resolve(src)
    assert len(pairs) == 1
    assert pairs[0][0] == src
    assert pairs[0][1] == tmp_path / "notes.redacted.txt"


def test_resolve_file_explicit_output(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text("hello")
    dest = tmp_path / "out" / "clean.txt"
    pairs = resolve(src, dest)
    assert pairs[0][1] == dest.resolve()


def test_resolve_file_explicit_output_directory(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text("hello")
    out = tmp_path / "clean"
    out.mkdir()
    pairs = resolve(src, out)
    assert pairs[0][1] == out.resolve() / "notes.redacted.txt"


def test_resolve_file_rejects_non_txt_input(tmp_path: Path) -> None:
    src = tmp_path / "notes.md"
    src.write_text("hello")

    with pytest.raises(ValueError, match="must be a .txt file"):
        resolve(src)


def test_resolve_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve(tmp_path / "nonexistent.txt")


# ---------------------------------------------------------------------------
# Directory mode
# ---------------------------------------------------------------------------


def test_resolve_dir_default_output_name(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("a")
    (corpus / "b.txt").write_text("b")
    pairs = resolve(corpus)
    out_root = tmp_path / "corpus_redacted"
    assert len(pairs) == 2
    dests = {p[1] for p in pairs}
    assert out_root / "a.redacted.txt" in dests
    assert out_root / "b.redacted.txt" in dests


def test_resolve_dir_explicit_output(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("a")
    out = tmp_path / "clean"
    pairs = resolve(corpus, out)
    assert pairs[0][1] == out / "a.redacted.txt"


def test_resolve_dir_mirrors_subdirectory_structure(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    sub = corpus / "sub"
    sub.mkdir(parents=True)
    (corpus / "top.txt").write_text("top")
    (sub / "nested.txt").write_text("nested")
    pairs = resolve(corpus)
    out_root = tmp_path / "corpus_redacted"
    dests = {p[1] for p in pairs}
    assert out_root / "top.redacted.txt" in dests
    assert out_root / "sub" / "nested.redacted.txt" in dests


def test_resolve_dir_skips_non_txt_files(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("a")
    (corpus / "b.csv").write_text("b")
    (corpus / "c.md").write_text("c")
    pairs = resolve(corpus)
    assert len(pairs) == 1
    assert pairs[0][0].name == "a.txt"


def test_resolve_dir_empty_returns_empty_list(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    assert resolve(corpus) == []


def test_resolve_dir_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve(tmp_path / "missing_dir")


def test_resolve_neither_file_nor_dir_raises(tmp_path: Path) -> None:
    fifo = tmp_path / "mystery"
    os.mkfifo(fifo)

    with pytest.raises(ValueError):
        resolve(fifo)


def test_output_root_single_pair_returns_parent(tmp_path: Path) -> None:
    dest = tmp_path / "out" / "notes.redacted.txt"

    assert output_root([(tmp_path / "notes.txt", dest)]) == dest.parent


def test_output_root_multiple_pairs_finds_common_parent(tmp_path: Path) -> None:
    root = tmp_path / "out"
    pairs = [
        (tmp_path / "a.txt", root / "left" / "a.redacted.txt"),
        (tmp_path / "b.txt", root / "right" / "deep" / "b.redacted.txt"),
    ]

    assert output_root(pairs) == root


def test_output_root_empty_raises() -> None:
    with pytest.raises(ValueError):
        output_root([])
