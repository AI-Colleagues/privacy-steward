"""Integration tests for the CLI (require model; marked slow)."""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import pytest


FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.txt"
CORPUS = FIXTURES / "corpus"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "privacy_steward.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Single-file mode
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_single_file_creates_redacted_output(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())
    result = _run(str(src))
    assert result.returncode == 0
    dest = tmp_path / "notes.redacted.txt"
    assert dest.exists()
    redacted = dest.read_text()
    assert "Alice Johnson" not in redacted


@pytest.mark.slow
def test_single_file_explicit_output(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())
    out = tmp_path / "clean.txt"
    result = _run(str(src), "--output", str(out))
    assert result.returncode == 0
    assert out.exists()


@pytest.mark.slow
def test_single_file_audit_dir_created(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())
    _run(str(src))
    audit_dir = tmp_path / ".audit"
    assert audit_dir.is_dir()
    audit_files = list(audit_dir.rglob("*.audit.json"))
    assert len(audit_files) == 1
    data = json.loads(audit_files[0].read_text())
    assert "entities" in data
    assert "source" in data


@pytest.mark.slow
def test_single_file_dry_run_writes_nothing(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())
    result = _run(str(src), "--dry-run")
    assert result.returncode == 0
    assert not (tmp_path / "notes.redacted.txt").exists()
    assert not (tmp_path / ".audit").exists()


@pytest.mark.slow
def test_single_file_verbose_output(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())
    result = _run(str(src), "-v")
    assert result.returncode == 0
    assert "Done" in result.stdout


@pytest.mark.slow
def test_single_file_entity_type_placeholder(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text("My name is Alice Johnson.\n")
    result = _run(str(src), "--placeholder", "[{entity_type}]")
    assert result.returncode == 0
    dest = tmp_path / "notes.redacted.txt"
    content = dest.read_text()
    assert "Alice Johnson" not in content
    # Should contain an entity-type label in square brackets
    assert "[" in content and "]" in content


@pytest.mark.slow
def test_single_file_report_flag(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())
    _run(str(src), "--report")
    report = tmp_path / "redaction_report.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["totals"]["files_processed"] == 1


# ---------------------------------------------------------------------------
# Directory mode
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_directory_mode_processes_all_txt_files(tmp_path: Path) -> None:
    import shutil

    corpus_copy = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus_copy)
    result = _run(str(corpus_copy))
    assert result.returncode == 0
    out_root = tmp_path / "corpus_redacted"
    assert out_root.is_dir()
    assert (out_root / "chapter1.redacted.txt").exists()
    assert (out_root / "chapter2.redacted.txt").exists()
    assert (out_root / "subdir" / "chapter3.redacted.txt").exists()


@pytest.mark.slow
def test_directory_mode_audit_mirrors_structure(tmp_path: Path) -> None:
    import shutil

    corpus_copy = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus_copy)
    _run(str(corpus_copy))
    audit_root = tmp_path / "corpus_redacted" / ".audit"
    assert audit_root.is_dir()
    audit_files = list(audit_root.rglob("*.audit.json"))
    assert len(audit_files) == 3  # one per .txt file


@pytest.mark.slow
def test_directory_mode_explicit_output(tmp_path: Path) -> None:
    import shutil

    corpus_copy = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus_copy)
    out = tmp_path / "sanitised"
    result = _run(str(corpus_copy), "--output", str(out))
    assert result.returncode == 0
    assert (out / "chapter1.redacted.txt").exists()


@pytest.mark.slow
def test_missing_path_exits_nonzero() -> None:
    result = _run("/tmp/definitely_does_not_exist_xyz.txt")
    assert result.returncode != 0
