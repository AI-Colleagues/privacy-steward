"""Tests for audit and report writers."""

from __future__ import annotations
import json
from pathlib import Path
from privacy_steward.models import EntitySpan
from privacy_steward.pipeline import DEFAULT_MODEL
from privacy_steward.reporter import write_audit, write_summary_report


def _span(start: int, end: int, entity_type: str = "private_person") -> EntitySpan:
    return EntitySpan(
        start=start, end=end, entity_type=entity_type, score=0.9, word="x"
    )


def test_write_audit_uses_destination_relative_to_output_root(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    dest = tmp_path / "out" / "sub" / "notes.redacted.txt"
    out_root = tmp_path / "out"
    audit_root = out_root / ".audit"

    write_audit(src, [_span(0, 5)], audit_root, out_root, dest)

    audit_file = audit_root / "sub" / "notes.audit.json"
    assert audit_file.exists()
    payload = json.loads(audit_file.read_text())
    assert payload["source"] == str(src)
    assert payload["destination"] == str(dest)
    assert payload["entities"][0]["entity_type"] == "private_person"
    assert "word" not in payload["entities"][0]


def test_write_audit_can_include_source_text(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    dest = tmp_path / "out" / "notes.redacted.txt"
    out_root = tmp_path / "out"
    audit_root = out_root / ".audit"

    write_audit(src, [_span(0, 5)], audit_root, out_root, dest, include_text=True)

    audit_file = audit_root / "notes.audit.json"
    payload = json.loads(audit_file.read_text())
    assert payload["entities"][0]["word"] == "x"


def test_write_audit_falls_back_when_destination_is_external(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    dest = tmp_path / "my.redacted.data.redacted.txt"
    out_root = tmp_path / "out"
    audit_root = out_root / ".audit"

    write_audit(src, [_span(0, 5)], audit_root, out_root, dest)

    audit_file = audit_root / "my.redacted.data.audit.json"
    assert audit_file.exists()


def test_write_summary_report_records_totals(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    results = [
        {
            "source": "/tmp/a.txt",
            "destination": "/tmp/out/a.redacted.txt",
            "entity_counts": {"private_person": 1},
            "total_entities": 1,
            "elapsed_seconds": 0.125,
        },
        {
            "source": "/tmp/b.txt",
            "destination": "/tmp/out/b.redacted.txt",
            "entity_counts": {"private_email": 2},
            "total_entities": 2,
            "elapsed_seconds": 0.375,
        },
    ]

    report_path = write_summary_report(results, out_root, model_id=DEFAULT_MODEL)

    assert report_path == out_root / "redaction_report.json"
    payload = json.loads(report_path.read_text())
    assert payload["model"] == DEFAULT_MODEL
    assert payload["totals"]["files_processed"] == 2
    assert payload["totals"]["entities_redacted"] == 3
    assert payload["totals"]["elapsed_seconds"] == 0.5
