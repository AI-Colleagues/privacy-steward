"""Tests for audit and report writers."""

from __future__ import annotations
import json
from pathlib import Path
from privacy_steward.models import EntitySpan
from privacy_steward.reporter import write_audit, write_summary_report


def _span(start: int, end: int, entity_type: str = "PER") -> EntitySpan:
    return EntitySpan(
        start=start,
        end=end,
        entity_type=entity_type,
        score=0.987654321,
        word="Alice Johnson",
    )


def test_write_audit_mirrors_destination_tree(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    audit_root = out_root / ".audit"
    src = tmp_path / "input" / "notes.txt"
    dest = out_root / "nested" / "notes.redacted.txt"
    src.parent.mkdir(parents=True)
    dest.parent.mkdir(parents=True)
    src.write_text("hello", encoding="utf-8")
    dest.write_text("redacted", encoding="utf-8")

    write_audit(src, [_span(0, 5)], audit_root, out_root, dest)

    audit = audit_root / "nested" / "notes.audit.json"
    data = json.loads(audit.read_text(encoding="utf-8"))
    assert data["source"] == str(src)
    assert data["destination"] == str(dest)
    assert data["entities"][0]["score"] == 0.987654


def test_write_audit_falls_back_to_destination_name(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    out_root = tmp_path / "out"
    src = tmp_path / "input.txt"
    dest = tmp_path / "elsewhere" / "notes.redacted.txt"
    dest.parent.mkdir(parents=True)
    src.write_text("hello", encoding="utf-8")
    dest.write_text("redacted", encoding="utf-8")

    write_audit(src, [], audit_root, out_root, dest)

    audit = audit_root / "notes.audit.json"
    assert audit.exists()


def test_write_summary_report_writes_totals(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    out_root.mkdir()
    results = [
        {
            "source": "a.txt",
            "destination": "a.redacted.txt",
            "entity_counts": {"PER": 1},
            "total_entities": 1,
            "elapsed_seconds": 0.125,
        },
        {
            "source": "b.txt",
            "destination": "b.redacted.txt",
            "entity_counts": {},
            "total_entities": 0,
            "elapsed_seconds": 0.375,
        },
    ]

    report_path = write_summary_report(results, out_root, model_id="custom-model")

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path == out_root / "redaction_report.json"
    assert data["model"] == "custom-model"
    assert data["totals"] == {
        "files_processed": 2,
        "entities_redacted": 1,
        "elapsed_seconds": 0.5,
    }
    assert len(data["files"]) == 2
    assert data["generated_at"]
