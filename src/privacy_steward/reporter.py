"""Audit-dir writer and optional aggregate report builder."""

from __future__ import annotations
import json
from datetime import UTC, datetime
from pathlib import Path
from privacy_steward.models import EntitySpan
from privacy_steward.pipeline import DEFAULT_MODEL


def write_audit(
    src: Path,
    spans: list[EntitySpan],
    audit_root: Path,
    out_root: Path,
    dest: Path,
) -> None:
    """Write per-file classification result to *audit_root*.

    The audit JSON mirrors the destination tree:
    ``<audit_root>/<rel_to_out_root>/<original_stem>.audit.json``
    """
    try:
        rel = dest.relative_to(out_root)
    except ValueError:
        rel = Path(dest.name)

    # Strip the ".redacted" portion added by the resolver
    original_stem = rel.stem.replace(".redacted", "")
    audit_path = audit_root / rel.parent / f"{original_stem}.audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": str(src),
        "destination": str(dest),
        "entities": [
            {
                "start": s.start,
                "end": s.end,
                "entity_type": s.entity_type,
                "score": round(s.score, 6),
                "word": s.word,
            }
            for s in spans
        ],
    }
    audit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_summary_report(
    results: list[dict],  # list of per-file result dicts built in cli.py
    out_root: Path,
    model_id: str = DEFAULT_MODEL,
) -> Path:
    """Write aggregate ``redaction_report.json`` to *out_root*."""
    total_entities = sum(r["total_entities"] for r in results)
    total_elapsed = sum(r["elapsed_seconds"] for r in results)

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model": model_id,
        "files": results,
        "totals": {
            "files_processed": len(results),
            "entities_redacted": total_entities,
            "elapsed_seconds": round(total_elapsed, 3),
        },
    }
    report_path = out_root / "redaction_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report_path
