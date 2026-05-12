"""Audit-dir writer and optional aggregate report builder."""

from __future__ import annotations
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict
from privacy_steward.models import EntitySpan
from privacy_steward.pipeline import DEFAULT_MODEL


def write_audit(
    src: Path,
    spans: list[EntitySpan],
    audit_root: Path,
    out_root: Path,
    dest: Path,
    *,
    include_text: bool = False,
) -> None:
    """Write per-file classification result to *audit_root*.

    The audit JSON mirrors the destination tree:
    ``<audit_root>/<rel_to_out_root>/<original_stem>.audit.json``
    """
    try:
        rel = dest.relative_to(out_root)
    except ValueError:
        rel = Path(dest.name)

    # Strip the trailing ".redacted" portion added by the resolver.
    original_stem = rel.stem.removesuffix(".redacted")
    audit_path = audit_root / rel.parent / f"{original_stem}.audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    entities: list[dict[str, object]] = []
    for span in spans:
        entity: dict[str, object] = {
            "start": span.start,
            "end": span.end,
            "entity_type": span.entity_type,
            "score": round(span.score, 6),
        }
        if include_text:
            entity["word"] = span.word
        entities.append(entity)

    payload = {
        "source": str(src),
        "destination": str(dest),
        "entities": entities,
    }
    audit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


class RedactionResult(TypedDict):
    """Per-file redaction summary emitted by the CLI."""

    source: str
    destination: str
    entity_counts: dict[str, int]
    total_entities: int
    elapsed_seconds: float


def write_summary_report(
    results: list[RedactionResult],
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
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report_path
