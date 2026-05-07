"""Span-based PII redaction."""

from __future__ import annotations
from privacy_steward.models import EntitySpan


def redact(
    text: str,
    spans: list[EntitySpan],
    placeholder: str = "[REDACTED]",
) -> str:
    """Replace *spans* in *text* with *placeholder*.

    Replacements are applied right-to-left so earlier offsets stay valid.
    *placeholder* may contain ``{entity_type}``, which is interpolated with
    the upper-cased entity label (e.g. ``[{entity_type}]`` → ``[PER]``).
    """
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        ph = placeholder.format(entity_type=span.entity_type.upper())
        text = text[: span.start] + ph + text[span.end :]
    return text
