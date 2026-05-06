"""NER pipeline wrapper for openai/privacy-filter."""

from __future__ import annotations
from typing import Any
from transformers import pipeline as hf_pipeline
from privacy_steward.models import EntitySpan


DEFAULT_MODEL = "openai/privacy-filter"
_CHUNK_SIZE = 2_000  # characters; well within 512-token window for typical prose


class NERPipeline:
    """Token-classification pipeline loaded once and reused across files."""

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:  # noqa: D107
        self._clf: Any = hf_pipeline(
            task="token-classification",
            model=model_id,
            aggregation_strategy="simple",
        )

    def predict(self, text: str) -> list[EntitySpan]:
        """Return PII entity spans detected in *text*.

        Long texts are split at paragraph boundaries so each chunk stays
        within the model's context window; offsets are stitched back.
        """
        if not text.strip():
            return []
        chunks = _split_paragraphs(text, _CHUNK_SIZE)
        spans: list[EntitySpan] = []
        offset = 0
        for chunk in chunks:
            raw: list[dict[str, Any]] = self._clf(chunk)
            for e in raw:
                spans.append(
                    EntitySpan(
                        start=e["start"] + offset,
                        end=e["end"] + offset,
                        entity_type=e["entity_group"],
                        score=float(e["score"]),
                        word=e["word"],
                    )
                )
            # +2 accounts for the '\n\n' separator between chunks in the original text
            offset += len(chunk) + 2
        return _merge_adjacent(spans)


def _merge_adjacent(spans: list[EntitySpan]) -> list[EntitySpan]:
    """Merge adjacent or overlapping spans into a single span.

    The model tokenizes at subword boundaries, so a single entity like
    "alice.j@acme.com" may be returned as two abutting spans (".com" split
    off separately).  Without merging, the redactor produces doubled tags
    and garbled surrounding text.

    For spans that touch or overlap, the merged span takes the entity_type
    and score of the highest-scoring constituent.
    """
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: s.start)
    merged: list[EntitySpan] = []
    current = sorted_spans[0]
    for span in sorted_spans[1:]:
        if span.start <= current.end:
            best = current if current.score >= span.score else span
            current = EntitySpan(
                start=current.start,
                end=max(current.end, span.end),
                entity_type=best.entity_type,
                score=best.score,
                word=current.word + span.word,
            )
        else:
            merged.append(current)
            current = span
    merged.append(current)
    return merged


def _split_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split *text* into chunks of at most *max_chars* at paragraph breaks."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para) if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # paragraph itself larger than limit — keep as one chunk
            current = para
    if current:
        chunks.append(current)
    return chunks
