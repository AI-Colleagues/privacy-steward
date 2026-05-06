"""NER pipeline wrapper for openai/privacy-filter."""

from __future__ import annotations
from typing import Any
from transformers import pipeline as hf_pipeline
from privacy_steward.models import EntitySpan


DEFAULT_MODEL = "openai/privacy-filter"
# The model supports up to 131 072 tokens; 64 000 chars keeps documents together
# so the model has full context and avoids mid-letter chunk boundaries.
_CHUNK_SIZE = 64_000


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
        spans = _trim_spans_whitespace(spans, text)
        return _merge_adjacent(spans, text)


def _trim_spans_whitespace(spans: list[EntitySpan], text: str) -> list[EntitySpan]:
    """Strip leading/trailing whitespace from each span's character range.

    The BPE tokenizer bundles the preceding space into the token (e.g.
    "ĠKaren" covers " Karen" at offsets 4–10).  Without trimming, redacting
    span (4, 10) on "Dear Karen," produces "Dear<TAG>," — the space is eaten.
    """
    trimmed: list[EntitySpan] = []
    for span in spans:
        start, end = span.start, span.end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            trimmed.append(
                EntitySpan(
                    start=start,
                    end=end,
                    entity_type=span.entity_type,
                    score=span.score,
                    word=text[start:end],
                )
            )
    return trimmed


def _merge_adjacent(spans: list[EntitySpan], text: str = "") -> list[EntitySpan]:
    """Merge adjacent or whitespace-separated spans into a single span.

    The model tokenizes at subword boundaries, so a single entity like
    "Karen Patel" may come back as two S-tagged spans ("Karen" and "Patel")
    separated by a space.  Merging any pair whose gap consists solely of
    whitespace collapses them into one placeholder.

    For spans that overlap or whose gap is all-whitespace, the merged span
    takes the entity_type and score of the highest-scoring constituent.
    """
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: s.start)
    merged: list[EntitySpan] = []
    current = sorted_spans[0]
    for span in sorted_spans[1:]:
        gap = text[current.end : span.start] if text else ""
        if not gap or gap.isspace():
            best = current if current.score >= span.score else span
            current = EntitySpan(
                start=current.start,
                end=max(current.end, span.end),
                entity_type=best.entity_type,
                score=best.score,
                word=current.word + gap + span.word,
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
