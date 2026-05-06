"""Slow smoke tests for the NER pipeline (require model download)."""

import pytest
from privacy_steward.models import EntitySpan
from privacy_steward.pipeline import (
    NERPipeline,
    _merge_adjacent,
    _split_paragraphs,
    _trim_spans_whitespace,
)


# ---------------------------------------------------------------------------
# _split_paragraphs — pure function, always runs
# ---------------------------------------------------------------------------


def test_split_paragraphs_short_text_stays_intact() -> None:
    text = "Hello world.\n\nSecond paragraph."
    chunks = _split_paragraphs(text, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0] == text  # both paragraphs fit in one chunk


def test_split_paragraphs_returns_all_content_when_joined() -> None:
    text = "A\n\nB\n\nC"
    chunks = _split_paragraphs(text, max_chars=5)
    # Each single-character paragraph becomes its own chunk
    joined = "\n\n".join(chunks)
    # All original content must be present
    for ch in ["A", "B", "C"]:
        assert ch in joined


def test_split_paragraphs_roundtrips_via_join() -> None:
    """Chunks joined with \\n\\n must reconstruct the original text exactly."""
    text = "Para one.\n\nPara two.\n\nPara three is longer.\n\nPara four."
    for max_chars in [10, 20, 1000]:
        chunks = _split_paragraphs(text, max_chars=max_chars)
        assert "\n\n".join(chunks) == text


def test_split_paragraphs_single_large_paragraph_kept() -> None:
    text = "x" * 5000
    chunks = _split_paragraphs(text, max_chars=100)
    assert len(chunks) == 1
    assert chunks[0] == text


# ---------------------------------------------------------------------------
# _merge_adjacent — pure function, always runs
# ---------------------------------------------------------------------------


def _make_span(
    start: int, end: int, entity_type: str = "PER", score: float = 0.99
) -> EntitySpan:
    return EntitySpan(
        start=start, end=end, entity_type=entity_type, score=score, word="x"
    )


def test_merge_adjacent_empty() -> None:
    assert _merge_adjacent([]) == []


def test_merge_adjacent_single_span() -> None:
    spans = [_make_span(5, 10)]
    assert _merge_adjacent(spans) == spans


def test_merge_adjacent_non_touching_spans_unchanged() -> None:
    text = "Hello world, goodbye."
    spans = [_make_span(0, 5), _make_span(13, 20)]
    result = _merge_adjacent(spans, text)
    assert len(result) == 2
    assert result[0].start == 0 and result[0].end == 5
    assert result[1].start == 13 and result[1].end == 20


def test_merge_adjacent_touching_spans_merged() -> None:
    """Two abutting spans for the same entity should collapse to one."""
    spans = [_make_span(4, 10), _make_span(10, 16)]
    result = _merge_adjacent(spans)
    assert len(result) == 1
    assert result[0].start == 4
    assert result[0].end == 16


def test_merge_adjacent_overlapping_spans_merged() -> None:
    spans = [_make_span(4, 12), _make_span(10, 16)]
    result = _merge_adjacent(spans)
    assert len(result) == 1
    assert result[0].start == 4
    assert result[0].end == 16


def test_merge_adjacent_mixed_types_uses_highest_score() -> None:
    """Adjacent spans of different entity types: winner is highest-scoring."""
    spans = [
        _make_span(0, 5, "ACCOUNT_NUMBER", score=0.87),
        _make_span(5, 10, "PHONE", score=0.64),
    ]
    result = _merge_adjacent(spans)
    assert len(result) == 1
    assert result[0].entity_type == "ACCOUNT_NUMBER"
    assert result[0].score == 0.87


def test_merge_adjacent_three_consecutive() -> None:
    spans = [_make_span(0, 5), _make_span(5, 10), _make_span(10, 15)]
    result = _merge_adjacent(spans)
    assert len(result) == 1
    assert result[0].start == 0 and result[0].end == 15


def test_merge_adjacent_whitespace_gap_merged() -> None:
    """Spans separated by only whitespace should collapse to one (e.g. 'Karen Patel')."""
    text = "Dear Karen Patel,"
    # After trimming, "Karen" is at (5,10) and "Patel" is at (11,16); gap=" "
    spans = [_make_span(5, 10), _make_span(11, 16)]
    result = _merge_adjacent(spans, text)
    assert len(result) == 1
    assert result[0].start == 5 and result[0].end == 16


def test_merge_adjacent_non_whitespace_gap_not_merged() -> None:
    """Spans with non-whitespace text between them must stay separate."""
    text = "Alice called Bob"
    spans = [_make_span(0, 5), _make_span(13, 16)]  # "Alice" and "Bob"
    result = _merge_adjacent(spans, text)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _trim_spans_whitespace — pure function, always runs
# ---------------------------------------------------------------------------


def test_trim_spans_whitespace_strips_leading_space() -> None:
    """BPE tokens include leading space; trim should remove it."""
    text = "Dear Karen Patel,"
    # " Karen" is at (4,10) — includes the space after "Dear"
    spans = [_make_span(4, 10)]
    result = _trim_spans_whitespace(spans, text)
    assert len(result) == 1
    assert result[0].start == 5  # space at 4 stripped
    assert result[0].end == 10


def test_trim_spans_whitespace_strips_trailing_space() -> None:
    text = "Hello world  "
    spans = [_make_span(0, 13)]
    result = _trim_spans_whitespace(spans, text)
    assert result[0].end == 11  # trailing spaces removed


def test_trim_spans_whitespace_drops_all_whitespace_span() -> None:
    text = "Hello   world"
    spans = [_make_span(5, 8)]  # three spaces
    result = _trim_spans_whitespace(spans, text)
    assert result == []


def test_trim_spans_whitespace_no_change_when_no_whitespace() -> None:
    text = "Karen"
    spans = [_make_span(0, 5)]
    result = _trim_spans_whitespace(spans, text)
    assert result[0].start == 0 and result[0].end == 5


# ---------------------------------------------------------------------------
# NERPipeline — requires model, marked slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_pipeline_detects_person_name() -> None:
    pipe = NERPipeline()
    spans = pipe.predict("My name is Alice Johnson and I live in Springfield.")
    assert any(s.entity_type for s in spans), "expected at least one entity"
    assert all(isinstance(s, EntitySpan) for s in spans)
    assert all(0 <= s.score <= 1 for s in spans)


@pytest.mark.slow
def test_pipeline_empty_text_returns_no_spans() -> None:
    pipe = NERPipeline()
    assert pipe.predict("") == []
    assert pipe.predict("   ") == []


@pytest.mark.slow
def test_pipeline_non_pii_text_returns_few_spans() -> None:
    pipe = NERPipeline()
    text = (
        "The quarterly earnings report showed strong growth across all sectors. "
        "Python 3.12 introduced significant performance improvements."
    )
    spans = pipe.predict(text)
    # Non-PII text should produce zero or very few detections
    assert len(spans) < 3


@pytest.mark.slow
def test_pipeline_offsets_are_valid() -> None:
    pipe = NERPipeline()
    text = "Contact alice@example.com or call +1-555-867-5309."
    spans = pipe.predict(text)
    for s in spans:
        assert 0 <= s.start < s.end <= len(text)
        assert text[s.start : s.end].strip()  # non-empty surface form
