"""Smoke tests for the NER pipeline."""

import pytest
from privacy_steward.models import EntitySpan
from privacy_steward.pipeline import (
    NERPipeline,
    token_spans_to_char_spans,
    trim_char_spans_whitespace,
)


# ---------------------------------------------------------------------------
# trim_char_spans_whitespace — pure function, always runs
# ---------------------------------------------------------------------------


def test_trim_strips_leading_space() -> None:
    text = "Dear Karen Patel,"
    spans = [(1, 4, 10)]  # label=1, " Karen" includes leading space
    result = trim_char_spans_whitespace(spans, text)
    assert len(result) == 1
    assert result[0] == (1, 5, 10)  # space at index 4 stripped


def test_trim_strips_trailing_space() -> None:
    text = "Hello world  "
    spans = [(1, 0, 13)]
    result = trim_char_spans_whitespace(spans, text)
    assert result[0][2] == 11


def test_trim_drops_all_whitespace_span() -> None:
    text = "Hello   world"
    spans = [(1, 5, 8)]  # three spaces
    assert trim_char_spans_whitespace(spans, text) == []


def test_trim_no_change_when_no_whitespace() -> None:
    text = "Karen"
    spans = [(1, 0, 5)]
    result = trim_char_spans_whitespace(spans, text)
    assert result == [(1, 0, 5)]


def test_trim_skips_out_of_bounds_span() -> None:
    text = "Hi"
    spans = [(1, 0, 10)]  # end beyond text length
    assert trim_char_spans_whitespace(spans, text) == []


# ---------------------------------------------------------------------------
# token_spans_to_char_spans — pure function, always runs
# ---------------------------------------------------------------------------


def test_token_spans_to_char_spans_basic() -> None:
    # 3 tokens: "Hello"(0-5), " world"(5-11), "!"(11-12)
    char_starts = [0, 5, 11]
    char_ends = [5, 11, 12]
    spans = [(1, 0, 2)]  # label=1, tokens 0..2 → chars 0..11
    result = token_spans_to_char_spans(spans, char_starts, char_ends)
    assert result == [(1, 0, 11)]


def test_token_spans_to_char_spans_skips_invalid() -> None:
    char_starts = [0, 5]
    char_ends = [5, 10]
    spans = [(1, 5, 10)]  # token indices beyond range
    assert token_spans_to_char_spans(spans, char_starts, char_ends) == []


def test_token_spans_to_char_spans_empty() -> None:
    assert token_spans_to_char_spans([], [0, 5], [5, 10]) == []


# ---------------------------------------------------------------------------
# NERPipeline — requires model + CUDA, marked slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_pipeline_detects_person_name() -> None:
    pipe = NERPipeline()
    spans = pipe.predict("My name is Alice Johnson and I live in Springfield.")
    assert any(s.entity_type for s in spans), "expected at least one entity"
    assert all(isinstance(s, EntitySpan) for s in spans)
    assert all(0.0 <= s.score <= 1.0 for s in spans)


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
    assert len(spans) < 3


@pytest.mark.slow
def test_pipeline_offsets_are_valid() -> None:
    pipe = NERPipeline()
    text = "Contact alice@example.com or call +1-555-867-5309."
    spans = pipe.predict(text)
    for s in spans:
        assert 0 <= s.start < s.end <= len(text)
        assert text[s.start : s.end].strip()
