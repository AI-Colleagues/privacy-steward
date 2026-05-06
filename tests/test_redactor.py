"""Unit tests for the redaction engine."""

from privacy_steward.models import EntitySpan
from privacy_steward.redactor import redact


def _span(start: int, end: int, entity_type: str = "PER") -> EntitySpan:
    return EntitySpan(
        start=start, end=end, entity_type=entity_type, score=0.99, word="x"
    )


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------


def test_redact_single_span() -> None:
    text = "Hello Alice Johnson, how are you?"
    spans = [_span(6, 19)]
    result = redact(text, spans)
    assert result == "Hello [REDACTED], how are you?"


def test_redact_no_spans_returns_original() -> None:
    text = "Nothing sensitive here."
    assert redact(text, []) == text


def test_redact_empty_text() -> None:
    assert redact("", []) == ""


def test_redact_full_text() -> None:
    text = "Alice"
    result = redact(text, [_span(0, 5)])
    assert result == "[REDACTED]"


# ---------------------------------------------------------------------------
# Multiple spans (right-to-left ordering)
# ---------------------------------------------------------------------------


def test_redact_multiple_spans_ordered_correctly() -> None:
    text = "Call Alice at 555-1234 please."
    # "Alice" at 5..10, "555-1234" at 14..22
    spans = [_span(5, 10, "PER"), _span(14, 22, "PHONE")]
    result = redact(text, spans)
    assert result == "Call [REDACTED] at [REDACTED] please."


def test_redact_spans_given_in_forward_order_still_correct() -> None:
    text = "Alice and Bob"
    spans = [_span(0, 5, "PER"), _span(10, 13, "PER")]
    result = redact(text, spans)
    assert result == "[REDACTED] and [REDACTED]"


# ---------------------------------------------------------------------------
# Placeholder templating
# ---------------------------------------------------------------------------


def test_redact_custom_literal_placeholder() -> None:
    text = "Name: Alice"
    result = redact(text, [_span(6, 11)], placeholder="***")
    assert result == "Name: ***"


def test_redact_entity_type_template() -> None:
    text = "alice@example.com"
    result = redact(text, [_span(0, 17, "EMAIL")], placeholder="[{entity_type}]")
    assert result == "[EMAIL]"


def test_redact_entity_type_template_mixed() -> None:
    text = "Alice, alice@example.com"
    spans = [_span(0, 5, "PER"), _span(7, 24, "EMAIL")]
    result = redact(text, spans, placeholder="[{entity_type}]")
    assert result == "[PER], [EMAIL]"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_redact_non_ascii_text_preserved() -> None:
    text = "Bonjour, Élise Dupont, comment allez-vous?"
    # Only redact "Élise Dupont" (bytes 9..21 in this string)
    start = text.index("Élise")
    end = start + len("Élise Dupont")
    result = redact(text, [_span(start, end)])
    assert "[REDACTED]" in result
    assert "Bonjour" in result
    assert "comment allez-vous?" in result


def test_redact_adjacent_spans() -> None:
    text = "AliceBob"
    spans = [_span(0, 5, "PER"), _span(5, 8, "PER")]
    result = redact(text, spans)
    assert result == "[REDACTED][REDACTED]"
