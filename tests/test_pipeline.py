"""Tests for the official OPF adapter."""

from __future__ import annotations
from dataclasses import dataclass
import pytest
import privacy_steward.pipeline as pipeline_module
from privacy_steward.models import EntitySpan
from privacy_steward.pipeline import NERPipeline


@dataclass(frozen=True)
class _FakeDetectedSpan:
    label: str
    start: int
    end: int
    text: str
    placeholder: str


@dataclass(frozen=True)
class _FakeResult:
    detected_spans: tuple[_FakeDetectedSpan, ...]


class _TextOnlyResult:
    pass


class _FakeOPF:
    instances: list[_FakeOPF] = []

    def __init__(
        self,
        *,
        model: str | None,
        device: str,
        output_mode: str,
        output_text_only: bool,
    ) -> None:
        self.model = model
        self.device = device
        self.output_mode = output_mode
        self.output_text_only = output_text_only
        self.seen_texts: list[str] = []
        self.__class__.instances.append(self)

    def redact(self, text: str) -> _FakeResult:
        self.seen_texts.append(text)
        return _FakeResult(
            detected_spans=(
                _FakeDetectedSpan(
                    label="private_person",
                    start=11,
                    end=24,
                    text="Alice Johnson",
                    placeholder="<PRIVATE_PERSON>",
                ),
            )
        )


def test_pipeline_uses_official_opf_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "OPF", _FakeOPF)
    _FakeOPF.instances.clear()

    pipe = NERPipeline(checkpoint="/tmp/checkpoint", device="cuda")
    spans = pipe.predict("Hello Alice Johnson.")

    assert len(_FakeOPF.instances) == 1
    opf = _FakeOPF.instances[0]
    assert opf.model == "/tmp/checkpoint"
    assert opf.device == "cuda"
    assert opf.output_mode == "typed"
    assert opf.output_text_only is False
    assert opf.seen_texts == ["Hello Alice Johnson."]

    assert spans == [
        EntitySpan(
            start=11,
            end=24,
            entity_type="private_person",
            score=1.0,
            word="Alice Johnson",
        )
    ]


def test_pipeline_empty_text_returns_no_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "OPF", _FakeOPF)
    _FakeOPF.instances.clear()

    pipe = NERPipeline()
    assert pipe.predict("") == []
    assert pipe.predict("   ") == []
    assert _FakeOPF.instances[-1].seen_texts == []


def test_pipeline_rejects_text_only_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline_module,
        "OPF",
        lambda **kwargs: type(
            "FakeOPF", (), {"redact": lambda self, text: _TextOnlyResult()}
        )(),
    )

    pipe = NERPipeline()

    with pytest.raises(TypeError, match="text-only output"):
        pipe.predict("Hello")
