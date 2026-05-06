"""Adapter around the official OpenAI Privacy Filter package."""

from __future__ import annotations
from opf import OPF
from privacy_steward.models import EntitySpan


DEFAULT_MODEL = "openai/privacy-filter"
DEFAULT_DEVICE = "cpu"


class NERPipeline:
    """Thin wrapper that converts OPF detections into local span objects."""

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        """Initialize the upstream OPF runtime with local defaults."""
        self._opf = OPF(
            model=checkpoint,
            device=device,
            output_mode="typed",
            output_text_only=False,
        )

    def predict(self, text: str) -> list[EntitySpan]:
        """Return detected PII spans for *text* using the upstream OPF runtime."""
        if not text.strip():
            return []

        result = self._opf.redact(text)
        if not hasattr(result, "detected_spans"):
            raise TypeError("OPF.redact returned text-only output; expected a result")

        return [
            EntitySpan(
                start=span.start,
                end=span.end,
                entity_type=span.label,
                score=1.0,
                word=span.text,
            )
            for span in result.detected_spans
        ]
