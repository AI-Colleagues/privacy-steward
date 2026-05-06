"""Pytest configuration and shared fixtures."""

from __future__ import annotations
from collections.abc import Generator
import pytest
from typer.testing import CliRunner
from privacy_steward import pipeline


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests that require model download/inference (deselect with -m 'not slow')",
    )


@pytest.fixture(autouse=True)
def clear_pipeline_caches() -> Generator[None, None, None]:
    """Reset cached pipeline helpers between tests."""
    pipeline._get_model_dir.cache_clear()
    pipeline.get_viterbi_transition_biases.cache_clear()
    pipeline.get_runtime.cache_clear()
    yield
    pipeline._get_model_dir.cache_clear()
    pipeline.get_viterbi_transition_biases.cache_clear()
    pipeline.get_runtime.cache_clear()


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a CLI runner for in-process Typer invocations."""
    return CliRunner()
