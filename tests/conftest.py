"""Pytest configuration — registers the 'slow' marker."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests that require model download/inference (deselect with -m 'not slow')",
    )
