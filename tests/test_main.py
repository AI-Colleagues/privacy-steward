"""Tests for the top-level main entry point."""

from __future__ import annotations
import privacy_steward.main as main_module


def test_main_prints_message(capsys) -> None:
    main_module.main()

    assert capsys.readouterr().out == "Hello from uv-template!\n"
