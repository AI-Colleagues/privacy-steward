"""Tests for the CLI without invoking the real model runtime."""

from __future__ import annotations
import json
import runpy
from pathlib import Path
import pytest
import typer
from typer.testing import CliRunner
import privacy_steward.cli as cli_module
from privacy_steward.models import EntitySpan


runner = CliRunner()


class _DummyProgress:
    def __init__(self) -> None:
        self.tasks: list[tuple[str, int | None]] = []
        self.updates: list[str] = []
        self.advances = 0

    def __enter__(self) -> _DummyProgress:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def add_task(self, description: str, total: int | None = None) -> int:
        self.tasks.append((description, total))
        return 1

    def update(self, task: int, description: str) -> None:
        self.updates.append(description)

    def advance(self, task: int) -> None:
        self.advances += 1


def _install_progress(monkeypatch: pytest.MonkeyPatch) -> _DummyProgress:
    progress = _DummyProgress()
    monkeypatch.setattr(cli_module, "_make_progress", lambda: progress)
    return progress


def _install_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    spans: list[EntitySpan],
) -> type:
    class FakePipeline:
        instances: list[FakePipeline] = []

        def __init__(self, *, checkpoint: str | None, device: str) -> None:
            self.checkpoint = checkpoint
            self.device = device
            self.seen_texts: list[str] = []
            self.__class__.instances.append(self)

        def predict(self, text: str) -> list[EntitySpan]:
            self.seen_texts.append(text)
            return list(spans)

    monkeypatch.setattr(cli_module, "NERPipeline", FakePipeline)
    return FakePipeline


def _raise_resolve_error(
    input_path: Path, output: Path | None
) -> list[tuple[Path, Path]]:
    raise ValueError("broken")


def test_make_progress_returns_progress_object() -> None:
    progress = cli_module._make_progress()
    from rich.progress import Progress

    assert isinstance(progress, Progress)


def test_redact_cmd_single_file_redacts_and_writes_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_progress(monkeypatch)
    fake_pipeline = _install_pipeline(
        monkeypatch,
        [
            EntitySpan(
                start=11,
                end=24,
                entity_type="PER",
                score=0.99,
                word="Alice Johnson",
            )
        ],
    )

    src = tmp_path / "notes.txt"
    src.write_text("My name is Alice Johnson.\n", encoding="utf-8")

    result = runner.invoke(cli_module.app, [str(src), "-v"])

    assert result.exit_code == 0
    assert len(fake_pipeline.instances) == 1
    assert fake_pipeline.instances[0].checkpoint is None
    assert fake_pipeline.instances[0].device == "cpu"
    assert fake_pipeline.instances[0].seen_texts == ["My name is Alice Johnson.\n"]

    dest = tmp_path / "notes.redacted.txt"
    assert dest.read_text(encoding="utf-8") == "My name is <PER>.\n"

    audit = tmp_path / ".audit" / "notes.audit.json"
    data = json.loads(audit.read_text(encoding="utf-8"))
    assert data["source"] == str(src)
    assert data["destination"] == str(dest)
    assert data["entities"] == [
        {
            "start": 11,
            "end": 24,
            "entity_type": "PER",
            "score": 0.99,
            "word": "Alice Johnson",
        }
    ]


def test_redact_cmd_single_file_with_empty_text_shows_no_pii(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_progress(monkeypatch)
    fake_pipeline = _install_pipeline(monkeypatch, [])

    src = tmp_path / "empty.txt"
    src.write_text("", encoding="utf-8")

    result = runner.invoke(cli_module.app, [str(src), "-v"])

    assert result.exit_code == 0
    assert fake_pipeline.instances[0].seen_texts == [""]
    assert (tmp_path / "empty.redacted.txt").read_text(encoding="utf-8") == ""
    assert "No PII detected." in result.output


def test_redact_cmd_directory_writes_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_progress(monkeypatch)
    _install_pipeline(monkeypatch, [])

    corpus = tmp_path / "corpus"
    nested = corpus / "subdir"
    nested.mkdir(parents=True)
    (corpus / "chapter1.txt").write_text("one", encoding="utf-8")
    (corpus / "chapter2.txt").write_text("two", encoding="utf-8")
    (nested / "chapter3.txt").write_text("three", encoding="utf-8")

    result = runner.invoke(cli_module.app, [str(corpus), "--report", "-v"])

    assert result.exit_code == 0

    out_root = tmp_path / "corpus_redacted"
    assert (out_root / "chapter1.redacted.txt").exists()
    assert (out_root / "chapter2.redacted.txt").exists()
    assert (out_root / "subdir" / "chapter3.redacted.txt").exists()

    report = out_root / "redaction_report.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["model"] == cli_module.DEFAULT_MODEL
    assert data["totals"]["files_processed"] == 3
    assert data["totals"]["entities_redacted"] == 0
    assert "Report written to" in result.output
    assert "Audit records written to" in result.output


def test_redact_cmd_report_without_verbose_prints_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_progress(monkeypatch)
    _install_pipeline(monkeypatch, [])

    src = tmp_path / "notes.txt"
    src.write_text("hello", encoding="utf-8")

    result = runner.invoke(cli_module.app, [str(src), "--report"])

    assert result.exit_code == 0
    assert (tmp_path / "notes.redacted.txt").exists()
    assert (tmp_path / "redaction_report.json").exists()
    assert "Report written to" not in result.output


def test_redact_cmd_dry_run_skips_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_progress(monkeypatch)
    _install_pipeline(
        monkeypatch,
        [
            EntitySpan(
                start=11,
                end=24,
                entity_type="PER",
                score=0.99,
                word="Alice Johnson",
            )
        ],
    )

    src = tmp_path / "notes.txt"
    src.write_text("My name is Alice Johnson.\n", encoding="utf-8")

    result = runner.invoke(cli_module.app, [str(src), "--dry-run", "--report", "-v"])

    assert result.exit_code == 0
    assert not (tmp_path / "notes.redacted.txt").exists()
    assert not (tmp_path / ".audit").exists()
    assert not (tmp_path / "redaction_report.json").exists()
    assert "Audit records written to" not in result.output


def test_redact_cmd_invalid_device_exits_nonzero(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text("hello", encoding="utf-8")

    result = runner.invoke(cli_module.app, [str(src), "--device", "tpu"])

    assert result.exit_code == 1
    assert "--device must be one of: cpu, cuda" in result.output


def test_redact_cmd_missing_path_exits_nonzero() -> None:
    result = runner.invoke(cli_module.app, ["/tmp/definitely_missing.txt"])

    assert result.exit_code == 1
    assert "path does not exist" in result.output


def test_redact_cmd_empty_directory_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_progress(monkeypatch)
    monkeypatch.setattr(cli_module, "resolve", lambda input_path, output: [])

    corpus = tmp_path / "corpus"
    corpus.mkdir()

    result = runner.invoke(cli_module.app, [str(corpus)])

    assert result.exit_code == 0
    assert "nothing to do" in result.output


def test_redact_cmd_resolve_error_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = tmp_path / "notes.txt"
    src.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(cli_module, "resolve", _raise_resolve_error)

    result = runner.invoke(cli_module.app, [str(src)])

    assert result.exit_code == 1
    assert "broken" in result.output


def test_redact_cmd_pipeline_init_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = tmp_path / "notes.txt"
    src.write_text("hello", encoding="utf-8")

    class FailingPipeline:
        def __init__(self, *, checkpoint: str | None, device: str) -> None:
            raise RuntimeError("no model")

    monkeypatch.setattr(cli_module, "NERPipeline", FailingPipeline)
    _install_progress(monkeypatch)

    result = runner.invoke(cli_module.app, [str(src)])

    assert result.exit_code == 1
    assert "failed to load checkpoint" in result.output


def test_redact_cmd_skips_invalid_utf8_file_and_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_progress(monkeypatch)
    _install_pipeline(monkeypatch, [])

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    valid = corpus / "good.txt"
    invalid = corpus / "bad.txt"
    valid.write_text("hello", encoding="utf-8")
    invalid.write_bytes(b"\xff\xfe\xff")

    result = runner.invoke(cli_module.app, [str(corpus)])

    assert result.exit_code == 0
    assert (tmp_path / "corpus_redacted" / "good.redacted.txt").exists()
    assert not (tmp_path / "corpus_redacted" / "bad.redacted.txt").exists()
    assert "cannot decode" in result.output


def test_cli_main_invokes_app(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_app() -> None:
        called["ran"] = True

    monkeypatch.setattr(cli_module, "app", fake_app)

    assert cli_module.main() is None
    assert called["ran"] is True


def test_cli_module_guard_invokes_app(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_call(self, *args, **kwargs) -> None:
        called["ran"] = True

    monkeypatch.setattr(typer.main.Typer, "__call__", fake_call)
    runpy.run_path(Path(cli_module.__file__).resolve(), run_name="__main__")

    assert called["ran"] is True
