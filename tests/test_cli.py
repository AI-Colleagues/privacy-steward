"""CLI tests that stub the model boundary."""

from __future__ import annotations
import json
from pathlib import Path
from privacy_steward import cli
from privacy_steward.models import EntitySpan


FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.txt"
CORPUS = FIXTURES / "corpus"


class FakePipeline:
    """Deterministic stand-in for the model-backed pipeline."""

    def __init__(self, model_id: str = cli.DEFAULT_MODEL) -> None:
        self.model_id = model_id

    def predict(self, text: str) -> list[EntitySpan]:
        spans: list[EntitySpan] = []
        needles = [
            ("Alice Johnson", "private_person"),
            ("alice.johnson@acme.com", "private_email"),
            ("+1-555-867-5309", "private_phone"),
            ("42 Maple Street", "private_address"),
            ("4532-1547-0823-6789", "account_number"),
            ("Bob Martinez", "private_person"),
            ("bob.martinez@personalmail.org", "private_email"),
            ("Carol Martinez", "private_person"),
            ("555-234-5678", "private_phone"),
            ("David Chen", "private_person"),
            ("support@techfirm.io", "private_email"),
            ("800-555-0199", "private_phone"),
        ]
        for needle, entity_type in needles:
            start = text.find(needle)
            if start == -1:
                continue
            spans.append(
                EntitySpan(
                    start=start,
                    end=start + len(needle),
                    entity_type=entity_type,
                    score=0.99,
                    word=needle,
                )
            )
        return spans


def _invoke(cli_runner, *args: str):
    return cli_runner.invoke(cli.app, list(args))


def test_single_file_creates_redacted_output(
    cli_runner, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())

    result = _invoke(cli_runner, str(src))

    assert result.exit_code == 0
    dest = tmp_path / "notes.redacted.txt"
    assert dest.exists()
    redacted = dest.read_text()
    assert "Alice Johnson" not in redacted
    assert "alice.johnson@acme.com" not in redacted
    assert "+1-555-867-5309" not in redacted


def test_single_file_explicit_output(cli_runner, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())
    out = tmp_path / "clean.txt"

    result = _invoke(cli_runner, str(src), "--output", str(out))

    assert result.exit_code == 0
    assert out.exists()


def test_single_file_audit_dir_created(cli_runner, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())

    result = _invoke(cli_runner, str(src))

    assert result.exit_code == 0
    audit_dir = tmp_path / ".audit"
    assert audit_dir.is_dir()
    audit_files = list(audit_dir.rglob("*.audit.json"))
    assert len(audit_files) == 1
    data = json.loads(audit_files[0].read_text())
    assert data["source"] == str(src.resolve())
    assert data["entities"]
    assert "word" not in data["entities"][0]


def test_single_file_include_text_in_audit_flag(
    cli_runner, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())

    result = _invoke(cli_runner, str(src), "--include-text-in-audit")

    assert result.exit_code == 0
    audit_file = next((tmp_path / ".audit").rglob("*.audit.json"))
    data = json.loads(audit_file.read_text())
    assert data["entities"][0]["word"]


def test_single_file_dry_run_writes_nothing(
    cli_runner, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())

    result = _invoke(cli_runner, str(src), "--dry-run")

    assert result.exit_code == 0
    assert not (tmp_path / "notes.redacted.txt").exists()
    assert not (tmp_path / ".audit").exists()


def test_single_file_verbose_output(cli_runner, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())

    result = _invoke(cli_runner, str(src), "-v")

    assert result.exit_code == 0
    assert "Done" in result.stdout


def test_single_file_entity_type_placeholder(
    cli_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    src = tmp_path / "notes.txt"
    src.write_text("My name is Alice Johnson.\n")

    result = _invoke(cli_runner, str(src), "--placeholder", "[{entity_type}]")

    assert result.exit_code == 0
    dest = tmp_path / "notes.redacted.txt"
    content = dest.read_text()
    assert "Alice Johnson" not in content
    assert "[PRIVATE_PERSON]" in content


def test_single_file_report_flag(cli_runner, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())

    result = _invoke(cli_runner, str(src), "--report")

    assert result.exit_code == 0
    report = tmp_path / "redaction_report.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["totals"]["files_processed"] == 1
    assert data["model"] == cli.DEFAULT_MODEL


def test_directory_mode_processes_all_txt_files(
    cli_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    import shutil

    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    corpus_copy = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus_copy)

    result = _invoke(cli_runner, str(corpus_copy))

    assert result.exit_code == 0
    out_root = tmp_path / "corpus_redacted"
    assert out_root.is_dir()
    assert (out_root / "chapter1.redacted.txt").exists()
    assert (out_root / "chapter2.redacted.txt").exists()
    assert (out_root / "subdir" / "chapter3.redacted.txt").exists()


def test_directory_mode_audit_mirrors_structure(
    cli_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    import shutil

    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    corpus_copy = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus_copy)

    result = _invoke(cli_runner, str(corpus_copy))

    assert result.exit_code == 0
    audit_root = tmp_path / "corpus_redacted" / ".audit"
    assert audit_root.is_dir()
    audit_files = list(audit_root.rglob("*.audit.json"))
    assert len(audit_files) == 3


def test_directory_mode_explicit_output(
    cli_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    import shutil

    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    corpus_copy = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus_copy)
    out = tmp_path / "sanitised"

    result = _invoke(cli_runner, str(corpus_copy), "--output", str(out))

    assert result.exit_code == 0
    assert (out / "chapter1.redacted.txt").exists()


def test_empty_directory_warns_and_exits_zero(
    cli_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    empty = tmp_path / "empty"
    empty.mkdir()

    result = _invoke(cli_runner, str(empty))

    assert result.exit_code == 0
    assert "no .txt files found" in result.output.lower()


def test_missing_path_exits_nonzero(cli_runner, monkeypatch) -> None:
    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)

    result = _invoke(cli_runner, "/tmp/definitely_does_not_exist_xyz.txt")

    assert result.exit_code == 1
    assert "path does not exist" in result.output.lower()


def test_resolve_errors_are_reported(cli_runner, monkeypatch, tmp_path: Path) -> None:
    import os

    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    fifo = tmp_path / "mystery"
    os.mkfifo(fifo)

    result = _invoke(cli_runner, str(fifo))

    assert result.exit_code == 1
    assert "neither a file nor a directory" in result.output.lower()


def test_model_load_failure_is_reported(
    cli_runner, monkeypatch, tmp_path: Path
) -> None:
    class RaisingPipeline:
        def __init__(self, model_id: str = cli.DEFAULT_MODEL) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(cli, "NERPipeline", RaisingPipeline)
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())

    result = _invoke(cli_runner, str(src))

    assert result.exit_code == 1
    assert "failed to load model" in result.output.lower()


def test_verbose_empty_file_reports_no_pii(
    cli_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    class EmptyPipeline(FakePipeline):
        def predict(self, text: str) -> list[EntitySpan]:
            return []

    monkeypatch.setattr(cli, "NERPipeline", EmptyPipeline)
    src = tmp_path / "notes.txt"
    src.write_text("plain text only\n")

    result = _invoke(cli_runner, str(src), "-v")

    assert result.exit_code == 0
    assert "No PII detected." in result.output


def test_verbose_report_prints_summary_path(
    cli_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())

    result = _invoke(cli_runner, str(src), "-v", "--report")

    assert result.exit_code == 0
    assert "Report written to" in result.output


def test_verbose_dry_run_omits_audit_message(
    cli_runner,
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    src = tmp_path / "notes.txt"
    src.write_text(SAMPLE.read_text())

    result = _invoke(cli_runner, str(src), "-v", "--dry-run")

    assert result.exit_code == 0
    assert "Audit records written" not in result.output


def test_cli_main_invokes_app(monkeypatch) -> None:
    called = {}

    def fake_app() -> None:
        called["value"] = True

    monkeypatch.setattr(cli, "app", fake_app)
    assert cli.main() is None
    assert called["value"] is True


def test_invalid_utf8_input_is_skipped(cli_runner, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "NERPipeline", FakePipeline)
    src = tmp_path / "notes.txt"
    src.write_bytes(b"\xff\xfe\xfd")

    result = _invoke(cli_runner, str(src))

    assert result.exit_code == 0
    assert "cannot decode" in result.output.lower()
