# Project Plan

## For privacy-steward

- **Version:** 0.1
- **Author:** Shaojie Jiang
- **Date:** 2026-05-06
- **Status:** Done

---

## Overview

Build a CLI tool that redacts PII from plain-text files using the `openai/privacy-filter` HuggingFace model. The project is structured in four milestones: foundation & model integration, core CLI and redaction engine, batch directory support, and finally benchmarking + documentation.

**Related Documents:**
- Requirements: [1_requirements.md](1_requirements.md)
- Design: [2_design.md](2_design.md)

---

## Milestones

### Milestone 1: Project Foundation & Model Integration

**Description:** Stand up the package skeleton, install dependencies, and verify that the `openai/privacy-filter` model can be loaded and run locally. Success criterion: a Python function accepts a string and returns a list of `EntitySpan` objects with correct offsets.

#### Task Checklist

- [x] Task 1.1: Update `pyproject.toml` — add `transformers`, `torch`, `typer`, `rich` as runtime dependencies; register `privacy-steward` console script
  - Dependencies: None
- [x] Task 1.2: Create `src/privacy_steward/models.py` — define `EntitySpan` dataclass
  - Dependencies: Task 1.1
- [x] Task 1.3: Create `src/privacy_steward/pipeline.py` — implement `NERPipeline` wrapping `transformers.pipeline("token-classification", aggregation_strategy="simple")` with paragraph-level chunking for long texts
  - Dependencies: Task 1.2
- [x] Task 1.4: Write `tests/test_pipeline.py` — `_split_paragraphs` unit tests (always run) + `NERPipeline` smoke tests (marked `@pytest.mark.slow`)
  - Dependencies: Task 1.3
- [x] Task 1.5: Verify `make lint` and `make test` pass
  - Dependencies: Task 1.4

---

### Milestone 2: Core Redaction Engine & Single-File CLI

**Description:** Implement the redaction logic and wire it to a functional `privacy-steward <file>` command. Success criterion: `privacy-steward tests/fixtures/sample.txt` produces a correctly redacted output file alongside the input.

> **Note:** Typer 0.12+ promotes single-command apps to root level, so the CLI syntax is `privacy-steward <path>` (not `privacy-steward redact <path>`).

#### Task Checklist

- [x] Task 2.1: Create `src/privacy_steward/redactor.py` — `redact(text, spans, placeholder)` with right-to-left span replacement; supports `{entity_type}` template
  - Dependencies: Milestone 1
- [x] Task 2.2: Write `tests/test_redactor.py` — edge cases: empty spans, multiple spans, adjacent spans, placeholder template, non-ASCII text
  - Dependencies: Task 2.1
- [x] Task 2.3: Create `src/privacy_steward/resolver.py` — `resolve(input_path, output_path)` for file mode; derive `.redacted.txt` default
  - Dependencies: None
- [x] Task 2.4: Write `tests/test_resolver.py` — file mode: explicit output, derived output, missing input raises
  - Dependencies: Task 2.3
- [x] Task 2.5: Create `src/privacy_steward/writer.py` and `src/privacy_steward/reporter.py`
  - Dependencies: None
- [x] Task 2.6: Create `src/privacy_steward/cli.py` — Typer app; always-on progress bar with ETA; wire resolver → pipeline → redactor → writer; `-v` prints per-file entity table
  - Dependencies: Task 2.1, Task 2.3
- [x] Task 2.7: Add fixture files `tests/fixtures/sample.txt` and `tests/fixtures/corpus/` (3 files, 1 subdirectory)
  - Dependencies: None
- [x] Task 2.8: Write `tests/test_cli.py` — integration tests via `subprocess` for single-file and directory modes (marked `@pytest.mark.slow`)
  - Dependencies: Task 2.6, Task 2.7
- [x] Task 2.9: `make lint && make test` pass (39/39)
  - Dependencies: Task 2.8

---

### Milestone 3: Directory Batch Processing & Output Control

**Description:** Extend the CLI to handle a directory input, recursive traversal, configurable output path, and a progress bar. Success criterion: `privacy-steward ./corpus/ --output ./corpus_clean/` processes all `.txt` files and mirrors the directory structure, writing audit JSONs to `.audit/`.

#### Task Checklist

- [x] Task 3.1: Extend `resolver.resolve` to handle directory input — recursive `.txt` enumeration (non-`.txt` silently skipped), mirror tree under output root
  - Dependencies: Milestone 2
- [x] Task 3.2: Write unit tests for `resolver.resolve` — directory mode: nested dirs, mixed file types, explicit output, derived output, empty dir
  - Dependencies: Task 3.1
- [x] Task 3.3: Add `--output` / `-o`, `--placeholder` / `-p`, `--dry-run`, `--verbose` / `-v`, `--model` flags to CLI
  - Dependencies: Task 3.1
- [x] Task 3.4: Always-on `rich.Progress` bar (spinner, bar, M/N, %, elapsed, ETA); `-v` adds per-file entity table printed to stdout
  - Dependencies: Task 3.3
- [x] Task 3.5: Automatic audit dir `.audit/` always created alongside output; per-file `<stem>.audit.json` mirrors output directory tree
  - Dependencies: Task 3.3
- [x] Task 3.6: `--report` flag writes aggregate `redaction_report.json` to output root via `reporter.write_summary_report`
  - Dependencies: Task 3.3
- [x] Task 3.7: Integration tests for directory mode — structure mirrored, audit dir created, 3 audit files for 3-file corpus
  - Dependencies: Task 3.1, Task 3.4
- [x] Task 3.8: `make lint && make test` pass (39/39)
  - Dependencies: Task 3.7

---

### Milestone 4: Benchmarking, Documentation & Release

**Description:** Benchmark `privacy-steward` against the OpenAI reference implementation on throughput, document results in `README.md`, and publish the package. Success criterion: `README.md` contains a reproducible benchmark table.

#### Task Checklist

- [x] Task 4.1: Write `benchmarks/generate_data.py` — generate 10 synthetic files × 10 emails each; write `benchmarks/benchmark_throughput.py` — measure tok/s for both tools, emit Markdown table to `benchmarks/results.md`
  - Dependencies: Milestone 3
- [x] Task 4.2: Rewrite `README.md` — project overview, install instructions, quick-start examples, output layout, options table, benchmark table, feature comparison vs. `opf`
  - Dependencies: Task 4.1
- [x] Task 4.3: Add GitHub Actions release workflow — publish to PyPI on version tag push
  - Dependencies: None
- [x] Task 4.4: Tag `v0.1.0` and verify PyPI package installs cleanly via `pip install privacy-steward`
  - Dependencies: Task 4.2, Task 4.3

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-05-06 | Shaojie Jiang | Initial draft |
| 2026-05-06 | Shaojie Jiang | M1–M4 (tasks 1.1–4.2) implemented; updated with actual implementation notes |
