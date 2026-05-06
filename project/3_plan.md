# Project Plan

## For privacy-steward

- **Version:** 0.1
- **Author:** Shaojie Jiang
- **Date:** 2026-05-06
- **Status:** Draft

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

- [ ] Task 1.1: Update `pyproject.toml` — add `transformers`, `torch` (CPU), `typer`, `rich` as runtime dependencies
  - Dependencies: None
- [ ] Task 1.2: Create `src/privacy_steward/models.py` — define `EntitySpan` dataclass
  - Dependencies: Task 1.1
- [ ] Task 1.3: Create `src/privacy_steward/pipeline.py` — implement `NERPipeline` class wrapping `transformers.pipeline("token-classification", model="openai/privacy-filter", aggregation_strategy="simple")`
  - Dependencies: Task 1.2
- [ ] Task 1.4: Write unit test `tests/test_pipeline.py` — smoke test: known PII sentence → at least one entity returned with score > 0.5 (mark `@pytest.mark.slow`)
  - Dependencies: Task 1.3
- [ ] Task 1.5: Verify `make lint` and `make test` pass
  - Dependencies: Task 1.4

---

### Milestone 2: Core Redaction Engine & Single-File CLI

**Description:** Implement the redaction logic and wire it to a functional `privacy-steward redact <file>` command. Success criterion: `privacy-steward redact tests/fixtures/sample.txt` produces a correctly redacted output file alongside the input.

#### Task Checklist

- [ ] Task 2.1: Create `src/privacy_steward/redactor.py` — implement `redact(text, spans, placeholder)` with right-to-left span replacement
  - Dependencies: Milestone 1
- [ ] Task 2.2: Write unit tests for `redactor.redact` — edge cases: empty spans, overlapping spans, full-file redaction, non-ASCII text
  - Dependencies: Task 2.1
- [ ] Task 2.3: Create `src/privacy_steward/resolver.py` — implement `resolve(input_path, output_path)` for single-file mode (derive `.redacted.txt` default)
  - Dependencies: None
- [ ] Task 2.4: Write unit tests for `resolver.resolve` — file mode: explicit output, derived output, missing input raises
  - Dependencies: Task 2.3
- [ ] Task 2.5: Create `src/privacy_steward/cli.py` — Typer app with `redact` command; wire resolver → pipeline → redactor → writer for single-file case
  - Dependencies: Task 2.1, Task 2.3
- [ ] Task 2.6: Register `privacy-steward` console script in `pyproject.toml` pointing to `privacy_steward.cli:app`
  - Dependencies: Task 2.5
- [ ] Task 2.7: Add fixture file `tests/fixtures/sample.txt` with synthetic PII (names, emails, phone numbers)
  - Dependencies: None
- [ ] Task 2.8: Write integration test — invoke `privacy-steward redact tests/fixtures/sample.txt` via `subprocess`; assert output file exists and known PII strings absent
  - Dependencies: Task 2.5, Task 2.7
- [ ] Task 2.9: `make lint && make test` pass; update `README.md` with basic install + usage example
  - Dependencies: Task 2.8

---

### Milestone 3: Directory Batch Processing & Output Control

**Description:** Extend the CLI to handle a directory input, recursive traversal, configurable output path, and a progress bar. Success criterion: `privacy-steward redact ./corpus/ --output ./corpus_clean/` processes all `.txt` files and mirrors the directory structure.

#### Task Checklist

- [ ] Task 3.1: Extend `resolver.resolve` to handle directory input — recursive `.txt` enumeration, non-`.txt` warning, mirror tree under output root
  - Dependencies: Milestone 2
- [ ] Task 3.2: Write unit tests for `resolver.resolve` — directory mode: nested dirs, mixed file types, explicit output, derived output
  - Dependencies: Task 3.1
- [ ] Task 3.3: Add `--output` / `-o` flag to CLI; route to resolver
  - Dependencies: Task 3.1
- [ ] Task 3.4: Add `--placeholder` / `-p` flag; support `{entity_type}` template interpolation in `redactor.redact`
  - Dependencies: Task 2.1
- [ ] Task 3.5: Add `--dry-run` flag — skip writer, print entity spans to stdout
  - Dependencies: Task 2.5
- [ ] Task 3.6: Integrate `rich.Progress` bar for directory mode; show per-file progress and ETA
  - Dependencies: Task 3.3
- [ ] Task 3.7: Create `src/privacy_steward/reporter.py` and add `--report` flag; write `redaction_report.json` to output directory
  - Dependencies: Task 3.3
- [ ] Task 3.8: Add integration test for directory mode — `privacy-steward redact tests/fixtures/corpus/ --output /tmp/test_out/`; assert structure mirrored and all PII absent
  - Dependencies: Task 3.1, Task 3.6
- [ ] Task 3.9: `make lint && make test` pass
  - Dependencies: Task 3.8

---

### Milestone 4: Benchmarking, Documentation & Release

**Description:** Benchmark `privacy-steward` against the OpenAI reference implementation on accuracy (F1) and throughput, document results in `README.md`, and publish the package. Success criterion: `README.md` contains a reproducible benchmark table and the package is installable from PyPI.

#### Task Checklist

- [ ] Task 4.1: Write `benchmarks/benchmark_throughput.py` — measure tokens/second for single-file and directory modes on a synthetic 10 MB corpus; compare to `openai/privacy-filter` reference script
  - Dependencies: Milestone 3
- [ ] Task 4.2: Rewrite `README.md` — replace template content with: project overview, install instructions, quick-start examples, benchmark table, advantages vs. OpenAI reference implementation
  - Dependencies: Task 4.1
- [ ] Task 4.3: Benchmark table content to include in README (filled after Task 4.1):
  - UX comparison (CLI vs. library API)
  - Throughput (tokens/s): privacy-steward vs. reference implementation
  - Directory batch processing: supported vs. not supported
  - Installable package: yes vs. no
  - Offline operation: yes vs. yes
  - Dependencies: Task 4.1
- [ ] Task 4.4: Add GitHub Actions release workflow — publish to PyPI on version tag push
  - Dependencies: None
- [ ] Task 4.5: Tag `v0.1.0` and verify PyPI package installs cleanly via `pip install privacy-steward`
  - Dependencies: Task 4.2, Task 4.4

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-05-06 | Shaojie Jiang | Initial draft |
