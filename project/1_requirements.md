# Requirements Document

## METADATA
- **Authors:** Shaojie Jiang
- **Project/Feature Name:** privacy-steward
- **Type:** Product
- **Summary:** A CLI tool that redacts PII from text files (single file or directory) using the `openai/privacy-filter` model via Hugging Face Transformers, storing sanitised output to a configurable location.
- **Owner:** Shaojie Jiang
- **Date Started:** 2026-05-06

## RELEVANT LINKS & STAKEHOLDERS

| Documents | Link | Owner |
|-----------|------|-------|
| HuggingFace Model | [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) | OpenAI |
| Reference Implementation | [github.com/openai/privacy-filter](https://github.com/openai/privacy-filter) | OpenAI |
| Design Document | [2_design.md](2_design.md) | Shaojie Jiang |
| Project Plan | [3_plan.md](3_plan.md) | Shaojie Jiang |

## PROBLEM DEFINITION

### Objectives
Build a developer-friendly CLI tool that automatically detects and redacts personally identifiable information (PII) from plain-text files, making it trivial to sanitise datasets, logs, or documents before sharing or downstream processing.

### Target users
- **Data engineers & ML practitioners** who need to strip PII from training corpora or log dumps before feeding them into pipelines.
- **Developers & QA engineers** who want to scrub production data for use in test environments.
- **Security / compliance teams** who must ensure files shared with third parties are PII-free.

### User Stories

| As a... | I want to... | So that... | Priority | Acceptance Criteria |
|---------|--------------|------------|----------|---------------------|
| Data engineer | Redact PII from a single text file via one CLI command | I can safely use it in a pipeline without manual review | P0 | `privacy-steward redact input.txt` produces `input.redacted.txt` with all detected entities replaced |
| Data engineer | Redact PII from an entire directory of text files | I can batch-process large datasets quickly | P0 | `privacy-steward redact ./corpus/` recurses and redacts every `.txt` file, preserving subdirectory structure |
| Developer | Specify an output path | Processed files land exactly where my pipeline expects them | P0 | `--output` / `-o` flag controls destination file or directory |
| Developer | Choose the redaction placeholder | I can distinguish entity types in downstream processing | P1 | `--placeholder` flag (default `[REDACTED]`) supports per-type templates e.g. `[{entity_type}]` |
| ML practitioner | Run redaction offline with no external API calls | Sensitive data never leaves the machine | P0 | All inference runs locally via the HuggingFace pipeline |
| Compliance officer | See a summary of what was redacted | I can audit the redaction pass | P1 | `--report` flag writes a JSON summary (file, entity type, count) alongside output |

### Context, Problems, Opportunities

PII leakage in text datasets is a pervasive risk. OpenAI released the `privacy-filter` NER model expressly for this purpose, but their reference implementation (`openai/privacy-filter` on GitHub) is a Python library without a first-class CLI or batch-directory workflow. Practitioners must write glue code to process directories, manage output paths, and integrate into shell pipelines.

`privacy-steward` closes this gap: it wraps the same model behind a polished CLI that handles the plumbing (recursive traversal, output layout, progress reporting) so that a single command can sanitise an entire corpus.

### Product Goals and Non-goals

**Goals:**
- Zero-friction PII redaction via a single installable CLI entry point.
- Support for single-file and recursive-directory processing.
- Configurable output location and placeholder format.
- Entirely offline — no network calls during inference.
- Detailed optional reporting for auditability.

**Non-goals (v1):**
- Non-text formats (PDF, DOCX, images, audio) — text files only.
- Custom model fine-tuning or model swapping beyond `openai/privacy-filter`.
- A web UI or REST API.
- Real-time streaming input (stdin) — file-path inputs only.

## PRODUCT DEFINITION

### Requirements

#### P0 — MVP

| ID | Requirement |
|----|-------------|
| R01 | `privacy-steward redact <path>` where `<path>` is a file or directory |
| R02 | Default output: alongside input with `.redacted` suffix before the extension (e.g. `notes.txt` → `notes.redacted.txt`) |
| R03 | `--output` / `-o <path>` overrides the destination (file when input is file; directory when input is directory) |
| R04 | Recursive directory traversal; only `.txt` files are processed; other files are ignored with a warning |
| R05 | Detected entities are replaced by `[REDACTED]` by default |
| R06 | All inference performed locally using `transformers.pipeline("token-classification", model="openai/privacy-filter")` |
| R07 | CLI prints a per-file progress line: filename, entity count, elapsed time |
| R08 | Non-zero exit code on any processing failure; errors are logged to stderr |

#### P1 — Subsequent iterations

| ID | Requirement |
|----|-------------|
| R09 | `--placeholder` / `-p` flag supports literal string or `{entity_type}` template |
| R10 | `--report` flag writes `redaction_report.json` to the output directory |
| R11 | `--dry-run` flag prints what would be redacted without writing files |
| R12 | `--workers` / `-w` flag for parallel file processing (default: CPU count) |
| R13 | `--model` flag allows overriding the HuggingFace model ID |

## TECHNICAL CONSIDERATIONS

### Architecture Overview

```
CLI entry point (Typer)
    │
    ├── Input resolver       — normalise file/dir paths, enumerate .txt files
    ├── NER pipeline         — load HF model once, run entity detection per file
    ├── Redaction engine     — span-based replacement with placeholder templating
    ├── Output writer        — write redacted text, mirror subdirectory structure
    └── Report builder (P1) — aggregate entity counts into JSON summary
```

All components are pure Python; the NER pipeline is the only heavy dependency.

### Technical Requirements

| Area | Requirement |
|------|-------------|
| Python | ≥ 3.12 |
| Core dependencies | `transformers`, `torch` (CPU), `typer`, `rich` |
| Packaging | Installable via `uv add privacy-steward` or `pip install privacy-steward`; exposes `privacy-steward` console script |
| Model caching | HuggingFace default cache (`~/.cache/huggingface`); first run triggers download |
| Platform | macOS, Linux; Windows best-effort |
| Encoding | UTF-8 only in v1; graceful error on other encodings |

### AI/ML Considerations

#### Data Requirements
No training data is needed — the project uses the pre-trained `openai/privacy-filter` model from HuggingFace Hub without fine-tuning. The model is downloaded on first use and cached locally.

#### Algorithm Selection
`openai/privacy-filter` is a NER (Named Entity Recognition) transformer model trained by OpenAI to detect PII entity spans. It is loaded via the HuggingFace `transformers` pipeline API (`pipeline("token-classification")`), which handles tokenisation, inference, and span aggregation. No alternative model is considered for v1.

#### Model Performance Requirements

| Metric | Target |
|--------|--------|
| Precision (PII detection) | ≥ 0.90 on held-out sample |
| Recall (PII detection) | ≥ 0.85 on held-out sample |
| Throughput | ≥ 5,000 tokens/second on CPU (single file) |
| Latency (first file, cold start) | < 10 s including model load |

## LAUNCH/ROLLOUT PLAN

### Success Metrics

| KPIs | Target & Rationale |
|------|--------------------|
| [Primary] CLI adoption (PyPI installs) | 100 installs within 30 days of public release |
| [Secondary] Batch throughput | Process 1 GB of plain text in < 5 min on M-series Mac |
| [Guardrail] False positive rate | < 5% non-PII tokens flagged as PII |

### Rollout Strategy

1. **Internal alpha**: install from source; validate accuracy and CLI UX on real datasets.
2. **PyPI beta**: publish `privacy-steward` package; gather feedback via GitHub Issues.
3. **GA**: stable release with full README benchmarks vs. OpenAI reference implementation.

## HYPOTHESIS & RISKS

**Hypothesis:** Wrapping `openai/privacy-filter` in a battery-included CLI will make PII redaction accessible to practitioners who currently skip the step due to integration friction. Adoption of the CLI should track inversely with the boilerplate required versus the OpenAI reference implementation.

**Risk 1 — Model accuracy ceiling:** The `openai/privacy-filter` model may have lower recall on domain-specific PII (e.g. internal employee IDs, non-English names). Mitigation: document known limitations; plan for `--model` override flag in P1 to allow swapping in a fine-tuned variant.

**Risk 2 — Large-file memory usage:** Loading transformers on CPU for very large files may exhaust RAM. Mitigation: chunk files at paragraph or sentence boundaries before inference; document memory requirements.

**Risk 3 — Encoding diversity:** Real-world text files use many encodings. Mitigation: fail fast with a clear error message pointing to `--encoding` flag (P1); default UTF-8.

## APPENDIX

- [HuggingFace `openai/privacy-filter` model card](https://huggingface.co/openai/privacy-filter)
- [OpenAI privacy-filter GitHub reference implementation](https://github.com/openai/privacy-filter)
- [HuggingFace `token-classification` pipeline docs](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TokenClassificationPipeline)
