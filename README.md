# privacy-steward

[![CI](https://github.com/AI-Colleagues/privacy-steward/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/AI-Colleagues/privacy-steward/actions/workflows/ci.yml?query=branch%3Amain)

A CLI tool that redacts PII from plain-text files using the
[`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) model via a native
PyTorch implementation. All inference runs locally — no data ever leaves your machine.

---

## Why privacy-steward?

The official OpenAI `opf` CLI processes one file at a time, requires manual installation, and
offers limited placeholder control. `privacy-steward` is a drop-in alternative built for
practitioners who need to sanitise datasets at scale:

| | privacy-steward | opf |
|---|:-:|:-:|
| **Zero-install one-liner** (`uvx`) | ✓ | — |
| **Directory batch processing** | ✓ | — |
| **~1.4× faster throughput** (native PyTorch vs. bundled runtime) | ✓ | — |
| **Progress bar with ETA** | ✓ | — |
| **Automatic per-file audit trail** | ✓ | — |
| **Typed placeholders** (`<PRIVATE_PERSON>`, `<ACCOUNT_NUMBER>`, …) | ✓ | fixed format |
| **`--model` flag** for any HF token-classification model | ✓ | — |
| Offline inference | ✓ | ✓ |

---

## Installation

No installation required — run directly with `uvx`:

```bash
uvx privacy-steward notes.txt
```

Or install permanently to add `privacy-steward` to your PATH:

```bash
uv tool install privacy-steward
```

First run downloads the `openai/privacy-filter` model weights and caches them in
`~/.cache/huggingface/`. Subsequent runs are fully offline.

> **Requirements:** Python 3.12 or later.

---

## Quick start

```bash
# Redact a single .txt file (output: notes.redacted.txt alongside the source)
privacy-steward notes.txt

# Write a single-file result into an existing output directory
privacy-steward notes.txt --output ./clean/

# Redact an entire directory of .txt files, write to a custom output location
privacy-steward ./corpus/ --output ./corpus_clean/

# Show detected entities as they are processed (-v)
privacy-steward notes.txt -v

# Preview without writing files
privacy-steward ./corpus/ --dry-run

# Write an aggregate JSON summary report
privacy-steward ./corpus/ --output ./corpus_clean/ --report
```

### Default output format

By default each detected entity is replaced with a typed label that reflects what was found:

```
Hi, my name is <PRIVATE_PERSON> and I work at Acme Corp.
You can reach me at <PRIVATE_EMAIL> or call me at <PRIVATE_PHONE>.
My home address is <PRIVATE_ADDRESS>.
The meeting is on <PRIVATE_DATE>. Visit us at <PRIVATE_URL>.
Please send the invoice to account number <ACCOUNT_NUMBER>.
My API key is <SECRET>.
```

Pass `--placeholder` to override: any literal string, or use `{entity_type}` for interpolation
(e.g. `--placeholder "[{entity_type}]"` → `[PRIVATE_PERSON]`).

### Output layout

For a single-file input, the input must be a `.txt` file. For a directory input,
redacted `.txt` files mirror the source tree, non-`.txt` files are skipped, and an
`.audit/` directory is always created alongside the redacted outputs:

```
corpus_clean/
├── chapter1.redacted.txt
├── chapter2.redacted.txt
├── subdir/
│   └── chapter3.redacted.txt
└── .audit/
    ├── chapter1.audit.json    ← offsets, labels, and scores for auditing
    ├── chapter2.audit.json
    └── subdir/
        └── chapter3.audit.json
```

Each audit JSON records the source path, destination path, and every detected span
(character offsets, entity type, and confidence score). To avoid re-exposing the PII
that was just redacted, audit records omit the original matched text by default. Pass
`--include-text-in-audit` only when you intentionally need surface forms in the audit
trail and can protect the `.audit/` directory accordingly.

---

## Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--output` | `-o` | derived | Output file or directory |
| `--placeholder` | `-p` | `<{entity_type}>` | Replacement string; `{entity_type}` is interpolated |
| `--report` | | off | Write `redaction_report.json` to output dir |
| `--dry-run` | | off | Show what would be redacted without writing files |
| `--verbose` | `-v` | off | Print per-file entity details alongside the progress bar |
| `--include-text-in-audit` | | off | Include original matched text in audit JSON files |
| `--model` | | `openai/privacy-filter` | HuggingFace model ID |

---

## Benchmark vs. OpenAI privacy-filter CLI (`opf`)

Both tools process text files through the OpenAI privacy-filter model family on CPU.
`opf` uses a custom bundled runtime; `privacy-steward` uses a native PyTorch
implementation loaded directly from the model's safetensors weights.

**Hardware:** Apple MacBook Pro (2020), Apple M1, 8-core CPU (4 Performance + 4 Efficiency),
16 GB unified memory. No GPU acceleration — all inference on CPU.

**Setup:** 10 synthetic files across diverse document types (emails, chat logs, support
tickets, contracts, invoices, etc.), 573–1,362 tokens per file, single process.

| Corpus | Size | Tokens | privacy-steward (s) | privacy-steward (tok/s) | opf (s) | opf (tok/s) | Speedup |
|--------|------|--------|--------------------:|------------------------:|--------:|------------:|---------|
| 01_emails.txt | 5 KB | 990 | 22.48 | 44 | 37.34 | 26 | 1.66× |
| 02_chat_logs.txt | 6 KB | 1,168 | 33.13 | 35 | 42.16 | 27 | 1.27× |
| 03_support_tickets.txt | 4 KB | 657 | 22.13 | 29 | 29.55 | 22 | 1.34× |
| 04_meeting_notes.txt | 5 KB | 969 | 25.13 | 38 | 33.39 | 29 | 1.33× |
| 05_contracts.txt | 5 KB | 954 | 22.44 | 42 | 29.53 | 32 | 1.32× |
| 06_invoices.txt | 3 KB | 573 | 19.02 | 30 | 27.60 | 20 | 1.45× |
| 07_intake_forms.txt | 4 KB | 750 | 20.84 | 35 | 28.88 | 25 | 1.39× |
| 08_travel_itineraries.txt | 4 KB | 604 | 19.44 | 31 | 27.82 | 21 | 1.43× |
| 09_incident_reports.txt | 5 KB | 836 | 23.25 | 35 | 31.54 | 26 | 1.36× |
| 10_crm_diary.txt | 7 KB | 1,362 | 30.45 | 44 | 40.73 | 33 | 1.34× |

_Benchmarks are reproducible: `uv run python benchmarks/benchmark_throughput.py`_
_(requires `benchmarks/data/` to be present)._

---

## Development

```bash
uv sync                                  # install all deps
make lint                                # ruff + mypy
make test                                # fast unit tests only
pytest -m slow                           # integration tests (require model)
uv run python benchmarks/benchmark_throughput.py  # run benchmarks
```
