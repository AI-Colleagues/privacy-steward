# privacy-steward

[![CI](https://github.com/AI-Colleagues/privacy-steward/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/AI-Colleagues/privacy-steward/actions/workflows/ci.yml?query=branch%3Amain)

A CLI tool that redacts PII from plain-text files using the official
[`openai/privacy-filter`](https://github.com/openai/privacy-filter) runtime.
All inference runs locally — no data ever leaves your machine.

---

## Why privacy-steward?

The official OpenAI `opf` CLI processes one file at a time, requires manual installation, and
offers limited placeholder control. `privacy-steward` is a drop-in alternative built for
practitioners who need to sanitise datasets at scale:

| | privacy-steward | opf |
|---|:-:|:-:|
| **Zero-install one-liner** (`uvx`) | ✓ | — |
| **Directory batch processing** | ✓ | — |
| **Progress bar with ETA** | ✓ | — |
| **Automatic per-file audit trail** | ✓ | — |
| **Typed placeholders** (`<PRIVATE_PERSON>`, `<ACCOUNT_NUMBER>`, …) | ✓ | fixed format |
| **`--checkpoint` flag** for an official OPF checkpoint | ✓ | — |
| **`--device cpu`** for CPU-only inference | ✓ | — |
| Offline inference | ✓ | ✓ |

---

## Installation

You can run the tool without installing anything:

```bash
uvx privacy-steward notes.txt
uv run privacy-steward notes.txt
```

```bash
# Permanent install — adds `privacy-steward` to your PATH
uv tool install privacy-steward
```

First run downloads the default `openai/privacy-filter` checkpoint (~500 MB) and caches it in
`~/.opf/privacy_filter/`. Subsequent runs are fully offline.

---

## Quick start

```bash
# Try without installing — runs in an ephemeral environment
uvx privacy-steward notes.txt

# Redact a single file (output: notes.redacted.txt alongside the source)
privacy-steward notes.txt

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
Please send the invoice to account number <ACCOUNT_NUMBER>.
My API key is <SECRET>.
```

Pass `--placeholder` to override: any literal string, or use `{entity_type}` for interpolation
(e.g. `--placeholder "[{entity_type}]"` → `[PRIVATE_PERSON]`).

### Output layout

For a directory input the redacted files mirror the source tree, and an `.audit/`
directory is always created alongside them:

```
corpus_clean/
├── chapter1.redacted.txt
├── chapter2.redacted.txt
├── subdir/
│   └── chapter3.redacted.txt
└── .audit/
    ├── chapter1.audit.json    ← raw entity spans for auditing
    ├── chapter2.audit.json
    └── subdir/
        └── chapter3.audit.json
```

Each audit JSON records the source path, destination path, and every detected span
(character offsets, entity type, confidence score, surface form).

---

## Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--output` | `-o` | derived | Output file or directory |
| `--placeholder` | `-p` | `<{entity_type}>` | Replacement string; `{entity_type}` is interpolated |
| `--report` | | off | Write `redaction_report.json` to output dir |
| `--dry-run` | | off | Show what would be redacted without writing files |
| `--verbose` | `-v` | off | Print per-file entity details alongside the progress bar |
| `--checkpoint` | | derived from `OPF_CHECKPOINT` / `~/.opf/privacy_filter` | Official OPF checkpoint directory |
| `--device` | | `cpu` | Inference device for the official OPF runtime |

---

## Development

```bash
uv sync                                  # install all deps
make lint                                # ruff + mypy
make test                                # fast unit tests only
pytest -m slow                           # integration tests (require model)
```
