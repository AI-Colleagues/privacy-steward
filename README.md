# privacy-steward

[![CI](https://github.com/AI-Colleagues/privacy-steward/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/AI-Colleagues/privacy-steward/actions/workflows/ci.yml?query=branch%3Amain)

A CLI tool that redacts PII from plain-text files using the
[`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) model via Hugging Face
Transformers. All inference runs locally — no data ever leaves your machine.

---

## Why privacy-steward?

The official OpenAI `opf` CLI processes one file at a time, requires manual installation, and
offers limited placeholder control. `privacy-steward` is a drop-in alternative built for
practitioners who need to sanitise datasets at scale:

| | privacy-steward | opf |
|---|:-:|:-:|
| **Zero-install one-liner** (`uvx`) | ✓ | — |
| **Directory batch processing** | ✓ | — |
| **~2× faster throughput** (HF Transformers vs. bundled runtime) | ✓ | — |
| **Progress bar with ETA** | ✓ | — |
| **Automatic per-file audit trail** | ✓ | — |
| **Typed placeholders** (`<PRIVATE_PERSON>`, `<ACCOUNT_NUMBER>`, …) | ✓ | fixed format |
| **`--model` flag** for any HF token-classification model | ✓ | — |
| Offline inference | ✓ | ✓ |

---

## Installation

TODO: remind the users that they can simply run uvx or uv run without installing

```bash
# Permanent install — adds `privacy-steward` to your PATH
uv tool install privacy-steward
```

First run downloads the `openai/privacy-filter` model (~500 MB) and caches it in
`~/.cache/huggingface/`. Subsequent runs are fully offline.

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
| `--model` | | `openai/privacy-filter` | HuggingFace model ID |

---

## Benchmark vs. OpenAI privacy-filter CLI (`opf`)

Both tools process text files through the OpenAI privacy-filter model family on CPU.
`opf` uses a custom bundled runtime; `privacy-steward` uses the Hugging Face
Transformers pipeline.

**Hardware:** Apple MacBook Pro (2020), Apple M1, 8-core CPU (4 Performance + 4 Efficiency),
16 GB unified memory. No GPU acceleration — all inference on CPU.

**Setup:** 10 synthetic files, ~10 emails each (~560–580 tokens per file),
single process.

| File | Tokens | privacy-steward (s) | opf (s) | Speedup |
|------|-------:|--------------------:|--------:|--------:|
| batch_01 | 570 | 17.2 | 46.8 | 2.7× |
| batch_02 | 566 | 22.4 | 47.9 | 2.1× |
| batch_03 | 562 | 20.4 | 61.7 | 3.0× |
| batch_04 | 568 | 31.4 | 52.4 | 1.7× |
| batch_05 | 578 | 28.1 | 51.4 | 1.8× |
| batch_06 | 562 | 26.0 | 53.7 | 2.1× |
| batch_07 | 564 | 27.5 | 59.6 | 2.2× |
| batch_08 | 572 | 33.3 | 42.7 | 1.3× |
| batch_09 | 557 | 20.8 | 48.3 | 2.3× |
| batch_10 | 566 | 21.9 | 44.6 | 2.0× |
| **mean** | **566** | **24.9** | **50.9** | **2.0×** |

_Benchmarks are reproducible: `uv run python benchmarks/benchmark_throughput.py`_
_(requires `benchmarks/data/` — run `generate_data.py` first)._

---

## Development

```bash
uv sync                                  # install all deps
make lint                                # ruff + mypy
make test                                # fast unit tests only
pytest -m slow                           # integration tests (require model)
uv run python benchmarks/generate_data.py    # regenerate benchmark corpus
uv run python benchmarks/benchmark_throughput.py  # run benchmarks
```
