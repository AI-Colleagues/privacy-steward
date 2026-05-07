# Design Document

## For privacy-steward CLI

- **Version:** 0.1
- **Author:** Shaojie Jiang
- **Date:** 2026-05-06
- **Status:** Approved

---

## Overview

`privacy-steward` is a command-line tool that detects and redacts PII (personally identifiable information) from plain-text files using the `openai/privacy-filter` NER model served locally through the Hugging Face `transformers` library. Users invoke a single `redact` command against a file or directory; the tool handles recursive traversal, entity span replacement, and output layout automatically.

The system is entirely offline — the model runs on the local CPU (or GPU if available) after a one-time download. No text is transmitted to any external service. This makes it suitable for handling sensitive data in air-gapped or compliance-constrained environments.

The design prioritises simplicity: one primary command, predictable output structure, and sensible defaults that require zero configuration for the common case.

## Components

- **CLI layer (`privacy_steward.cli`)** — Typer application exposing the `redact` command and all flags. Responsible for argument parsing, input validation, and orchestrating the pipeline. Uses `rich` for progress display and coloured output.

- **Input resolver (`privacy_steward.resolver`)** — Accepts a file or directory path and returns an ordered list of `(source_path, target_path)` tuples. Mirrors subdirectory structure under the configured output root. Warns (stderr) and skips non-`.txt` files.

- **NER pipeline (`privacy_steward.pipeline`)** — Thin wrapper around `transformers.pipeline("token-classification", model=..., aggregation_strategy="simple")`. Loaded once at startup and reused across all files. Returns a list of `EntitySpan` objects (start, end, entity type, score).

- **Redaction engine (`privacy_steward.redactor`)** — Pure function: given text and a list of `EntitySpan` objects, applies span replacements from right-to-left to preserve character offsets, and returns redacted text. Supports a configurable placeholder string or `{entity_type}` template.

- **Output writer (`privacy_steward.writer`)** — Creates destination directories if absent, writes redacted text as UTF-8, and (optionally) writes the JSON report.

- **Report builder (`privacy_steward.reporter`)** — Aggregates per-file entity counts into a summary dict and serialises to `redaction_report.json`.

## Request Flows

### Flow 1: Single-file redaction (default output)

```
$ privacy-steward redact notes.txt
```

1. CLI receives `path=notes.txt`; resolver confirms it is a file; target path is derived as `notes.redacted.txt` in the same directory.
2. NER pipeline is initialised (model loaded from HF cache or downloaded).
3. `notes.txt` is read as UTF-8 text.
4. Text is passed to the NER pipeline, which returns `EntitySpan` list.
5. Redaction engine replaces spans with `[REDACTED]` (right-to-left).
6. Output writer writes `notes.redacted.txt`.
7. CLI prints: `notes.txt → notes.redacted.txt  (12 entities, 0.42 s)`.

### Flow 2: Directory redaction with explicit output path

```
$ privacy-steward redact ./corpus/ --output ./corpus_clean/
```

1. CLI receives `path=./corpus/`, `output=./corpus_clean/`.
2. Resolver walks `./corpus/` recursively, collecting all `.txt` files; non-`.txt` files emit a warning to stderr. Produces list of `(src, dest)` pairs that mirror the directory tree under `./corpus_clean/`.
3. NER pipeline initialised once.
4. For each `(src, dest)` pair: read → infer → redact → write. `rich.Progress` bar updated.
5. After all files: report builder (if `--report`) writes `./corpus_clean/redaction_report.json`.
6. CLI prints summary: `23 files processed, 847 entities redacted, 14.3 s total`.

### Flow 3: Dry run

```
$ privacy-steward redact ./corpus/ --dry-run
```

Steps 1–4 as above, but output writer is skipped. CLI prints entity spans to stdout for each file.

## API Contracts

`privacy-steward` is a CLI tool, not a service. The internal Python API is documented below for programmatic use.

### `pipeline.NERPipeline`

```python
class NERPipeline:
    def __init__(self, model_id: str = "openai/privacy-filter") -> None: ...

    def predict(self, text: str) -> list[EntitySpan]: ...
```

### `redactor.redact`

```python
def redact(
    text: str,
    spans: list[EntitySpan],
    placeholder: str = "[REDACTED]",
) -> str: ...
```

### `resolver.resolve`

```python
def resolve(
    input_path: Path,
    output_path: Path | None,
) -> list[tuple[Path, Path]]: ...
```

## Data Models / Schemas

### `EntitySpan`

```python
@dataclass
class EntitySpan:
    start: int          # character offset in source text
    end: int            # character offset in source text (exclusive)
    entity_type: str    # e.g. "PER", "ORG", "LOC", "EMAIL", "PHONE"
    score: float        # model confidence [0, 1]
    word: str           # surface form (for logging / report)
```

### Redaction report (`redaction_report.json`)

```json
{
  "generated_at": "2026-05-06T12:00:00Z",
  "model": "openai/privacy-filter",
  "files": [
    {
      "source": "corpus/file1.txt",
      "destination": "corpus_clean/file1.redacted.txt",
      "entity_counts": {
        "PER": 4,
        "EMAIL": 2,
        "PHONE": 1
      },
      "total_entities": 7,
      "elapsed_seconds": 0.41
    }
  ],
  "totals": {
    "files_processed": 1,
    "entities_redacted": 7,
    "elapsed_seconds": 0.41
  }
}
```

### CLI flag schema

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--output` | `-o` | `Path` | derived | Output file or directory |
| `--placeholder` | `-p` | `str` | `[REDACTED]` | Replacement string; `{entity_type}` is interpolated |
| `--report` | | `bool` | `False` | Write JSON redaction report |
| `--dry-run` | | `bool` | `False` | Print spans without writing files |
| `--workers` | `-w` | `int` | `os.cpu_count()` | Parallel worker processes (P1) |
| `--model` | | `str` | `openai/privacy-filter` | HuggingFace model ID (P1) |
| `--encoding` | | `str` | `utf-8` | Input file encoding (P1) |

## Security Considerations

- **No network egress during inference** — all computation is local; the HuggingFace model is fetched once and cached.
- **Input validation** — paths are resolved to absolute form before use; symlink traversal is limited to the declared input directory to prevent directory traversal attacks.
- **No shell expansion** — file enumeration uses `pathlib`, not `glob` with shell expansion; there is no subprocess invocation on user-supplied strings.
- **Output isolation** — if `--output` points outside the current working tree, the user is prompted for confirmation (unless `--yes` flag is set).
- **Redacted text is not logged** — progress output only shows filenames and entity counts, never the redacted content.

## Performance Considerations

| Scenario | Expected throughput |
|----------|---------------------|
| Single file, CPU (M2 Pro) | ~8,000 tokens/s |
| Batch directory, 4 workers, CPU | ~28,000 tokens/s aggregate |
| GPU (CUDA) | ~60,000 tokens/s (via `--device cuda`) |

- **Model is loaded once** at startup, not per file; pipeline.predict() calls are the only hot path.
- **Chunking strategy** — files are split at paragraph boundaries (`\n\n`) for large documents (> 512 tokens) to stay within the model's context window; spans are stitched back with correct offsets.
- **Progress bar** uses `rich.Progress` with a spinner and ETA; it does not add measurable overhead.

## Testing Strategy

- **Unit tests** — `redactor.redact` is fully deterministic given a fixed span list; test all edge cases (overlapping spans, empty file, all-redacted file, non-ASCII characters).
- **Unit tests** — `resolver.resolve` for file/directory modes, explicit vs. derived output paths, non-`.txt` file filtering.
- **Integration tests** — end-to-end `subprocess` invocation of `privacy-steward redact` against fixture files; assert output text matches expected redactions and exit code is 0.
- **Model smoke test** — single sentence with a known name/email run through the full pipeline; assert the entity is detected with score > 0.5. Marked `@pytest.mark.slow` and skipped in CI unless `PRIVACY_STEWARD_SLOW_TESTS=1`.
- **Manual QA checklist:**
  - [ ] Single file → default output path correct
  - [ ] Directory → subdirectory structure mirrored
  - [ ] `--dry-run` writes nothing to disk
  - [ ] `--report` produces valid JSON
  - [ ] Non-`.txt` files produce a stderr warning, not an error
  - [ ] Large file (> 10 MB) completes without OOM

## Rollout Plan

1. **Phase 1 — Internal alpha:** Install from source via `uv sync`. Validate accuracy on internal test corpus. Iterate on CLI UX based on team feedback.
2. **Phase 2 — PyPI beta:** Publish `privacy-steward` to PyPI. Add benchmarking section to README. Open GitHub Issues for community feedback.
3. **Phase 3 — GA:** Stable `1.0.0` release. P1 features (`--workers`, `--model`, `--report`) complete. Automated CI/CD pipeline for release on tag push.

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-05-06 | Shaojie Jiang | Initial draft |
