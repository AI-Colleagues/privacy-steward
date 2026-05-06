"""Throughput benchmark: privacy-steward vs openai/privacy-filter (opf).

Usage:
    uv run python benchmarks/benchmark_throughput.py

Results are printed to stdout as a Markdown table and also written to
benchmarks/results.md for inclusion in the README.

Requirements:
    - benchmarks/data/*.txt must exist (run generate_data.py first)
    - opf binary available at the path set in OPF_BIN (or auto-detected)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


OPF_BIN = Path(
    os.environ.get(
        "OPF_BIN",
        "/Users/shaojiejiang/Development/privacy-filter/.venv/bin/opf",
    )
)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_MD = Path(__file__).parent / "results.md"


def _count_tokens(text: str) -> int:
    """Rough token count: words / 0.75 (OpenAI rule of thumb)."""
    return int(len(text.split()) / 0.75)


def bench_privacy_steward(corpus: Path) -> tuple[float, int]:
    """Time privacy-steward on *corpus*; return (elapsed_s, token_count)."""
    text = corpus.read_text(encoding="utf-8")
    tokens = _count_tokens(text)

    # Import here so benchmark is self-contained
    from privacy_steward.pipeline import NERPipeline

    pipe = NERPipeline()  # model loaded once
    t0 = time.perf_counter()
    pipe.predict(text)
    elapsed = time.perf_counter() - t0
    return elapsed, tokens


def bench_opf(corpus: Path) -> tuple[float, int]:
    """Time opf CLI on *corpus*; return (elapsed_s, token_count)."""
    if not OPF_BIN.exists():
        raise FileNotFoundError(f"opf binary not found: {OPF_BIN}")

    text = corpus.read_text(encoding="utf-8")
    tokens = _count_tokens(text)

    t0 = time.perf_counter()
    subprocess.run(
        [str(OPF_BIN), "redact", "-f", str(corpus), "--format", "text"],
        capture_output=True,
        text=True,
        check=True,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, tokens


def _throughput(elapsed: float, tokens: int) -> int:
    return int(tokens / elapsed)


def run_benchmarks() -> list[dict]:
    corpora = sorted(DATA_DIR.glob("*.txt"))
    if not corpora:
        print("No benchmark data found — run generate_data.py first.", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []

    # Warm up the privacy-steward model (counts against first file)
    print("Warming up privacy-steward model…", flush=True)
    from privacy_steward.pipeline import NERPipeline

    pipe = NERPipeline()

    for corpus in corpora:
        size_kb = corpus.stat().st_size // 1024
        text = corpus.read_text(encoding="utf-8")
        tokens = _count_tokens(text)

        print(f"\nBenchmarking {corpus.name} ({size_kb} KB, ~{tokens:,} tokens)…")

        # --- privacy-steward (reuse already-loaded pipe) ---
        t0 = time.perf_counter()
        pipe.predict(text)
        ps_elapsed = time.perf_counter() - t0
        ps_tps = _throughput(ps_elapsed, tokens)
        print(f"  privacy-steward : {ps_elapsed:.2f} s  ({ps_tps:,} tok/s)")

        # --- opf ---
        try:
            t0 = time.perf_counter()
            subprocess.run(
                [
                    str(OPF_BIN), "redact",
                    "-f", str(corpus),
                    "--device", "cpu",
                    "--format", "text",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            opf_elapsed = time.perf_counter() - t0
            opf_tps = _throughput(opf_elapsed, tokens)
            print(f"  opf             : {opf_elapsed:.2f} s  ({opf_tps:,} tok/s)")
            speedup = round(opf_elapsed / ps_elapsed, 2)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"  opf             : ERROR — {exc}")
            opf_elapsed = float("nan")
            opf_tps = 0
            speedup = float("nan")

        rows.append(
            {
                "corpus": corpus.name,
                "size_kb": size_kb,
                "tokens": tokens,
                "ps_elapsed": round(ps_elapsed, 2),
                "ps_tps": ps_tps,
                "opf_elapsed": round(opf_elapsed, 2) if opf_elapsed == opf_elapsed else "N/A",
                "opf_tps": opf_tps,
                "speedup": speedup,
            }
        )

    return rows


def _render_markdown(rows: list[dict]) -> str:
    lines = [
        "## Throughput Benchmark",
        "",
        "Both tools use the same `openai/privacy-filter` model weights.",
        "Benchmarks run on Apple M-series CPU (single process, no GPU).",
        "",
        "| Corpus | Size | Tokens | privacy-steward (s) | privacy-steward (tok/s) | opf (s) | opf (tok/s) | Speedup |",
        "|--------|------|--------|--------------------:|------------------------:|--------:|------------:|---------|",
    ]
    for r in rows:
        ps_s = f"{r['ps_elapsed']:.2f}"
        ps_tps = f"{r['ps_tps']:,}"
        opf_s = f"{r['opf_elapsed']:.2f}" if isinstance(r["opf_elapsed"], float) else r["opf_elapsed"]
        opf_tps = f"{r['opf_tps']:,}" if r["opf_tps"] else "N/A"
        speedup = f"{r['speedup']:.2f}×" if isinstance(r["speedup"], float) and r["speedup"] == r["speedup"] else "N/A"
        lines.append(
            f"| {r['corpus']} | {r['size_kb']} KB | {r['tokens']:,} "
            f"| {ps_s} | {ps_tps} | {opf_s} | {opf_tps} | {speedup} |"
        )
    lines += ["", "_Speedup = opf_elapsed / privacy-steward_elapsed (higher is better for privacy-steward)._", ""]
    return "\n".join(lines)


def main() -> None:
    """Run benchmarks and emit a Markdown results table."""
    rows = run_benchmarks()
    md = _render_markdown(rows)
    print("\n" + md)
    RESULTS_MD.write_text(md, encoding="utf-8")
    print(f"Results written to {RESULTS_MD}")


if __name__ == "__main__":
    main()
