"""CLI entry point for privacy-steward."""

from __future__ import annotations
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from privacy_steward.models import EntitySpan
from privacy_steward.pipeline import DEFAULT_MODEL, NERPipeline
from privacy_steward.redactor import redact
from privacy_steward.reporter import RedactionResult, write_audit, write_summary_report
from privacy_steward.resolver import output_root, resolve
from privacy_steward.writer import write_text


app = typer.Typer(
    name="privacy-steward",
    help="Redact PII from text files using openai/privacy-filter.",
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        try:
            v = _pkg_version("privacy-steward")
        except PackageNotFoundError:
            v = "unknown"
        typer.echo(f"privacy-steward {v}")
        raise typer.Exit()


_console = Console(stderr=False)
_err = Console(stderr=True)


def _make_progress() -> Progress:
    """Return a Rich Progress bar that estimates ETA from per-file timing."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=True),
        transient=False,
    )


def _resolve_pairs(input_path: Path, output: Path | None) -> list[tuple[Path, Path]]:
    """Validate input path and resolve source/destination pairs."""
    if not input_path.exists():
        _err.print(f"[red]Error:[/red] path does not exist: {input_path}")
        raise typer.Exit(code=1)

    try:
        pairs = resolve(input_path, output)
    except (FileNotFoundError, ValueError) as exc:
        _err.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not pairs:
        _err.print("[yellow]Warning:[/yellow] no .txt files found — nothing to do.")
        raise typer.Exit(code=0)
    return pairs


def _load_pipeline(model: str) -> NERPipeline:
    """Load the requested model or exit with a CLI-friendly error."""
    try:
        return NERPipeline(model_id=model)
    except Exception as exc:  # noqa: BLE001
        _err.print(f"[red]Error:[/red] failed to load model: {exc}")
        raise typer.Exit(code=1) from exc


def _print_completion(
    results: list[RedactionResult],
    *,
    dry_run: bool,
    audit_dir: Path,
) -> None:
    """Print the final verbose completion message."""
    total = sum(r["total_entities"] for r in results)
    elapsed_total = sum(r["elapsed_seconds"] for r in results)
    action = "would redact" if dry_run else "redacted"
    _console.print(
        f"\n[green]Done[/green] — {action} [bold]{total}[/bold] entities "
        f"across [bold]{len(results)}[/bold] file(s) in {elapsed_total:.2f} s"
    )
    if not dry_run and results:
        _console.print(f"Audit records written to [bold]{audit_dir}[/bold]")


def _process_file(
    *,
    pipe: NERPipeline,
    src: Path,
    dest: Path,
    placeholder: str,
    dry_run: bool,
    verbose: bool,
    audit_dir: Path,
    out_root: Path,
    include_text_in_audit: bool,
) -> RedactionResult | None:
    """Redact one source file and return its summary result."""
    t0 = time.perf_counter()
    try:
        text = src.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        _err.print(f"[red]Error:[/red] cannot decode {src} as UTF-8: {exc}")
        return None

    try:
        spans = pipe.predict(text)
    except ValueError as exc:
        _err.print(f"[red]Error:[/red] cannot process {src}: {exc}")
        return None

    redacted_text = redact(text, spans, placeholder)
    elapsed = time.perf_counter() - t0

    if not dry_run:
        write_text(dest, redacted_text)
        write_audit(
            src,
            spans,
            audit_dir,
            out_root,
            dest,
            include_text=include_text_in_audit,
        )

    entity_counts: dict[str, int] = {}
    for span in spans:
        entity_counts[span.entity_type] = entity_counts.get(span.entity_type, 0) + 1

    if verbose:
        _print_file_details(src, dest, spans, elapsed, dry_run)

    return {
        "source": str(src),
        "destination": str(dest),
        "entity_counts": entity_counts,
        "total_entities": len(spans),
        "elapsed_seconds": round(elapsed, 3),
    }


@app.command()
def redact_cmd(  # noqa: PLR0913
    input_path: Annotated[Path, typer.Argument(help="File or directory to redact.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file or directory."),
    ] = None,
    placeholder: Annotated[
        str,
        typer.Option(
            "--placeholder",
            "-p",
            help="Replacement string. Use {entity_type} for label interpolation.",
        ),
    ] = "<{entity_type}>",
    report: Annotated[
        bool,
        typer.Option(
            "--report", help="Write aggregate redaction_report.json to output dir."
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Show what would be redacted without writing files."
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Print per-file entity details alongside the progress bar.",
        ),
    ] = False,
    include_text_in_audit: Annotated[
        bool,
        typer.Option(
            "--include-text-in-audit",
            help="Include original matched text in audit JSON files.",
        ),
    ] = False,
    model: Annotated[
        str,
        typer.Option("--model", help="HuggingFace model ID."),
    ] = DEFAULT_MODEL,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """Redact PII from INPUT_PATH (file or directory of .txt files).

    A progress bar with per-file ETA is always shown on stderr.
    Use -v to additionally print the detected entities for each file.
    """
    pairs = _resolve_pairs(input_path, output)
    pipe = _load_pipeline(model)

    out_root = output_root(pairs)
    audit_dir = out_root / ".audit"
    results: list[RedactionResult] = []

    with _make_progress() as progress:
        prefix = "(dry-run) " if dry_run else ""
        task = progress.add_task(
            f"{prefix}[cyan]Starting…[/cyan]",
            total=len(pairs),
        )

        for src, dest in pairs:
            progress.update(task, description=f"{prefix}[cyan]{src.name}[/cyan]")
            result = _process_file(
                pipe=pipe,
                src=src,
                dest=dest,
                placeholder=placeholder,
                dry_run=dry_run,
                verbose=verbose,
                audit_dir=audit_dir,
                out_root=out_root,
                include_text_in_audit=include_text_in_audit,
            )
            if result is not None:
                results.append(result)
            progress.advance(task)

    if report and results and not dry_run:
        report_path = write_summary_report(results, out_root, model)
        if verbose:
            _console.print(f"Report written to [bold]{report_path}[/bold]")

    if verbose:
        _print_completion(results, dry_run=dry_run, audit_dir=audit_dir)


def _print_file_details(
    src: Path,
    dest: Path,
    spans: list[EntitySpan],
    elapsed: float,
    dry_run: bool,
) -> None:
    header = f"{'(dry-run) ' if dry_run else ''}{src.name} → {dest.name}"
    _console.rule(f"[bold]{header}[/bold]")
    if not spans:
        _console.print("  No PII detected.")
        return
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Entity type", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Surface form")
    for s in spans:
        table.add_row(s.entity_type.upper(), f"{s.score:.4f}", repr(s.word))
    _console.print(table)
    _console.print(
        f"  [dim]{len(spans)} entity/entities detected in {elapsed:.3f} s[/dim]"
    )


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
