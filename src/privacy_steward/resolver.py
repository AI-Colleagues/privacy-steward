"""Input/output path resolution for file and directory modes."""

from __future__ import annotations
import os
from pathlib import Path


def resolve(
    input_path: Path,
    output_path: Path | None = None,
) -> list[tuple[Path, Path]]:
    """Return ``(source, destination)`` pairs for all files to process.

    File mode: returns one pair; destination defaults to
    ``<stem>.redacted<suffix>`` alongside the source.

    Directory mode: recursively enumerates ``.txt`` files and mirrors
    the directory tree under the output root.  Non-``.txt`` files are
    silently skipped.
    """
    input_path = input_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix != ".txt":
            raise ValueError(f"File input must be a .txt file: {input_path}")
        if output_path is not None and output_path.exists() and output_path.is_dir():
            dest = output_path.resolve() / _default_file_dest(input_path).name
        else:
            dest = (
                output_path.resolve()
                if output_path is not None
                else _default_file_dest(input_path)
            )
        return [(input_path, dest)]

    if input_path.is_dir():
        out_root = (
            output_path.resolve()
            if output_path is not None
            else input_path.parent / f"{input_path.name}_redacted"
        )
        pairs: list[tuple[Path, Path]] = []
        for src in sorted(input_path.rglob("*.txt")):
            rel = src.relative_to(input_path)
            dest = out_root / rel.with_name(f"{rel.stem}.redacted{rel.suffix}")
            pairs.append((src, dest))
        return pairs

    raise ValueError(f"Input path is neither a file nor a directory: {input_path}")


def output_root(pairs: list[tuple[Path, Path]]) -> Path:
    """Return the common parent directory of all destination paths."""
    if not pairs:
        raise ValueError("No file pairs to derive output root from.")
    dest_parents = [str(dest.parent) for _, dest in pairs]
    return Path(os.path.commonpath(dest_parents))


def _default_file_dest(src: Path) -> Path:
    return src.with_name(f"{src.stem}.redacted{src.suffix}")
