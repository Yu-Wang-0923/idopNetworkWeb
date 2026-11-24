"""Helpers shared by task modules."""
from __future__ import annotations

from pathlib import Path


def write_text(path: Path, text: str) -> Path:
    """Write ``text`` as UTF-8 exactly as given, creating parent folders.

    Newline translation stays off: ``DataFrame.to_csv()`` already ends lines
    with ``os.linesep``, so text mode would turn each CRLF into CR CR LF on
    Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return path
