"""Offline Polyglot opening-book backend.

Reads a standard Polyglot ``.bin`` book (popularity-weighted moves) via
``chess.polyglot``. This works fully offline, so it is a reliable fallback when
the lichess masters explorer is unreachable or the client IP is blocked.

The bundled ``data/komodo.bin`` is a broad ~578k-position book (Komodo opening
book by Salvo Spitaleri); point ``CHESS_REVIEW_BOOK`` (or the ``path`` argument)
at another ``.bin`` to use a different book.
"""
from __future__ import annotations

import os
from typing import Optional

import chess
import chess.polyglot

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "data", "komodo.bin")


class PolyglotBook:
    """Fail-soft reader for a Polyglot ``.bin`` opening book."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.environ.get("CHESS_REVIEW_BOOK") or _DEFAULT_PATH
        self.available = os.path.exists(self.path)

    def lookup(self, fen: str, top: int = 2) -> list[dict]:
        """Return the top weighted book moves at ``fen``.

        Each entry is ``{"san", "weight", "pct"}`` where ``pct`` is the move's
        share of the total weight at this position. Returns ``[]`` when the book
        is missing, the position is not in book, or the FEN is invalid.
        """
        if not self.available or not fen:
            return []
        try:
            board = chess.Board(fen)
        except ValueError:
            return []
        try:
            with chess.polyglot.open_reader(self.path) as reader:
                entries = sorted(reader.find_all(board), key=lambda e: -e.weight)
        except (OSError, ValueError):
            return []
        if not entries:
            return []
        total = sum(e.weight for e in entries) or 1
        out: list[dict] = []
        for e in entries[:top]:
            out.append({
                "san": board.san(e.move),
                "weight": e.weight,
                "pct": round(e.weight / total * 100, 1),
            })
        return out


_default_book: Optional[PolyglotBook] = None


def get_default_book() -> PolyglotBook:
    """Lazily construct and cache the default bundled Polyglot book."""
    global _default_book
    if _default_book is None:
        _default_book = PolyglotBook()
    return _default_book
