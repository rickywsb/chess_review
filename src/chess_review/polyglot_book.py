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
                "uci": e.move.uci(),
                "weight": e.weight,
                "pct": round(e.weight / total * 100, 1),
            })
        return out

    def detect_deviation(self, moves: list[chess.Move]) -> Optional[dict]:
        """Locate where a game runs out of opening book ("脱谱点").

        A move keeps the game *in book* as long as the position it produces
        still has book continuations — even if the move itself was not the
        book's top choice (this correctly follows transpositions and tolerates
        narrow top-of-book coverage). The deviation is the first move after
        which the resulting position has **no** book moves left, i.e. there is
        no "main choice" to follow anymore. This typically lands 6-8+ moves in
        rather than firing on a normal early move like 2.Nf3.

        Returns a dict ``{deviation_ply, deviation_side, deviation_move_san,
        book_move_san, last_book_ply}`` (1-based plies), or ``None`` when the
        book is unavailable so callers can fall back to another source.
        """
        if not self.available:
            return None
        board = chess.Board()
        last_book_ply = 0
        try:
            reader = chess.polyglot.open_reader(self.path)
        except (OSError, ValueError):
            return None
        with reader:
            for ply, move in enumerate(moves, start=1):
                parent_entries = list(reader.find_all(board))
                if not parent_entries:
                    # Already out of book before this move (shouldn't happen,
                    # since we stop as soon as book ends); treat as exhausted.
                    break
                top = max(parent_entries, key=lambda e: e.weight)
                dev_side = board.turn          # side to move == side playing this move
                dev_san = board.san(move)
                book_san = board.san(top.move)
                board.push(move)
                if list(reader.find_all(board)):
                    # Resulting position still has main choices: stay in book.
                    last_book_ply = ply
                    continue
                # No book moves left after this move: the opening ends here.
                return {
                    "deviation_ply": ply,
                    "deviation_side": dev_side,
                    "deviation_move_san": dev_san,
                    "book_move_san": book_san,
                    "last_book_ply": last_book_ply,
                }
        return {
            "deviation_ply": None,
            "deviation_side": None,
            "deviation_move_san": None,
            "book_move_san": None,
            "last_book_ply": last_book_ply,
        }


_default_book: Optional[PolyglotBook] = None


def get_default_book() -> PolyglotBook:
    """Lazily construct and cache the default bundled Polyglot book."""
    global _default_book
    if _default_book is None:
        _default_book = PolyglotBook()
    return _default_book
