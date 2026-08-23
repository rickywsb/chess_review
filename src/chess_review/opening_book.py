"""Opening theory detection using the lichess chess-openings dataset."""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional

import chess
import chess.pgn


@dataclass
class OpeningInfo:
    eco: str
    name: str
    deviation_ply: Optional[int]      # first ply that left theory (1-based), None if in book to end
    deviation_side: Optional[bool]
    deviation_move_san: Optional[str]  # the out-of-book move the player chose
    book_move_san: Optional[str]       # a known theoretical continuation, if any
    last_book_ply: int                 # number of plies that stayed in theory


class OpeningBook:
    """In-memory index of opening positions.

    Built from the merged lichess `openings.tsv` (columns: eco, name, pgn).
    Every position along every known line is indexed so that theory deviation
    can be detected at any depth.
    """

    def __init__(self) -> None:
        self._positions: set[str] = set()          # epd of every in-book position
        self._endpoints: dict[str, tuple[str, str]] = {}  # epd -> (eco, name)
        self._children: dict[str, set[str]] = {}   # epd -> {san continuations}
        self.loaded = False

    # ---- loading ------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str] = None) -> "OpeningBook":
        book = cls()
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "data", "openings.tsv")
        if not os.path.exists(path):
            return book  # empty book -> deviation detection disabled
        with open(path, encoding="utf-8") as fh:
            book._parse(fh)
        book.loaded = True
        return book

    def _parse(self, fh: io.TextIOBase) -> None:
        header = fh.readline()
        cols = [c.strip().lower() for c in header.rstrip("\n").split("\t")]
        try:
            i_eco, i_name, i_pgn = cols.index("eco"), cols.index("name"), cols.index("pgn")
        except ValueError:
            i_eco, i_name, i_pgn = 0, 1, 2

        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= i_pgn:
                continue
            eco, name, pgn = parts[i_eco], parts[i_name], parts[i_pgn]
            self._index_line(eco, name, pgn)

    def _index_line(self, eco: str, name: str, pgn: str) -> None:
        board = chess.Board()
        self._positions.add(board.epd())
        game = chess.pgn.read_game(io.StringIO(f"{pgn} *"))
        if game is None:
            return
        for move in game.mainline_moves():
            parent_epd = board.epd()
            san = board.san(move)
            self._children.setdefault(parent_epd, set()).add(san)
            board.push(move)
            self._positions.add(board.epd())
        self._endpoints[board.epd()] = (eco, name)

    # ---- detection ----------------------------------------------------------
    def detect(self, moves: list[chess.Move]) -> Optional[OpeningInfo]:
        """Walk a game's moves and locate where it left theory."""
        if not self.loaded:
            return None

        board = chess.Board()
        eco, name = "", ""
        last_book_ply = 0
        last_book_epd = board.epd()

        for ply, move in enumerate(moves, start=1):
            parent_epd = board.epd()
            san = board.san(move)
            board.push(move)
            epd = board.epd()
            if epd in self._positions:
                last_book_ply = ply
                last_book_epd = epd
                if epd in self._endpoints:
                    eco, name = self._endpoints[epd]
                continue

            # This move left theory.
            book_children = self._children.get(parent_epd)
            book_move = sorted(book_children)[0] if book_children else None
            return OpeningInfo(
                eco=eco,
                name=name,
                deviation_ply=ply,
                deviation_side=not board.turn,  # side that just moved
                deviation_move_san=san,
                book_move_san=book_move,
                last_book_ply=last_book_ply,
            )

        # Whole game stayed within theory (rare/short games).
        return OpeningInfo(
            eco=eco,
            name=name,
            deviation_ply=None,
            deviation_side=None,
            deviation_move_san=None,
            book_move_san=None,
            last_book_ply=last_book_ply,
        )
