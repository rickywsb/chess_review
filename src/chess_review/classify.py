"""Move classification and phase detection thresholds."""
from __future__ import annotations

from typing import TYPE_CHECKING

import chess

if TYPE_CHECKING:
    from .models import MoveAnalysis

# Centipawn-loss thresholds (1 pawn = 100 cp), matching the report definitions:
#   "漏着" / blunder  : loss >= 200 (>= 2 pawns)
#   "错着" / mistake  : 100 <= loss < 200 (1-2 pawns)
#   inaccuracy        : 50 <= loss < 100
BLUNDER = 200
MISTAKE = 100
INACCURACY = 50
GOOD = 10  # <= this is considered essentially best


def classify_loss(cp_loss: int) -> str:
    if cp_loss >= BLUNDER:
        return "blunder"
    if cp_loss >= MISTAKE:
        return "mistake"
    if cp_loss >= INACCURACY:
        return "inaccuracy"
    if cp_loss <= GOOD:
        return "best"
    return "good"


# ---- phase detection --------------------------------------------------------
# Opening   = still following book theory AND within the first 15 full moves
#             (once a side leaves book, or after move 15, we call it middlegame).
# Endgame   = queens are off the board, OR the game has reached move 50, OR very
#             little material remains even with queens on.
# Otherwise = middlegame.
OPENING_LAST_MOVE = 15
ENDGAME_MOVE = 50
ENDGAME_MATERIAL_WITH_QUEENS = 20

_PIECE_POINTS = {
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def non_pawn_material(board: chess.Board) -> int:
    total = 0
    for piece_type, pts in _PIECE_POINTS.items():
        total += len(board.pieces(piece_type, chess.WHITE)) * pts
        total += len(board.pieces(piece_type, chess.BLACK)) * pts
    return total


def has_queens(board: chess.Board) -> bool:
    return bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))


def classify_phase(board: chess.Board, in_book: bool = True,
                   book_loaded: bool = False) -> str:
    """Classify the phase of `board` (the position before a move is played).

    `in_book` says whether the position is still within opening theory; it is
    only trusted when `book_loaded` is True. When no book is available we fall
    back to the move-number heuristic for the opening.
    """
    mn = board.fullmove_number

    # ---- endgame (checked first) -------------------------------------------
    # Queens off the board is the classic endgame signal; move 50+ or almost no
    # material also qualify.
    if not has_queens(board):
        return "endgame"
    if mn >= ENDGAME_MOVE:
        return "endgame"
    if non_pawn_material(board) <= ENDGAME_MATERIAL_WITH_QUEENS:
        return "endgame"

    # ---- opening -----------------------------------------------------------
    if mn <= OPENING_LAST_MOVE and (in_book or not book_loaded):
        return "opening"

    # ---- middlegame --------------------------------------------------------
    return "middlegame"


# ---------------------------------------------------------------------------
# significance / selection layer
# ---------------------------------------------------------------------------
# Outcome zones from the mover's point of view. A move only deserves a full
# write-up when it *changes the outcome* — throws away a win, drops from a
# holdable position into a losing one, or misses a forced mate that also costs
# the decisive edge. Large centipawn swings that keep the game in the same
# decisive zone (e.g. +9.45 -> +7.28, both completely winning) are noise.
WIN_CP = 300     # >= this = decisive advantage (胜势)
EDGE_CP = 100    # >= this = clear advantage; within ±EDGE_CP = balanced (均势)

# Why a flagged move matters — drives differentiated language downstream and,
# later, grounds the LLM phrasing pass.
TAG_ZH = {
    "missed_mate": "错过强制杀",
    "threw_game": "把局面走坏了",
    "lost_win": "让到手的优势溜走",
    "big_error": "严重失误",
}


def outcome_zone(cp: int, mate: "int | None") -> int:
    """Bucket an eval (mover POV) into an outcome zone: 2=胜势, 1=优势, 0=均势,
    -1=劣势, -2=败势. Forced mates collapse to the extreme zones."""
    if mate is not None:
        return 2 if mate > 0 else -2
    if cp >= WIN_CP:
        return 2
    if cp >= EDGE_CP:
        return 1
    if cp > -EDGE_CP:
        return 0
    if cp > -WIN_CP:
        return -1
    return -2


def significance(m: "MoveAnalysis", threshold: int = MISTAKE) -> "tuple[bool, str]":
    """Decide whether a move is worth commenting on, and tag *why* it matters.

    Returns ``(keep, tag)``. Selection is based on whether the move changed the
    *outcome* of the game, not on the raw centipawn loss — a big drop that keeps
    the game in the same decisive zone is filtered out as noise."""
    before = outcome_zone(m.eval_before_mover, m.mate_before)
    after = outcome_zone(m.eval_after_mover, m.mate_after)

    # Threw away a forced mate — but only worth noting if it also cost the
    # decisive edge. Giving up a mate-in-6 while still up a queen (+6) is not
    # instructive; the win was never in doubt.
    if m.mate_before is not None and m.mate_before > 0 and \
            (m.mate_after is None or m.mate_after <= 0) and after < 2:
        return True, "missed_mate"
    # Was at least equal, now worse than equal — fell into real trouble.
    if before >= 0 and after <= -1:
        return True, "threw_game"
    # Had a genuine advantage, now no better than equal — let the win slip.
    if before >= 1 and after <= 0:
        return True, "lost_win"
    # Same broad zone and still crushing / already lost: nitpicking, skip it.
    if before >= 2 and after >= 2:
        return False, "still_winning"
    if before <= -2 and after <= -2:
        return False, "already_lost"
    # Otherwise flag only outright blunders that stayed in the same zone.
    if m.cp_loss >= max(BLUNDER, threshold):
        return True, "big_error"
    return False, "minor"
