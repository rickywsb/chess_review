"""Move classification and phase detection thresholds."""
from __future__ import annotations

import chess

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
