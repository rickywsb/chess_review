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
# Opening   = first 15 full moves.
# Endgame   = total non-pawn material (both sides, kings excluded) <= 26,
#             or <= 20 while any queen remains on the board.
# Otherwise = middlegame.
OPENING_LAST_MOVE = 15
ENDGAME_MATERIAL = 26
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


def classify_phase(board: chess.Board) -> str:
    """Classify the phase of `board` (the position before a move is played)."""
    if board.fullmove_number <= OPENING_LAST_MOVE:
        return "opening"
    npm = non_pawn_material(board)
    threshold = ENDGAME_MATERIAL_WITH_QUEENS if has_queens(board) else ENDGAME_MATERIAL
    if npm <= threshold:
        return "endgame"
    return "middlegame"
