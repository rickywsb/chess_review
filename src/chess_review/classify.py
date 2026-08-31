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


# ---------------------------------------------------------------------------
# eval-drop classification (the "why did the number fall" layer)
# ---------------------------------------------------------------------------
# Once a move is flagged, we still need to say *what kind* of mistake it was and
# frame its magnitude honestly. The single most common inaccuracy is telling a
# student "局面明显变差" when they are in fact still winning — so the resulting
# STATE (from the eval after the move) drives the framing, not the size of the
# drop. Categories are decided from the numbers + the material swing that the
# engine's refutation line actually produces (see render._move_verdict).
_STATE_WORD = {
    2: "仍是胜势", 1: "仍占优势", 0: "已回到均势",
    -1: "已处于下风", -2: "已落入败势",
}

CATEGORY_ZH = {
    "threw_mate": "错过强制杀",
    "lost_material": "丢失子力",
    "lost_the_win": "把优势走没了",
    "positional_slip": "细微的位置性选择",
    "big_error": "严重失误",
}

# Honest framing instructions handed to the LLM so it does not over/under-state.
CATEGORY_FRAME = {
    "threw_mate": "重点是错过了直接的强制杀；按 resulting_state，你依然占优，"
                  "绝不能说局面明显变差，语气应是『可惜，本可一击制胜』。",
    "lost_material": "核心问题是丢失了子力；请依据『对手最强回应』说明具体丢了什么。",
    "lost_the_win": "核心问题是把已经到手的优势/胜势走没了，回到了均势甚至更糟。",
    "positional_slip": "这是一步细微的位置性选择，没有立刻的吃子或杀着，评估只是小幅下滑；"
                       "请坦诚说明这一点，不要硬编具体棋理或战术。",
    "big_error": "请顺着『对手最强回应』把后果讲清楚，不要空泛。",
}


def state_word(after_cp: int, mate_after: "int | None") -> str:
    """Human word for the resulting position, honest about magnitude."""
    return _STATE_WORD[outcome_zone(after_cp, mate_after)]


def classify_delta(before_cp: int, after_cp: int,
                   mate_before: "int | None", mate_after: "int | None",
                   material_swing: int, cp_loss: int) -> "tuple[str, str]":
    """Classify *why* the eval dropped and describe the resulting state.

    Returns ``(category, resulting_state_word)``. ``material_swing`` is the net
    material change for the mover along the engine's refutation line (negative
    => the mover loses material). Rules run in priority order."""
    state = state_word(after_cp, mate_after)
    zb = outcome_zone(before_cp, mate_before)
    za = outcome_zone(after_cp, mate_after)
    threw_mate = (mate_before is not None and mate_before > 0
                  and not (mate_after is not None and mate_after > 0))

    # Had a forced mate, gave it up, but still winning: don't scare the student.
    if threw_mate and za >= 1:
        return "threw_mate", state
    # The refutation line actually wins material off the mover.
    if material_swing <= -1:
        return "lost_material", state
    # Had a real edge, now no better than equal: the win slipped away.
    if zb >= 1 and za <= 0:
        return "lost_the_win", state
    # Threw a mate and is no longer winning -> it cost the game, not just a mate.
    if threw_mate:
        return "lost_the_win", state
    # Small drop, no material change, no mate: a quiet positional slip.
    if cp_loss < 150 and material_swing == 0:
        return "positional_slip", state
    return "big_error", state

