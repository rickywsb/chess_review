"""Data models for chess game analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Evaluations are stored in centipawns from a fixed perspective and clamped to
# this magnitude so that forced-mate scores do not distort aggregate stats.
CP_CLAMP = 3000


@dataclass
class MoveAnalysis:
    """Engine analysis of a single played move."""

    ply: int                    # 1-based half-move index
    move_number: int            # full move number the move belongs to
    color: bool                 # chess.WHITE / chess.BLACK — side that moved
    san: str
    uci: str
    fen_before: str
    fen_after: str

    # All evals below are in centipawns.
    eval_before_mover: int      # best available eval before the move (mover POV)
    eval_after_mover: int       # eval after the played move (mover POV)
    eval_before_white: int      # same position, White POV (for trajectory)
    eval_after_white: int
    cp_loss: int                # max(0, eval_before_mover - eval_after_mover)

    best_move_uci: str
    best_move_san: str

    phase: str                  # 'opening' | 'middlegame' | 'endgame'
    classification: str         # 'best'|'good'|'inaccuracy'|'mistake'|'blunder'

    best_is_capture: bool
    best_is_check: bool         # best move gives check (forcing)
    played_is_capture: bool
    in_book: bool               # move is still within opening theory

    mate_before: Optional[int] = None   # signed mate distance, mover POV
    mate_after: Optional[int] = None
    best_line_san: list[str] = field(default_factory=list)  # first plies of the best line
    # Opponent's best line *after* the played move — i.e. how the mistake is
    # punished. SAN, starting from ``fen_after``. Empty when the move was fine.
    refutation_line_san: list[str] = field(default_factory=list)
    # MultiPV context (filled only for critiqued moves): how far the best move
    # led the second best (mover-POV centipawns), and how many near-equal
    # options existed within 30cp of the best. Lets explanations tell an
    # only-move position apart from one with several equally good choices.
    alt_gap_cp: int = 0
    alt_count: int = 1

    @property
    def is_blunder(self) -> bool:
        return self.classification == "blunder"

    @property
    def is_mistake(self) -> bool:
        return self.classification == "mistake"

    @property
    def is_forcing_miss(self) -> bool:
        """Best move was a capture or check but the player did not find it."""
        return (self.best_is_capture or self.best_is_check) and self.uci != self.best_move_uci


@dataclass
class GameAnalysis:
    """Full analysis of one game."""

    white: str
    black: str
    result: str                 # '1-0' | '0-1' | '1/2-1/2' | '*'
    date: str
    event: str
    site: str
    eco: str
    opening_name: str
    headers: dict = field(default_factory=dict)

    moves: list[MoveAnalysis] = field(default_factory=list)
    final_eval_white: int = 0

    deviation_ply: Optional[int] = None      # first ply that left opening theory
    deviation_side: Optional[bool] = None
    deviation_move_san: Optional[str] = None
    deviation_book_san: Optional[str] = None  # last known book continuation, if any

    # ---- convenience --------------------------------------------------------
    def player_color(self, name: str) -> Optional[bool]:
        """Return chess.WHITE/BLACK for `name`, or None if not in this game."""
        import chess

        low = name.strip().lower()
        if low and low in self.white.lower():
            return chess.WHITE
        if low and low in self.black.lower():
            return chess.BLACK
        return None

    def player_moves(self, color: bool) -> list[MoveAnalysis]:
        return [m for m in self.moves if m.color == color]

    def result_score(self, color: bool) -> Optional[float]:
        """Score for `color`: 1.0 win, 0.5 draw, 0.0 loss, None if unfinished."""
        import chess

        if self.result == "1-0":
            return 1.0 if color == chess.WHITE else 0.0
        if self.result == "0-1":
            return 1.0 if color == chess.BLACK else 0.0
        if self.result == "1/2-1/2":
            return 0.5
        return None

    def eval_trajectory_white(self) -> list[int]:
        """White-POV eval at every node (length = len(moves) + 1)."""
        traj = [m.eval_before_white for m in self.moves]
        traj.append(self.final_eval_white)
        return traj
