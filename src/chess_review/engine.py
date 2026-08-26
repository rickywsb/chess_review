"""Stockfish engine wrapper."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine

from .models import CP_CLAMP

# Large sentinel used when converting mate scores to centipawns before clamping.
MATE_CP = 100_000


@dataclass
class ScoreInfo:
    """Normalized engine output for one position."""

    cp_white: int          # clamped centipawns, White POV
    cp_mover: int          # clamped centipawns, side-to-move POV
    best_move: Optional[chess.Move]
    mate_mover: Optional[int]  # signed mate distance from side-to-move POV
    pv: list = None            # principal variation (list[chess.Move])


def _clamp(value: int) -> int:
    return max(-CP_CLAMP, min(CP_CLAMP, value))


def default_threads() -> int:
    """Leave one core free for the OS; use at least one thread."""
    cpu = os.cpu_count() or 1
    return max(1, cpu - 1)


def _env_int(name: str, default: int) -> int:
    """Read a non-negative int from the environment, falling back to default."""
    value = os.environ.get(name)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return default


def resolve_engine_path(explicit: Optional[str] = None) -> str:
    """Find a Stockfish binary.

    Order: explicit arg -> CHESS_ENGINE_PATH env -> bundled data/engines ->
    `stockfish` on PATH.
    """
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise FileNotFoundError(f"Engine not found at: {explicit}")

    env = os.environ.get("CHESS_ENGINE_PATH")
    if env and os.path.isfile(env):
        return env

    bundled_dir = os.path.join(os.path.dirname(__file__), "data", "engines")
    if os.path.isdir(bundled_dir):
        for name in sorted(os.listdir(bundled_dir)):
            candidate = os.path.join(bundled_dir, name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

    found = shutil.which("stockfish")
    if found:
        return found

    raise FileNotFoundError(
        "No Stockfish engine found. Install it (e.g. `brew install stockfish`) "
        "or run `chess-review setup`, or pass --engine /path/to/stockfish."
    )


class Engine:
    """Thin wrapper around a UCI engine that returns normalized scores."""

    def __init__(
        self,
        path: Optional[str] = None,
        depth: int = 18,
        threads: Optional[int] = None,
        hash_mb: Optional[int] = None,
        movetime: Optional[int] = None,
    ) -> None:
        self.path = resolve_engine_path(path)
        self.depth = depth
        # movetime (milliseconds per position) takes precedence over depth.
        self.movetime = movetime
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        # Infra tuning falls back to env vars so a deployment can size the
        # engine to its machine without code changes:
        #   CHESS_ENGINE_THREADS, CHESS_ENGINE_HASH_MB
        if threads is None:
            threads = _env_int("CHESS_ENGINE_THREADS", 0)
        if hash_mb is None:
            hash_mb = _env_int("CHESS_ENGINE_HASH_MB", 512)
        options = {}
        if "Threads" in self._engine.options:
            options["Threads"] = threads if threads and threads > 0 else default_threads()
        if "Hash" in self._engine.options:
            options["Hash"] = hash_mb
        if options:
            self._engine.configure(options)
        self.threads = options.get("Threads", 1)

    def describe(self) -> str:
        budget = f"{self.movetime}ms/move" if self.movetime else f"depth {self.depth}"
        return f"{budget}, {self.threads} threads"

    def _limit(self) -> chess.engine.Limit:
        if self.movetime:
            return chess.engine.Limit(time=self.movetime / 1000.0)
        return chess.engine.Limit(depth=self.depth)

    def analyse(self, board: chess.Board) -> ScoreInfo:
        info = self._engine.analyse(board, self._limit())
        pov = info["score"]
        white_score = pov.white()
        mover_score = pov.pov(board.turn)

        cp_white = _clamp(white_score.score(mate_score=MATE_CP))
        cp_mover = _clamp(mover_score.score(mate_score=MATE_CP))

        pv = info.get("pv") or []
        best_move = pv[0] if pv else None

        return ScoreInfo(
            cp_white=cp_white,
            cp_mover=cp_mover,
            best_move=best_move,
            mate_mover=mover_score.mate(),
            pv=list(pv[:8]),
        )

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
