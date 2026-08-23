"""Optional lichess cloud-eval source.

lichess exposes deep, pre-computed evaluations for *popular* positions via
https://lichess.org/api/cloud-eval . There is no public API to analyze an
arbitrary position on demand, so coverage is essentially limited to openings
and well-known positions. This wrapper uses a cloud eval when available and
falls back to the local engine otherwise.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

import chess

from .engine import MATE_CP, Engine, ScoreInfo, _clamp

_CLOUD_URL = "https://lichess.org/api/cloud-eval"


class CloudEngine:
    """Engine-compatible wrapper: lichess cloud eval first, local fallback."""

    def __init__(self, local: Engine, timeout: float = 5.0) -> None:
        self.local = local
        self.timeout = timeout
        self.movetime = local.movetime
        self.depth = local.depth
        self.threads = local.threads
        self._cache: dict[str, ScoreInfo | None] = {}
        self.hits = 0
        self.misses = 0

    def describe(self) -> str:
        return f"lichess cloud-eval + local fallback ({self.local.describe()})"

    def analyse(self, board: chess.Board) -> ScoreInfo:
        info = self._cloud(board)
        if info is not None:
            self.hits += 1
            return info
        self.misses += 1
        return self.local.analyse(board)

    def _cloud(self, board: chess.Board) -> ScoreInfo | None:
        fen = board.fen()
        if fen in self._cache:
            return self._cache[fen]
        result = None
        try:
            url = f"{_CLOUD_URL}?{urllib.parse.urlencode({'fen': fen, 'multiPv': 1})}"
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:  # noqa: S310
                data = json.load(resp)
            result = self._parse(board, data)
        except Exception:
            result = None
        self._cache[fen] = result
        return result

    @staticmethod
    def _parse(board: chess.Board, data: dict) -> ScoreInfo | None:
        pv = data.get("pvs")
        if not pv:
            return None
        pv = pv[0]
        moves = (pv.get("moves") or "").split()
        best_move = None
        move_list = []
        for i, u in enumerate(moves[:8]):
            try:
                move_list.append(chess.Move.from_uci(u))
            except ValueError:
                break
        if move_list:
            best_move = move_list[0]

        mate = pv.get("mate")
        if mate is not None:
            cp_mover = _clamp((MATE_CP - abs(int(mate))) * (1 if mate > 0 else -1))
            mate_mover = int(mate)
        elif "cp" in pv:
            cp_mover = _clamp(int(pv["cp"]))
            mate_mover = None
        else:
            return None

        cp_white = cp_mover if board.turn == chess.WHITE else -cp_mover
        return ScoreInfo(
            cp_white=cp_white,
            cp_mover=cp_mover,
            best_move=best_move,
            mate_mover=mate_mover,
            pv=move_list,
        )

    def close(self) -> None:
        self.local.close()

    def __enter__(self) -> "CloudEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
