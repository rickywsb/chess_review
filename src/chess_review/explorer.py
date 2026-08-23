"""Optional lichess *masters* opening-explorer client.

Given a position (FEN) this queries the public opening explorer
(https://explorer.lichess.ovh/masters) to obtain:

* the most popular master moves (first choice / second choice, by game count),
* the opening ECO code and name for the position, and
* top master games (e.g. 2500+ GMs) that reached the position.

Network access is entirely optional: every call fails soft (returns ``None``)
so a report still builds when offline or when the endpoint is unreachable. The
data is used to enrich the opening breakdown of a game review.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

_MASTERS_URL = "https://explorer.lichess.ovh/masters"
_UA = "chess-review/0.1 (+https://github.com/rickywsb/chess_review)"


@dataclass
class ExplorerMove:
    san: str
    uci: str
    white: int
    draws: int
    black: int
    avg_rating: int

    @property
    def games(self) -> int:
        return self.white + self.draws + self.black

    def white_score_pct(self) -> float:
        t = self.games or 1
        return (self.white + 0.5 * self.draws) / t * 100.0


@dataclass
class ExplorerGame:
    white_name: str
    white_rating: int
    black_name: str
    black_rating: int
    winner: str                 # 'white' | 'black' | '' (draw)
    year: Optional[int]
    game_id: str


@dataclass
class ExplorerData:
    eco: str
    name: str
    total_games: int
    moves: list = field(default_factory=list)       # list[ExplorerMove]
    top_games: list = field(default_factory=list)   # list[ExplorerGame]


class OpeningExplorer:
    """Cached, fail-soft client for the lichess masters opening explorer."""

    def __init__(self, timeout: float = 6.0, enabled: bool = True) -> None:
        self.timeout = timeout
        self.enabled = enabled
        self._cache: dict[str, Optional[ExplorerData]] = {}
        self.available: Optional[bool] = None  # None until first successful/failed call
        self.last_error: Optional[str] = None

    def lookup(self, fen: str, moves: int = 4, top_games: int = 3) -> Optional[ExplorerData]:
        if not self.enabled:
            return None
        if fen in self._cache:
            return self._cache[fen]
        raw = self._fetch(fen, moves, top_games)
        parsed = self._parse(raw) if raw is not None else None
        self._cache[fen] = parsed
        return parsed

    # ---- internals ---------------------------------------------------------
    def _fetch(self, fen: str, moves: int, top_games: int) -> Optional[dict]:
        # Once the endpoint has refused us (e.g. HTTP 401/403 IP block, or a
        # network error) stop hammering it for the rest of the run.
        if self.available is False:
            return None
        params = urllib.parse.urlencode({"fen": fen, "moves": moves, "topGames": top_games})
        req = urllib.request.Request(f"{_MASTERS_URL}?{params}", headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                data = json.load(resp)
            self.available = True
            return data
        except urllib.error.HTTPError as e:
            self.available = False
            if e.code in (401, 403):
                self.last_error = (
                    f"HTTP {e.code}: lichess 拒绝了本机对开局库(explorer.lichess.ovh)的请求"
                    "（该 IP 被限流/封禁）。请更换网络或关闭 VPN 后重试。"
                )
            elif e.code == 429:
                self.last_error = "HTTP 429: 请求过于频繁，被 lichess 限流，请稍后再试。"
            else:
                self.last_error = f"HTTP {e.code}: {e.reason}"
            sys.stderr.write(f"[explorer] {self.last_error}\n")
            return None
        except Exception as e:  # noqa: BLE001
            self.available = False
            self.last_error = f"{type(e).__name__}: {e}"
            sys.stderr.write(f"[explorer] 无法访问大师开局库：{self.last_error}\n")
            return None

    @staticmethod
    def _parse(data: dict) -> Optional[ExplorerData]:
        if not isinstance(data, dict):
            return None
        moves: list[ExplorerMove] = []
        for m in data.get("moves", []) or []:
            moves.append(ExplorerMove(
                san=m.get("san", ""),
                uci=m.get("uci", ""),
                white=int(m.get("white", 0) or 0),
                draws=int(m.get("draws", 0) or 0),
                black=int(m.get("black", 0) or 0),
                avg_rating=int(m.get("averageRating", 0) or 0),
            ))
        games: list[ExplorerGame] = []
        for g in data.get("topGames", []) or []:
            wp = g.get("white", {}) or {}
            bp = g.get("black", {}) or {}
            games.append(ExplorerGame(
                white_name=wp.get("name", "?"),
                white_rating=int(wp.get("rating", 0) or 0),
                black_name=bp.get("name", "?"),
                black_rating=int(bp.get("rating", 0) or 0),
                winner=(g.get("winner") or ""),
                year=g.get("year"),
                game_id=g.get("id", ""),
            ))
        opening = data.get("opening") or {}
        total = int(data.get("white", 0) or 0) + int(data.get("draws", 0) or 0) \
            + int(data.get("black", 0) or 0)
        return ExplorerData(
            eco=opening.get("eco", "") if opening else "",
            name=opening.get("name", "") if opening else "",
            total_games=total,
            moves=moves,
            top_games=games,
        )
