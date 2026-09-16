"""Local web frontend for chess-review.

A small Flask app that lets a coach drag-and-drop a PGN and get a report:

  1. 学员对局分析（双方视角）  -> mode="student"  (dual)
  2. 某位棋手对局（一方视角）  -> mode="player"   (single focus)
  3. 某位棋手历史数据回测      -> mode="backtest" (player report across games)

Analysis runs synchronously with a local Stockfish engine, which is fine for
a single coach on one machine. Reports are returned as full HTML documents and
shown in an iframe on the page.
"""
from __future__ import annotations

import io
import hmac
import os
import threading
import time
from collections import defaultdict, deque
from typing import Optional

import chess.pgn
from flask import Flask, Response, jsonify, request, send_from_directory

from .analysis import analyze_game
from .classify import MISTAKE
from .engine import Engine
from .history import DEFAULT_HISTORY_DB, PlayerHistoryStore
from .metrics import build_player_report
from .master_db import MasterOpeningDatabase
from .opening_book import OpeningBook
from .render import (
    build_game_view,
    render_game_html,
    render_player_html,
)

_WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
_HISTORY_DB = os.environ.get("CHESS_REVIEW_HISTORY_DB", DEFAULT_HISTORY_DB)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return default


# Server-side search budget. A deployment can raise these for a beefier machine
# via CHESS_REVIEW_DEPTH / CHESS_REVIEW_MAX_DEPTH without touching the UI.
_DEFAULT_DEPTH = _env_int("CHESS_REVIEW_DEPTH", 18)
_MAX_DEPTH = max(_DEFAULT_DEPTH, _env_int("CHESS_REVIEW_MAX_DEPTH", 26))

# Load the opening book once per process (shared, read-only).
_BOOK: Optional[OpeningBook] = None


def _book() -> OpeningBook:
    global _BOOK
    if _BOOK is None:
        _BOOK = OpeningBook.load()
    return _BOOK


def _read_pgn_text(text: str) -> list[chess.pgn.Game]:
    games: list[chess.pgn.Game] = []
    stream = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        games.append(game)
    return games


def _clamp_int(value, default: int, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


# --- abuse / resource guards for the public endpoint ----------------------
# The analyze endpoint is expensive (Stockfish + a paid LLM call), so a public
# deployment needs basic protection. All limits are env-tunable; set
# CHESS_REVIEW_RATE_MAX=0 to disable rate limiting (e.g. for local use).
_RATE_MAX = _env_int("CHESS_REVIEW_RATE_MAX", 20)        # requests / window / IP
_RATE_WINDOW = _env_int("CHESS_REVIEW_RATE_WINDOW", 60)  # seconds
_MAX_PGN_BYTES = _env_int("CHESS_REVIEW_MAX_PGN_BYTES", 1_000_000)  # ~1 MB
_MAX_GAMES = _env_int("CHESS_REVIEW_MAX_GAMES", 200)


class _RateLimiter:
    """Fixed-window per-client limiter. In-process (per gunicorn worker), which
    is enough to blunt abusive bursts against the paid analyze endpoint without
    adding an external store. ``max_hits <= 0`` disables it."""

    def __init__(self, max_hits: int, window_s: int) -> None:
        self.max_hits = max_hits
        self.window_s = window_s
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)`` for ``key``."""
        if self.max_hits <= 0:
            return True, 0
        now = time.monotonic()
        cutoff = now - self.window_s
        with self._lock:
            dq = self._hits[key]
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.max_hits:
                return False, max(1, int(dq[0] + self.window_s - now) + 1)
            dq.append(now)
            if len(self._hits) > 4096:  # bound memory: drop drained buckets
                for k in [k for k, v in self._hits.items() if not v]:
                    self._hits.pop(k, None)
            return True, 0


_LIMITER = _RateLimiter(_RATE_MAX, _RATE_WINDOW)


def _client_key() -> str:
    """Best-effort client identity. Behind the fly.io proxy the real IP is in
    ``Fly-Client-IP``; otherwise use the first ``X-Forwarded-For`` hop, then the
    socket address."""
    ip = request.headers.get("Fly-Client-IP")
    if not ip:
        xff = request.headers.get("X-Forwarded-For", "")
        ip = xff.split(",")[0].strip() if xff else ""
    return ip or (request.remote_addr or "unknown")


def _coach_authorized() -> bool:
    token = os.environ.get("CHESS_REVIEW_COACH_TOKEN")
    if token:
        auth = request.authorization
        return bool(
            auth
            and hmac.compare_digest(auth.username or "", "coach")
            and hmac.compare_digest(auth.password or "", token)
        )
    forwarded = request.headers.get("Fly-Client-IP") or request.headers.get(
        "X-Forwarded-For")
    return not forwarded and request.remote_addr in ("127.0.0.1", "::1")


def _coach_unauthorized() -> Response:
    if not os.environ.get("CHESS_REVIEW_COACH_TOKEN"):
        return Response(
            "Coach access is not configured.", 503,
            {"Retry-After": "300"},
        )
    return Response(
        "Coach access requires authentication.", 401,
        {"WWW-Authenticate": 'Basic realm="Chess Review Coach"'},
    )


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(_WEB_DIR, "index.html")

    @app.get("/coach")
    def coach():
        if not _coach_authorized():
            return _coach_unauthorized()
        return send_from_directory(_WEB_DIR, "coach.html")

    @app.get("/api/coach/students")
    def coach_students():
        if not _coach_authorized():
            return _coach_unauthorized()
        with PlayerHistoryStore(_HISTORY_DB) as store:
            students = store.coach_students()
        return jsonify(ok=True, students=students)

    @app.get("/api/coach/students/<person_id>")
    def coach_student(person_id: str):
        if not _coach_authorized():
            return _coach_unauthorized()
        try:
            with PlayerHistoryStore(_HISTORY_DB) as store:
                identity = store.person_identity(person_id)
                profile_id = store.latest_person_profile_id(person_id)
                analyses = (store.load_person_analyses(person_id, profile_id)
                            if profile_id else [])
                summary = next(
                    item for item in store.coach_students()
                    if item["person_id"] == person_id)
        except (ValueError, StopIteration):
            return jsonify(ok=False, error="未找到该学员档案。"), 404
        report = build_player_report(
            analyses, identity["display_name"], aliases=identity["aliases"])
        return jsonify(ok=True, student=summary, report=report)

    @app.post("/api/analyze")
    def analyze():
        allowed, retry = _LIMITER.allow(_client_key())
        if not allowed:
            resp = jsonify(ok=False,
                           error=f"请求过于频繁，请约 {retry} 秒后再试。")
            resp.status_code = 429
            resp.headers["Retry-After"] = str(retry)
            return resp

        # PGN can come from an uploaded file or a pasted text field.
        pgn_text = ""
        if "file" in request.files and request.files["file"].filename:
            pgn_text = request.files["file"].read().decode("utf-8", errors="replace")
        else:
            pgn_text = request.form.get("pgn", "")

        if len(pgn_text) > _MAX_PGN_BYTES:
            return jsonify(ok=False, error="PGN 内容过大，请精简后再试。"), 413

        mode = request.form.get("mode", "student")
        player = (request.form.get("player") or "").strip() or None
        depth = _clamp_int(request.form.get("depth"), _DEFAULT_DEPTH, 6, _MAX_DEPTH)
        max_games = _clamp_int(request.form.get("max_games"), min(20, _MAX_GAMES), 1, _MAX_GAMES)

        if not pgn_text.strip():
            return jsonify(ok=False, error="没有收到 PGN，请拖入或粘贴对局。"), 400

        games = _read_pgn_text(pgn_text)
        if not games:
            return jsonify(ok=False, error="无法从内容中解析出对局，请确认是有效的 PGN。"), 400

        if mode in ("student", "player") and not player and mode == "player":
            return jsonify(ok=False, error="单方视角需要填写棋手名字。"), 400
        if mode == "backtest" and not player:
            return jsonify(ok=False, error="历史回测需要填写棋手名字。"), 400

        try:
            master_path = os.environ.get("CHESS_REVIEW_MASTER_DB")
            master_db = MasterOpeningDatabase(master_path) if master_path else None
            with Engine(depth=depth) as engine:
                if mode == "backtest":
                    subset = games[:max_games]
                    analyses = [analyze_game(g, engine, book=_book(), master_db=master_db)
                                for g in subset]
                    report = build_player_report(analyses, player)
                    html = render_player_html(report)
                    title = f"{player} · 历史回测（{len(subset)} 局）"
                    return jsonify(ok=True, html=html, title=title,
                                   note=report.get("error"))

                # single-game modes analyze the first game in the file
                game = games[0]
                ga = analyze_game(game, engine, book=_book(), progress=False,
                                  master_db=master_db)
                dual = (mode == "student")
                view = build_game_view(ga, player=player, threshold=MISTAKE,
                                       with_svg=True, dual=dual)
                html = render_game_html(view)
                w, b = ga.white, ga.black
                mode_zh = "双方视角" if dual else f"聚焦 {player or '未指定'}"
                title = f"{w} vs {b} · {mode_zh}"
                return jsonify(ok=True, html=html, title=title)
        except FileNotFoundError:
            app.logger.exception("Stockfish engine is unavailable")
            return jsonify(ok=False, error="未找到 Stockfish 引擎，请先运行 "
                           "`chess-review setup`。"), 500
        except Exception:  # noqa: BLE001 - isolate engine and rendering failures
            app.logger.exception("Chess analysis failed")
            return jsonify(ok=False, error="分析失败，请稍后重试。"), 500
        finally:
            if "master_db" in locals() and master_db is not None:
                master_db.close()

    return app


def run_web(host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    app = create_app()
    print(f"chess-review web 已启动： http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
