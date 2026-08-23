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
import os
from typing import Optional

import chess.pgn
from flask import Flask, jsonify, request, send_from_directory

from .analysis import analyze_game
from .classify import MISTAKE
from .engine import Engine
from .metrics import build_player_report
from .opening_book import OpeningBook
from .render import (
    build_game_view,
    render_game_html,
    render_player_html,
)

_WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

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


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(_WEB_DIR, "index.html")

    @app.post("/api/analyze")
    def analyze():
        # PGN can come from an uploaded file or a pasted text field.
        pgn_text = ""
        if "file" in request.files and request.files["file"].filename:
            pgn_text = request.files["file"].read().decode("utf-8", errors="replace")
        else:
            pgn_text = request.form.get("pgn", "")

        mode = request.form.get("mode", "student")
        player = (request.form.get("player") or "").strip() or None
        depth = _clamp_int(request.form.get("depth"), 14, 6, 24)
        max_games = _clamp_int(request.form.get("max_games"), 20, 1, 200)
        use_explorer = request.form.get("explorer", "1") not in ("0", "false", "off", "")

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
            with Engine(depth=depth) as engine:
                if mode == "backtest":
                    subset = games[:max_games]
                    analyses = [analyze_game(g, engine, book=_book()) for g in subset]
                    report = build_player_report(analyses, player)
                    html = render_player_html(report)
                    title = f"{player} · 历史回测（{len(subset)} 局）"
                    return jsonify(ok=True, html=html, title=title,
                                   note=report.get("error"))

                # single-game modes analyze the first game in the file
                game = games[0]
                ga = analyze_game(game, engine, book=_book(), progress=False)
                dual = (mode == "student")
                explorer = None
                if use_explorer:
                    from .explorer import OpeningExplorer
                    explorer = OpeningExplorer()
                view = build_game_view(ga, player=player, threshold=MISTAKE,
                                       with_svg=True, dual=dual, explorer=explorer)
                html = render_game_html(view)
                w, b = ga.white, ga.black
                mode_zh = "双方视角" if dual else f"聚焦 {player or '未指定'}"
                title = f"{w} vs {b} · {mode_zh}"
                return jsonify(ok=True, html=html, title=title)
        except FileNotFoundError as exc:
            return jsonify(ok=False, error=f"未找到 Stockfish 引擎：{exc}。"
                           "请先运行 `chess-review setup`。"), 500
        except Exception as exc:  # noqa: BLE001 - surface any engine/parse error to UI
            return jsonify(ok=False, error=f"分析失败：{exc}"), 500

    return app


def run_web(host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    app = create_app()
    print(f"chess-review web 已启动： http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
