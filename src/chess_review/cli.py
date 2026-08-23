"""Command-line interface for chess-review."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterator

import chess.pgn

from .analysis import analyze_game
from .classify import MISTAKE
from .cloud import CloudEngine
from .engine import Engine
from .metrics import build_player_report
from .opening_book import OpeningBook
from .render import (
    build_game_view,
    render_game_html,
    render_game_markdown,
    render_player_html,
    render_player_markdown,
)


def _slug(text: str) -> str:
    keep = [c if c.isalnum() else "-" for c in text.strip()]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "game"


def _iter_games(paths: list[str]) -> Iterator[chess.pgn.Game]:
    for path in paths:
        if not os.path.exists(path):
            print(f"warning: file not found: {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                yield game


def _write(out_dir: str, base: str, fmt: list[str], md: str, html: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    if "md" in fmt:
        p = os.path.join(out_dir, base + ".md")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(md)
        written.append(p)
    if "html" in fmt:
        p = os.path.join(out_dir, base + ".html")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(html)
        written.append(p)
    return written


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_setup(args: argparse.Namespace) -> int:
    from .setup_engine import run_setup

    run_setup(explicit_engine=args.engine, skip_book=args.skip_book, force_book=args.force_book)
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    fmt = [f.strip() for f in args.format.split(",") if f.strip()]
    book = OpeningBook.load()
    if not book.loaded:
        print("note: opening book not found — run `chess-review setup` to enable "
              "theory-deviation detection.", file=sys.stderr)

    games = list(_iter_games([args.pgn]))
    if not games:
        print("No games found.", file=sys.stderr)
        return 1
    if args.game_index != "all":
        try:
            idx = int(args.game_index)
        except ValueError:
            print("--game-index must be an integer or 'all'.", file=sys.stderr)
            return 2
        if idx < 0 or idx >= len(games):
            print(f"--game-index out of range (0..{len(games) - 1}).", file=sys.stderr)
            return 2
        games = [games[idx]]

    written: list[str] = []
    with Engine(path=args.engine, depth=args.depth, threads=args.threads,
                movetime=args.movetime) as local:
        engine = CloudEngine(local) if args.cloud else local
        for i, game in enumerate(games):
            w = game.headers.get("White", "White")
            b = game.headers.get("Black", "Black")
            print(f"Analyzing: {w} vs {b} ({engine.describe()}) ...")
            ga = analyze_game(game, engine, book=book, progress=True)
            view = build_game_view(ga, player=args.player, threshold=args.threshold,
                                   with_svg=("html" in fmt), dual=args.both)
            md = render_game_markdown(view)
            html = render_game_html(view)
            base = _slug(f"{w}-vs-{b}-{ga.date}") or f"game-{i}"
            written += _write(args.out, base, fmt, md, html)

    for p in written:
        print(f"wrote {p}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    fmt = [f.strip() for f in args.format.split(",") if f.strip()]
    book = OpeningBook.load()

    games = list(_iter_games(args.pgn))
    if args.limit:
        games = games[: args.limit]
    if not games:
        print("No games found.", file=sys.stderr)
        return 1

    print(f"Analyzing {len(games)} games for '{args.player}' ...")
    analyses = []
    with Engine(path=args.engine, depth=args.depth, threads=args.threads,
                movetime=args.movetime) as local:
        engine = CloudEngine(local) if args.cloud else local
        print(f"  engine: {engine.describe()}")
        for i, game in enumerate(games, 1):
            print(f"\r  game {i}/{len(games)}", end="", flush=True)
            analyses.append(analyze_game(game, engine, book=book))
    print()

    report = build_player_report(analyses, args.player)
    md = render_player_markdown(report)
    html = render_player_html(report)
    base = _slug(args.player) + "-report"
    written = _write(args.out, base, fmt, md, html)
    for p in written:
        print(f"wrote {p}")
    if report.get("error"):
        print(report["error"], file=sys.stderr)
        return 1
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    from .webapp import run_web

    run_web(host=args.host, port=args.port, debug=args.debug)
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chess-review",
        description="Coach-facing chess game review: single-game reviews and player reports.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_engine_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--engine", help="Path to Stockfish (defaults to auto-detect).")
        sp.add_argument("--depth", type=int, default=18, help="Engine search depth (default 18).")
        sp.add_argument("--movetime", type=int, default=None,
                        help="Milliseconds per position (overrides --depth; e.g. 800).")
        sp.add_argument("--threads", type=int, default=None,
                        help="Engine threads (default: all cores minus one).")
        sp.add_argument("--cloud", action="store_true",
                        help="Use lichess cloud eval when available (mainly openings), local fallback.")
        sp.add_argument("--out", default="reports", help="Output directory (default 'reports').")
        sp.add_argument("--format", default="md,html", help="Comma list: md,html (default both).")

    sp_setup = sub.add_parser("setup", help="Locate Stockfish and download the opening book.")
    sp_setup.add_argument("--engine", help="Path to an existing Stockfish binary.")
    sp_setup.add_argument("--skip-book", action="store_true", help="Do not download the opening book.")
    sp_setup.add_argument("--force-book", action="store_true", help="Re-download the opening book.")
    sp_setup.set_defaults(func=cmd_setup)

    sp_rev = sub.add_parser("review", help="Review a single PGN (one or more games).")
    sp_rev.add_argument("pgn", help="Path to a PGN file.")
    sp_rev.add_argument("--player", help="Focus critical moments on this player's moves.")
    sp_rev.add_argument("--both", action="store_true",
                        help="Dual perspective: full report for BOTH sides (student review).")
    sp_rev.add_argument("--game-index", default="0",
                        help="Which game in the file: an index or 'all' (default 0).")
    sp_rev.add_argument("--threshold", type=int, default=MISTAKE,
                        help="Min centipawn loss to list as a critical moment (default 100).")
    add_engine_opts(sp_rev)
    sp_rev.set_defaults(func=cmd_review)

    sp_rep = sub.add_parser("report", help="Build a player tracking report across many games.")
    sp_rep.add_argument("pgn", nargs="+", help="One or more PGN files.")
    sp_rep.add_argument("--player", required=True, help="Target player name (matches PGN headers).")
    sp_rep.add_argument("--limit", type=int, help="Only analyze the first N games (for a quick pass).")
    add_engine_opts(sp_rep)
    sp_rep.set_defaults(func=cmd_report)

    sp_web = sub.add_parser("web", help="Launch the drag-and-drop web frontend.")
    sp_web.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    sp_web.add_argument("--port", type=int, default=8000, help="Port (default 8000).")
    sp_web.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    sp_web.set_defaults(func=cmd_web)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
