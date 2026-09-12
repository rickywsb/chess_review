"""Command-line interface for chess-review."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
import sys
from typing import Iterator

import chess.pgn

from . import coach_llm
from .analysis import analyze_game
from .classify import MISTAKE, significance
from .cloud import CloudEngine
from .dataset import (
    build_provenance,
    build_record,
    game_id_for,
    split_for_game,
    write_jsonl,
)
from .engine import Engine
from .metrics import build_player_report
from .master_db import MasterOpeningDatabase, build_master_database
from .opening_book import OpeningBook
from .render import (
    build_explanation_context,
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


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


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
    with ExitStack() as stack:
        local = stack.enter_context(Engine(
            path=args.engine, depth=args.depth, threads=args.threads,
            movetime=args.movetime))
        master_db = (stack.enter_context(MasterOpeningDatabase(args.master_db))
                     if args.master_db else None)
        engine = CloudEngine(local) if args.cloud else local
        for i, game in enumerate(games):
            w = game.headers.get("White", "White")
            b = game.headers.get("Black", "Black")
            print(f"Analyzing: {w} vs {b} ({engine.describe()}) ...")
            ga = analyze_game(game, engine, book=book, progress=True,
                              master_db=master_db)
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
    with ExitStack() as stack:
        local = stack.enter_context(Engine(
            path=args.engine, depth=args.depth, threads=args.threads,
            movetime=args.movetime))
        master_db = (stack.enter_context(MasterOpeningDatabase(args.master_db))
                     if args.master_db else None)
        engine = CloudEngine(local) if args.cloud else local
        print(f"  engine: {engine.describe()}")
        for i, game in enumerate(games, 1):
            print(f"\r  game {i}/{len(games)}", end="", flush=True)
            analyses.append(analyze_game(game, engine, book=book,
                                         master_db=master_db))
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


def cmd_dataset(args: argparse.Namespace) -> int:
    """Export verified, versioned training examples as canonical JSONL."""
    if not coach_llm.available():
        print("Training export requires a configured LLM teacher.", file=sys.stderr)
        return 2
    games = list(_iter_games(args.pgn))
    if args.limit:
        games = games[:args.limit]
    if not games:
        print("No games found.", file=sys.stderr)
        return 1

    book = OpeningBook.load()
    records_by_id = {}
    stats = {
        "games": len(games),
        "games_without_player": 0,
        "candidate_moves": 0,
        "teacher_calls": 0,
        "accepted": 0,
        "rejected": 0,
        "duplicates": 0,
    }
    with Engine(path=args.engine, depth=args.depth, threads=args.threads,
                hash_mb=args.hash_mb) as engine:
        provenance = build_provenance(engine.metadata(), book)
        for index, game in enumerate(games, 1):
            if stats["teacher_calls"] >= args.max_samples:
                break
            print(f"\r  dataset game {index}/{len(games)}", end="", flush=True)
            analysis = analyze_game(game, engine, book=book)
            color = analysis.player_color(args.player) if args.player else None
            if args.player and color is None:
                stats["games_without_player"] += 1
                continue
            game_id = game_id_for(game)
            split = split_for_game(game_id)
            for move in analysis.moves:
                if color is not None and move.color != color:
                    continue
                keep, tag = significance(move, args.threshold)
                if not keep:
                    continue
                stats["candidate_moves"] += 1
                if stats["teacher_calls"] >= args.max_samples:
                    break
                stats["teacher_calls"] += 1
                context = build_explanation_context(move, tag)
                generation = coach_llm.generate_training_explanation(context)
                if generation is None:
                    stats["rejected"] += 1
                    continue
                try:
                    record = build_record(
                        game_id=game_id,
                        split=split,
                        ply=move.ply,
                        context=context,
                        generation=generation,
                        provenance=provenance,
                        source={
                            "fen_before": move.fen_before,
                            "fen_after": move.fen_after,
                            "played_uci": move.uci,
                            "best_move_uci": move.best_move_uci,
                            "best_line_san": move.best_line_san,
                            "refutation_line_san": move.refutation_line_san,
                        },
                    )
                except ValueError:
                    stats["rejected"] += 1
                    continue
                if record["sample_id"] in records_by_id:
                    stats["duplicates"] += 1
                else:
                    records_by_id[record["sample_id"]] = record
    print()
    records = list(records_by_id.values())
    stats["accepted"] = len(records)
    manifest = write_jsonl(args.out, records, provenance=provenance, stats=stats)
    print(f"wrote {args.out} ({len(records)} records)")
    print(f"wrote {manifest}")
    if not records:
        print("No verified training samples were generated.", file=sys.stderr)
        return 1
    return 0


def cmd_master_db_build(args: argparse.Namespace) -> int:
    try:
        stats = build_master_database(
            args.pgn, args.out, max_ply=args.max_ply,
            min_rating=args.min_rating, force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"master-db build failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}")
    print("indexed {games_indexed}/{games_seen} games, {positions_indexed} positions; "
          "duplicates {duplicate_games}, rating-filtered {skipped_rating}, "
          "invalid {skipped_invalid}".format(**stats.as_dict()))
    return 0


def cmd_master_db_query(args: argparse.Namespace) -> int:
    try:
        with MasterOpeningDatabase(args.db) as database:
            result = database.lookup(args.fen, top=args.top)
    except (FileNotFoundError, ValueError) as exc:
        print(f"master-db query failed: {exc}", file=sys.stderr)
        return 2
    if result is None:
        print("No master games found for this position.")
        return 1
    print(f"position {result.position_key}: {result.total_games} games")
    for move in result.moves:
        rating = f", avg rating {move.avg_rating}" if move.avg_rating else ""
        print(f"  {move.san} ({move.uci}): {move.games} games, "
              f"{move.play_rate:.1f}%, White score {move.white_score_pct:.1f}%{rating}")
    return 0


def cmd_master_db_info(args: argparse.Namespace) -> int:
    try:
        with MasterOpeningDatabase(args.db) as database:
            info = {"identity": database.identity(), "metadata": database.metadata()}
    except (FileNotFoundError, ValueError) as exc:
        print(f"master-db info failed: {exc}", file=sys.stderr)
        return 2
    import json

    print(json.dumps(info, ensure_ascii=False, sort_keys=True, indent=2))
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
        sp.add_argument("--master-db", default=os.environ.get("CHESS_REVIEW_MASTER_DB"),
                help="Local SQLite master opening database.")
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

    sp_data = sub.add_parser(
        "dataset", help="Export verified training examples as versioned JSONL.")
    sp_data.add_argument("pgn", nargs="+", help="One or more PGN files.")
    sp_data.add_argument("--out", default="datasets/chess-review.jsonl",
                         help="JSONL output path (default datasets/chess-review.jsonl).")
    sp_data.add_argument("--player", help="Only export this player's critical moves.")
    sp_data.add_argument("--limit", type=_positive_int,
                         help="Analyze only the first N games.")
    sp_data.add_argument("--max-samples", type=_positive_int, default=100,
                         help="Maximum paid teacher calls (default 100).")
    sp_data.add_argument("--threshold", type=int, default=MISTAKE,
                         help="Minimum centipawn loss considered (default 100).")
    sp_data.add_argument("--engine", help="Path to Stockfish (defaults to auto-detect).")
    sp_data.add_argument("--depth", type=_positive_int, default=18,
                         help="Fixed Stockfish depth (default 18).")
    sp_data.add_argument("--threads", type=_positive_int, default=1,
                         help="Stockfish threads; keep 1 for reproducibility (default 1).")
    sp_data.add_argument("--hash-mb", type=_positive_int, default=1024,
                         help="Fixed Stockfish hash size in MB (default 1024).")
    sp_data.set_defaults(func=cmd_dataset)

    sp_master = sub.add_parser(
        "master-db", help="Build and inspect a local master opening database.")
    master_sub = sp_master.add_subparsers(dest="master_command", required=True)

    sp_master_build = master_sub.add_parser(
        "build", help="Build SQLite opening statistics from PGN or ZIP archives.")
    sp_master_build.add_argument("pgn", nargs="+", help="PGN, ZIP, GZ, or BZ2 sources.")
    sp_master_build.add_argument("--out", default="data/master-openings.sqlite",
                                 help="Output SQLite path.")
    sp_master_build.add_argument("--max-ply", type=_positive_int, default=40,
                                 help="Maximum plies indexed per game (default 40).")
    sp_master_build.add_argument("--min-rating", type=int, default=0,
                                 help="Minimum rating required for both players.")
    sp_master_build.add_argument("--force", action="store_true",
                                 help="Atomically replace an existing database.")
    sp_master_build.set_defaults(func=cmd_master_db_build)

    sp_master_query = master_sub.add_parser(
        "query", help="Query master move statistics for a FEN.")
    sp_master_query.add_argument("--db", default="data/master-openings.sqlite")
    sp_master_query.add_argument("--fen", required=True)
    sp_master_query.add_argument("--top", type=_positive_int, default=5)
    sp_master_query.set_defaults(func=cmd_master_db_query)

    sp_master_info = master_sub.add_parser(
        "info", help="Show database version, filters, sources, and content hash.")
    sp_master_info.add_argument("--db", default="data/master-openings.sqlite")
    sp_master_info.set_defaults(func=cmd_master_db_info)

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
