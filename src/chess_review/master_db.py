"""Local SQLite opening statistics built from strong-player PGN archives."""
from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import os
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional, TextIO

import chess
import chess.pgn
import chess.polyglot

SCHEMA_VERSION = "1"
DEFAULT_MAX_PLY = 40


@dataclass(frozen=True)
class MasterMoveStats:
    uci: str
    san: str
    games: int
    white_wins: int
    draws: int
    black_wins: int
    avg_rating: Optional[int]
    play_rate: float
    first_year: Optional[int]
    last_year: Optional[int]

    @property
    def white_score_pct(self) -> float:
        return round((self.white_wins + 0.5 * self.draws) / self.games * 100, 1)


@dataclass(frozen=True)
class MasterPosition:
    position_key: str
    total_games: int
    moves: list[MasterMoveStats]
    source: str
    database_version: str


@dataclass
class BuildStats:
    games_seen: int = 0
    games_indexed: int = 0
    duplicate_games: int = 0
    skipped_invalid: int = 0
    skipped_rating: int = 0
    positions_indexed: int = 0

    def as_dict(self) -> dict:
        return dict(vars(self))


def position_key(board: chess.Board) -> str:
    """Return a transposition-aware, SQLite-safe Polyglot position key."""
    return f"{chess.polyglot.zobrist_hash(board):016x}"


def _game_id(game: chess.pgn.Game) -> str:
    digest = hashlib.sha256()
    digest.update(game.board().fen().encode("ascii"))
    identity = {
        key: game.headers.get(key, "")
        for key in ("Event", "Site", "Date", "Round", "White", "Black", "Result")
    }
    digest.update(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for move in game.mainline_moves():
        digest.update(move.uci().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _year(value: str) -> Optional[int]:
    try:
        year = int((value or "")[:4])
    except ValueError:
        return None
    return year if 1000 <= year <= 9999 else None


def _rating(headers, name: str) -> Optional[int]:
    try:
        value = int(headers.get(name, ""))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _result_counts(result: str) -> Optional[tuple[int, int, int]]:
    return {
        "1-0": (1, 0, 0),
        "1/2-1/2": (0, 1, 0),
        "0-1": (0, 0, 1),
    }.get(result)


def _open_text(path: str) -> Iterator[TextIO]:
    """Yield text streams from PGN, gzip, bzip2, or Elite-style ZIP files."""
    lower = path.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist()
                       if name.lower().endswith((".pgn", ".txt")) and not name.endswith("/")]
            if not members:
                raise ValueError(f"ZIP contains no PGN files: {path}")
            for name in members:
                with archive.open(name) as raw:
                    with io.TextIOWrapper(raw, encoding="utf-8", errors="replace") as text:
                        yield text
        return
    if lower.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as text:
            yield text
        return
    if lower.endswith(".bz2"):
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as text:
            yield text
        return
    with open(path, encoding="utf-8", errors="replace") as text:
        yield text


def _iter_games(paths: list[str]) -> Iterator[chess.pgn.Game]:
    for path in paths:
        for stream in _open_text(path):
            while True:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                yield game


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY
        );
        CREATE TABLE move_stats (
            position_key TEXT NOT NULL,
            move_uci TEXT NOT NULL,
            white_wins INTEGER NOT NULL,
            draws INTEGER NOT NULL,
            black_wins INTEGER NOT NULL,
            rating_sum INTEGER NOT NULL,
            rating_count INTEGER NOT NULL,
            first_year INTEGER,
            last_year INTEGER,
            PRIMARY KEY (position_key, move_uci)
        ) WITHOUT ROWID;
        CREATE INDEX move_stats_position ON move_stats(position_key);
    """)


def _source_identity(path: str) -> dict:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "filename": os.path.basename(path),
        "bytes": os.path.getsize(path),
        "sha256": digest.hexdigest(),
    }


def build_master_database(paths: list[str], output: str, *, max_ply: int = DEFAULT_MAX_PLY,
                          min_rating: int = 0, force: bool = False,
                          batch_games: int = 1000) -> BuildStats:
    """Build a fresh, atomic SQLite index from one or more PGN archives."""
    if not paths:
        raise ValueError("At least one PGN source is required")
    if max_ply < 1 or min_rating < 0 or batch_games < 1:
        raise ValueError("Invalid master database build settings")
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(missing[0])
    output = os.path.abspath(output)
    if os.path.exists(output) and not force:
        raise FileExistsError(f"Database already exists: {output}")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temp = output + ".tmp"
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(temp + suffix)
        except FileNotFoundError:
            pass

    stats = BuildStats()
    connection = sqlite3.connect(temp)
    try:
        _create_schema(connection)
        upsert = """
            INSERT INTO move_stats (
                position_key, move_uci, white_wins, draws, black_wins,
                rating_sum, rating_count, first_year, last_year
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(position_key, move_uci) DO UPDATE SET
                white_wins = white_wins + excluded.white_wins,
                draws = draws + excluded.draws,
                black_wins = black_wins + excluded.black_wins,
                rating_sum = rating_sum + excluded.rating_sum,
                rating_count = rating_count + excluded.rating_count,
                first_year = CASE
                    WHEN first_year IS NULL THEN excluded.first_year
                    WHEN excluded.first_year IS NULL THEN first_year
                    ELSE MIN(first_year, excluded.first_year)
                END,
                last_year = CASE
                    WHEN last_year IS NULL THEN excluded.last_year
                    WHEN excluded.last_year IS NULL THEN last_year
                    ELSE MAX(last_year, excluded.last_year)
                END
        """
        for game in _iter_games(paths):
            stats.games_seen += 1
            headers = game.headers
            counts = _result_counts(headers.get("Result", ""))
            white_rating = _rating(headers, "WhiteElo")
            black_rating = _rating(headers, "BlackElo")
            variant = (headers.get("Variant") or "Standard").lower()
            if game.errors or counts is None or variant not in ("standard", "chess"):
                stats.skipped_invalid += 1
                continue
            if min_rating and (white_rating is None or black_rating is None or
                               min(white_rating, black_rating) < min_rating):
                stats.skipped_rating += 1
                continue
            game_id = _game_id(game)
            inserted = connection.execute(
                "INSERT OR IGNORE INTO games(game_id) VALUES (?)", (game_id,)).rowcount
            if not inserted:
                stats.duplicate_games += 1
                continue

            rating_values = [rating for rating in (white_rating, black_rating)
                             if rating is not None]
            rating_sum = sum(rating_values)
            rating_count = len(rating_values)
            year = _year(headers.get("Date", ""))
            board = game.board()
            indexed = 0
            for ply, move in enumerate(game.mainline_moves(), start=1):
                if ply > max_ply:
                    break
                connection.execute(upsert, (
                    position_key(board), move.uci(), *counts,
                    rating_sum, rating_count, year, year,
                ))
                board.push(move)
                indexed += 1
            stats.positions_indexed += indexed
            stats.games_indexed += 1
            if stats.games_indexed % batch_games == 0:
                connection.commit()

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "database_version": datetime.now(timezone.utc).date().isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "max_ply": max_ply,
            "min_rating": min_rating,
            "sources": [_source_identity(path) for path in paths],
            "stats": stats.as_dict(),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [(key, json.dumps(value, ensure_ascii=False, sort_keys=True))
             for key, value in metadata.items()],
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        connection.close()
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(temp + suffix)
            except FileNotFoundError:
                pass
        raise
    else:
        connection.close()
    os.replace(temp, output)
    return stats


class MasterOpeningDatabase:
    """Read-only query interface for a locally built master opening database."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        self._connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self._metadata = self._read_metadata()
        if self._metadata.get("schema_version") != SCHEMA_VERSION:
            self.close()
            raise ValueError("Unsupported master database schema")

    def _read_metadata(self) -> dict:
        try:
            rows = self._connection.execute("SELECT key, value FROM metadata")
            return {key: json.loads(value) for key, value in rows}
        except sqlite3.DatabaseError as exc:
            raise ValueError("Invalid master opening database") from exc

    def metadata(self) -> dict:
        return dict(self._metadata)

    def identity(self) -> dict:
        """Return a content-addressed identity suitable for report provenance."""
        return {
            "filename": os.path.basename(self.path),
            "sha256": _source_identity(self.path)["sha256"],
            "schema_version": self._metadata.get("schema_version"),
            "database_version": self._metadata.get("database_version"),
            "max_ply": self._metadata.get("max_ply"),
            "min_rating": self._metadata.get("min_rating"),
        }

    def lookup(self, board_or_fen: chess.Board | str,
               top: int = 5) -> Optional[MasterPosition]:
        if top < 0:
            raise ValueError("top cannot be negative")
        board = (chess.Board(board_or_fen) if isinstance(board_or_fen, str)
                 else board_or_fen.copy(stack=False))
        key = position_key(board)
        rows = list(self._connection.execute("""
            SELECT move_uci, white_wins, draws, black_wins, rating_sum,
                   rating_count, first_year, last_year
            FROM move_stats
            WHERE position_key = ?
            ORDER BY (white_wins + draws + black_wins) DESC, move_uci
        """, (key,)))
        if not rows:
            return None
        total = sum(row[1] + row[2] + row[3] for row in rows)
        moves = []
        selected_rows = rows if top == 0 else rows[:top]
        for row in selected_rows:
            uci, white, draws, black, rating_sum, rating_count, first_year, last_year = row
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                continue
            games = white + draws + black
            moves.append(MasterMoveStats(
                uci=uci,
                san=board.san(move),
                games=games,
                white_wins=white,
                draws=draws,
                black_wins=black,
                avg_rating=round(rating_sum / rating_count) if rating_count else None,
                play_rate=round(games / total * 100, 1),
                first_year=first_year,
                last_year=last_year,
            ))
        if not moves:
            return None
        return MasterPosition(
            position_key=key,
            total_games=total,
            moves=moves,
            source="local-sqlite",
            database_version=str(self._metadata.get("database_version", "")),
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "MasterOpeningDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()