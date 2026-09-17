"""Persistent, versioned cache for long-term player analysis."""
from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import uuid
import zlib
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Iterator, Optional

import chess.pgn

from .models import GameAnalysis, MoveAnalysis
from .opening_book import OpeningBook
from .polyglot_book import get_default_book

SCHEMA_VERSION = "2"
ANALYSIS_FORMAT_VERSION = "1"
DEFAULT_HISTORY_DB = "data/player-history.sqlite"


class HistoryDataError(ValueError):
    """Stored history data is corrupt or incompatible with this version."""


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha256_value(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_identity(path: Optional[str]) -> Optional[dict]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "filename": os.path.basename(path),
        "sha256": digest.hexdigest(),
    }


def _normalize_player(name: str) -> str:
    return " ".join(name.casefold().split())


def history_game_id(game: chess.pgn.Game) -> str:
    """Identify a real game while deduplicating repeated PGN imports."""
    headers = game.headers
    payload = {
        "headers": {
            key: headers.get(key, "")
            for key in ("Event", "Site", "Date", "Round", "White", "Black", "Result")
        },
        "initial_fen": game.board().fen(),
        "moves": [move.uci() for move in game.mainline_moves()],
    }
    return _sha256_value(payload)


def analysis_profile(engine_metadata: dict, opening_book: Optional[OpeningBook] = None,
                     master_identity: Optional[dict] = None) -> dict:
    """Describe every input that can change a cached ``GameAnalysis``."""
    engine = dict(engine_metadata)
    engine_path = engine.pop("path", None)
    source_dir = os.path.dirname(__file__)
    polyglot = get_default_book()
    try:
        chess_version = version("python-chess")
    except PackageNotFoundError:
        chess_version = "unknown"
    return {
        "format_version": ANALYSIS_FORMAT_VERSION,
        "python_chess_version": chess_version,
        "engine": {
            **engine,
            "binary": _file_identity(engine_path),
        },
        "analysis_code": {
            filename: _file_identity(os.path.join(source_dir, filename))
            for filename in (
                "analysis.py", "classify.py", "engine.py", "master_db.py",
                "models.py", "opening_book.py", "polyglot_book.py",
            )
        },
        "opening_books": {
            "named": _file_identity(opening_book.path) if opening_book else None,
            "polyglot": _file_identity(polyglot.path),
        },
        "master_database": master_identity,
    }


def analysis_profile_id(profile: dict) -> str:
    return _sha256_value(profile)


def _serialize_game(game: chess.pgn.Game) -> str:
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
    return game.accept(exporter)


def _serialize_analysis(analysis: GameAnalysis) -> bytes:
    record = {
        "format_version": ANALYSIS_FORMAT_VERSION,
        "analysis": asdict(analysis),
    }
    payload = _canonical_json(record).encode("utf-8")
    return zlib.compress(payload, level=9)


def _deserialize_analysis(payload: bytes) -> GameAnalysis:
    try:
        record = json.loads(zlib.decompress(payload).decode("utf-8"))
        if not isinstance(record, dict) or \
                record.get("format_version") != ANALYSIS_FORMAT_VERSION:
            raise HistoryDataError("Unsupported cached analysis format")
        data = record.get("analysis")
        if not isinstance(data, dict):
            raise HistoryDataError("Invalid cached analysis payload")
        data["moves"] = [MoveAnalysis(**move) for move in data.get("moves", [])]
        return GameAnalysis(**data)
    except HistoryDataError:
        raise
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError, zlib.error) as exc:
        raise HistoryDataError("Invalid cached analysis payload") from exc


class PlayerHistoryStore:
    """SQLite-backed PGN archive and content-addressed analysis cache."""

    def __init__(self, path: str = DEFAULT_HISTORY_DB) -> None:
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                pgn TEXT NOT NULL,
                white TEXT NOT NULL,
                white_key TEXT NOT NULL,
                black TEXT NOT NULL,
                black_key TEXT NOT NULL,
                result TEXT NOT NULL,
                played_date TEXT NOT NULL,
                event TEXT NOT NULL,
                site TEXT NOT NULL,
                ingested_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS games_white_key ON games(white_key);
            CREATE INDEX IF NOT EXISTS games_black_key ON games(black_key);
            CREATE TABLE IF NOT EXISTS analysis_profiles (
                profile_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS game_analyses (
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                profile_id TEXT NOT NULL REFERENCES analysis_profiles(profile_id),
                analysis BLOB NOT NULL,
                analyzed_at TEXT NOT NULL,
                PRIMARY KEY (game_id, profile_id)
            );
            CREATE INDEX IF NOT EXISTS game_analyses_profile
                ON game_analyses(profile_id);
        """)
        stored = self._connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if stored is not None and stored[0] not in ("1", SCHEMA_VERSION):
            self.close()
            raise ValueError("Unsupported player history database schema")
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS people (
                person_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS person_aliases (
                person_id TEXT NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
                alias TEXT NOT NULL,
                alias_key TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (person_id, alias_key)
            );
            CREATE INDEX IF NOT EXISTS person_aliases_person
                ON person_aliases(person_id);
            CREATE TABLE IF NOT EXISTS external_accounts (
                account_id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
                source TEXT NOT NULL,
                external_id TEXT NOT NULL,
                username TEXT,
                profile_url TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (source, external_id)
            );
            CREATE INDEX IF NOT EXISTS external_accounts_person
                ON external_accounts(person_id);
            CREATE TABLE IF NOT EXISTS source_sync_state (
                account_id TEXT PRIMARY KEY
                    REFERENCES external_accounts(account_id) ON DELETE CASCADE,
                cursor_json TEXT NOT NULL,
                etag TEXT,
                last_modified TEXT,
                last_attempt_at TEXT,
                last_success_at TEXT,
                last_error TEXT
            );
            CREATE TABLE IF NOT EXISTS game_participants (
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                person_id TEXT NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
                color TEXT NOT NULL CHECK (color IN ('white', 'black')),
                linked_by TEXT NOT NULL,
                PRIMARY KEY (game_id, person_id),
                UNIQUE (game_id, color)
            );
            CREATE INDEX IF NOT EXISTS game_participants_person
                ON game_participants(person_id);
            CREATE TABLE IF NOT EXISTS game_sources (
                source TEXT NOT NULL,
                source_game_id TEXT NOT NULL,
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                account_id TEXT REFERENCES external_accounts(account_id)
                    ON DELETE SET NULL,
                source_url TEXT,
                retrieved_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (source, source_game_id)
            );
            CREATE INDEX IF NOT EXISTS game_sources_game
                ON game_sources(game_id);
        """)
        self._connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (SCHEMA_VERSION,),
        )
        self._connection.commit()

    @staticmethod
    def _required_text(value: str, field: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError(f"{field} must not be empty")
        return normalized

    def create_person(self, display_name: str, aliases: tuple[str, ...] = ()) -> str:
        """Create one canonical person and reserve their exact PGN aliases."""
        name = self._required_text(display_name, "display_name")
        person_id = f"person_{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connection:
                self._connection.execute("""
                    INSERT INTO people(person_id, display_name, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (person_id, name, now, now))
                for alias in (name, *aliases):
                    self._add_alias(person_id, alias, "manual", now)
        except sqlite3.IntegrityError as exc:
            raise ValueError("A person alias is already assigned") from exc
        return person_id

    def _add_alias(self, person_id: str, alias: str, source: str, now: str) -> bool:
        value = self._required_text(alias, "alias")
        alias_key = _normalize_player(value)
        inserted = self._connection.execute("""
            INSERT OR IGNORE INTO person_aliases (
                person_id, alias, alias_key, source, created_at
            ) VALUES (?, ?, ?, ?, ?)
        """, (person_id, value, alias_key, source, now)).rowcount > 0
        owner = self._connection.execute(
            "SELECT person_id FROM person_aliases WHERE alias_key=?",
            (alias_key,),
        ).fetchone()
        if owner is None or owner[0] != person_id:
            raise ValueError("A person alias is already assigned")
        self._connection.execute("""
            INSERT OR IGNORE INTO game_participants(game_id, person_id, color, linked_by)
            SELECT game_id, ?, 'white', 'alias' FROM games WHERE white_key=?
        """, (person_id, alias_key))
        self._connection.execute("""
            INSERT OR IGNORE INTO game_participants(game_id, person_id, color, linked_by)
            SELECT game_id, ?, 'black', 'alias' FROM games WHERE black_key=?
        """, (person_id, alias_key))
        return inserted

    def add_alias(self, person_id: str, alias: str, source: str = "manual") -> bool:
        """Add an exact PGN name alias and link matching stored games."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connection:
                inserted = self._add_alias(
                    person_id, alias, self._required_text(source, "source"), now)
                self._connection.execute(
                    "UPDATE people SET updated_at=? WHERE person_id=?",
                    (now, person_id),
                )
                if self._connection.execute("SELECT changes()").fetchone()[0] == 0:
                    raise ValueError(f"Unknown person: {person_id}")
        except sqlite3.IntegrityError as exc:
            raise ValueError("A person alias is already assigned") from exc
        return inserted

    def add_external_account(self, person_id: str, source: str, external_id: str,
                             username: Optional[str] = None,
                             profile_url: Optional[str] = None,
                             metadata: Optional[dict] = None) -> str:
        """Bind a stable source account to a canonical person."""
        source_key = self._required_text(source, "source").casefold()
        stable_id = self._required_text(external_id, "external_id")
        account_id = _sha256_value({"source": source_key, "external_id": stable_id})
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connection:
                existing = self._connection.execute("""
                    SELECT account_id, person_id FROM external_accounts
                    WHERE source=? AND external_id=?
                """, (source_key, stable_id)).fetchone()
                if existing is not None and existing[1] != person_id:
                    raise ValueError("External account is already assigned")
                normalized_username = (
                    self._required_text(username, "username") if username else None)
                if existing is None:
                    self._connection.execute("""
                        INSERT INTO external_accounts (
                            account_id, person_id, source, external_id, username,
                            profile_url, metadata_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        account_id, person_id, source_key, stable_id,
                        normalized_username, profile_url,
                        _canonical_json(metadata or {}), now, now,
                    ))
                else:
                    account_id = existing[0]
                    self._connection.execute("""
                        UPDATE external_accounts
                        SET username=?, profile_url=?, metadata_json=?, updated_at=?
                        WHERE account_id=?
                    """, (normalized_username, profile_url,
                          _canonical_json(metadata or {}), now, account_id))
                if username:
                    self._add_alias(person_id, username, source_key, now)
        except sqlite3.IntegrityError as exc:
            raise ValueError("External account is already assigned or person is unknown") from exc
        return account_id

    def people(self) -> list[dict]:
        """Return canonical people with aliases and external accounts."""
        rows = self._connection.execute("""
            SELECT person_id, display_name, created_at, updated_at
            FROM people ORDER BY display_name, person_id
        """)
        result = []
        for person_id, display_name, created_at, updated_at in rows:
            aliases = [row[0] for row in self._connection.execute("""
                SELECT alias FROM person_aliases
                WHERE person_id=?
                ORDER BY CASE WHEN alias_key=? THEN 0 ELSE 1 END, alias_key
            """, (person_id, _normalize_player(display_name)))]
            accounts = []
            for row in self._connection.execute("""
                SELECT account_id, source, external_id, username, profile_url,
                       metadata_json
                FROM external_accounts WHERE person_id=? ORDER BY source, external_id
            """, (person_id,)):
                accounts.append({
                    "account_id": row[0], "source": row[1], "external_id": row[2],
                    "username": row[3], "profile_url": row[4],
                    "metadata": json.loads(row[5]),
                })
            result.append({
                "person_id": person_id, "display_name": display_name,
                "created_at": created_at, "updated_at": updated_at,
                "aliases": aliases, "accounts": accounts,
            })
        return result

    def external_account(self, account_id: str) -> dict:
        row = self._connection.execute("""
            SELECT account_id, person_id, source, external_id, username,
                   profile_url, metadata_json
            FROM external_accounts WHERE account_id=?
        """, (account_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown external account: {account_id}")
        return {
            "account_id": row[0], "person_id": row[1], "source": row[2],
            "external_id": row[3], "username": row[4], "profile_url": row[5],
            "metadata": json.loads(row[6]),
        }

    def person_identity(self, person_id: str) -> dict:
        row = self._connection.execute("""
            SELECT display_name FROM people WHERE person_id=?
        """, (person_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown person: {person_id}")
        aliases = [item[0] for item in self._connection.execute("""
            SELECT alias FROM person_aliases
            WHERE person_id=?
            ORDER BY CASE WHEN alias_key=? THEN 0 ELSE 1 END, alias_key
        """, (person_id, _normalize_player(row[0])))]
        return {
            "person_id": person_id,
            "display_name": row[0],
            "aliases": aliases,
        }

    def save_sync_state(self, account_id: str, cursor: dict,
                        etag: Optional[str] = None,
                        last_modified: Optional[str] = None,
                        error: Optional[str] = None) -> None:
        """Persist a connector cursor without exposing provider details to the schema."""
        now = datetime.now(timezone.utc).isoformat()
        success_at = None if error else now
        try:
            with self._connection:
                self._connection.execute("""
                    INSERT INTO source_sync_state (
                        account_id, cursor_json, etag, last_modified,
                        last_attempt_at, last_success_at, last_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(account_id) DO UPDATE SET
                        cursor_json=excluded.cursor_json,
                        etag=excluded.etag,
                        last_modified=excluded.last_modified,
                        last_attempt_at=excluded.last_attempt_at,
                        last_success_at=COALESCE(excluded.last_success_at,
                                                 source_sync_state.last_success_at),
                        last_error=excluded.last_error
                """, (account_id, _canonical_json(cursor), etag, last_modified,
                      now, success_at, error))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Unknown external account: {account_id}") from exc

    def sync_state(self, account_id: str) -> Optional[dict]:
        row = self._connection.execute("""
            SELECT cursor_json, etag, last_modified, last_attempt_at,
                   last_success_at, last_error
            FROM source_sync_state WHERE account_id=?
        """, (account_id,)).fetchone()
        if row is None:
            return None
        return {
            "cursor": json.loads(row[0]), "etag": row[1],
            "last_modified": row[2], "last_attempt_at": row[3],
            "last_success_at": row[4], "last_error": row[5],
        }

    def person_games(self, person_id: str) -> Iterator[tuple[str, chess.pgn.Game]]:
        rows = self._connection.execute("""
            SELECT g.game_id, g.pgn FROM games g
            JOIN game_participants gp ON gp.game_id=g.game_id
            WHERE gp.person_id=? ORDER BY g.played_date, g.game_id
        """, (person_id,))
        for game_id, pgn in rows:
            game = chess.pgn.read_game(io.StringIO(pgn))
            if game is None or game.errors:
                raise HistoryDataError(f"Invalid stored PGN for game {game_id}")
            yield game_id, game

    def person_games_recent(
            self, person_id: str, limit: int) -> Iterator[tuple[str, chess.pgn.Game]]:
        """Return a person's newest stored games first, bounded by ``limit``."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = self._connection.execute("""
            SELECT g.game_id, g.pgn FROM games g
            JOIN game_participants gp ON gp.game_id=g.game_id
            WHERE gp.person_id=?
            ORDER BY NULLIF(g.played_date, '') DESC, g.game_id DESC
            LIMIT ?
        """, (person_id, limit))
        for game_id, pgn in rows:
            game = chess.pgn.read_game(io.StringIO(pgn))
            if game is None or game.errors:
                raise HistoryDataError(f"Invalid stored PGN for game {game_id}")
            yield game_id, game

    def ingest(self, game: chess.pgn.Game, *, source: Optional[str] = None,
               source_game_id: Optional[str] = None,
               source_url: Optional[str] = None,
               account_id: Optional[str] = None,
               source_metadata: Optional[dict] = None) -> tuple[str, bool]:
        game_id = history_game_id(game)
        headers = game.headers
        now = datetime.now(timezone.utc).isoformat()
        source_key = self._required_text(source, "source").casefold() if source else None
        stable_source_id = (
            self._required_text(source_game_id, "source_game_id")
            if source_game_id else None)
        if (source_key is None) != (stable_source_id is None):
            raise ValueError("source and source_game_id must be provided together")
        with self._connection:
            if source_key is not None:
                existing_source = self._connection.execute("""
                    SELECT game_id FROM game_sources
                    WHERE source=? AND source_game_id=?
                """, (source_key, stable_source_id)).fetchone()
                if existing_source is not None and existing_source[0] != game_id:
                    raise HistoryDataError(
                        f"Source game changed: {source_key}/{stable_source_id}")
            inserted = self._connection.execute("""
                INSERT OR IGNORE INTO games (
                    game_id, pgn, white, white_key, black, black_key, result,
                    played_date, event, site, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id,
                _serialize_game(game),
                headers.get("White", "?"),
                _normalize_player(headers.get("White", "?")),
                headers.get("Black", "?"),
                _normalize_player(headers.get("Black", "?")),
                headers.get("Result", "*"),
                headers.get("Date", headers.get("UTCDate", "")),
                headers.get("Event", ""),
                headers.get("Site", ""),
                now,
            )).rowcount > 0
            for color, player_key in (
                ("white", _normalize_player(headers.get("White", "?"))),
                ("black", _normalize_player(headers.get("Black", "?"))),
            ):
                owner = self._connection.execute(
                    "SELECT person_id FROM person_aliases WHERE alias_key=?",
                    (player_key,),
                ).fetchone()
                if owner is not None:
                    self._connection.execute("""
                        INSERT OR IGNORE INTO game_participants (
                            game_id, person_id, color, linked_by
                        ) VALUES (?, ?, ?, 'alias')
                    """, (game_id, owner[0], color))
            if source_key is not None:
                self._connection.execute("""
                    INSERT OR REPLACE INTO game_sources (
                        source, source_game_id, game_id, account_id, source_url,
                        retrieved_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (source_key, stable_source_id, game_id, account_id,
                      source_url, now, _canonical_json(source_metadata or {})))
        return game_id, inserted

    def register_profile(self, profile: dict) -> str:
        profile_id = analysis_profile_id(profile)
        self._connection.execute("""
            INSERT OR IGNORE INTO analysis_profiles (
                profile_id, profile_json, created_at
            ) VALUES (?, ?, ?)
        """, (profile_id, _canonical_json(profile),
              datetime.now(timezone.utc).isoformat()))
        self._connection.commit()
        return profile_id

    def player_games(self, player: str) -> Iterator[tuple[str, chess.pgn.Game]]:
        player_key = _normalize_player(player)
        rows = self._connection.execute("""
            SELECT game_id, pgn FROM games
            WHERE white_key=? OR black_key=?
            ORDER BY played_date, game_id
        """, (player_key, player_key))
        for game_id, pgn in rows:
            game = chess.pgn.read_game(io.StringIO(pgn))
            if game is None or game.errors:
                raise HistoryDataError(f"Invalid stored PGN for game {game_id}")
            yield game_id, game

    def player_games_recent(
            self, player: str, limit: int) -> Iterator[tuple[str, chess.pgn.Game]]:
        """Return the newest games matching a legacy player name."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        player_key = _normalize_player(player)
        rows = self._connection.execute("""
            SELECT game_id, pgn FROM games
            WHERE white_key=? OR black_key=?
            ORDER BY NULLIF(played_date, '') DESC, game_id DESC
            LIMIT ?
        """, (player_key, player_key, limit))
        for game_id, pgn in rows:
            game = chess.pgn.read_game(io.StringIO(pgn))
            if game is None or game.errors:
                raise HistoryDataError(f"Invalid stored PGN for game {game_id}")
            yield game_id, game

    def has_analysis(self, game_id: str, profile_id: str) -> bool:
        row = self._connection.execute("""
            SELECT 1 FROM game_analyses WHERE game_id=? AND profile_id=?
        """, (game_id, profile_id)).fetchone()
        return row is not None

    def save_analysis(self, game_id: str, profile_id: str,
                      analysis: GameAnalysis) -> None:
        self._connection.execute("""
            INSERT OR REPLACE INTO game_analyses (
                game_id, profile_id, analysis, analyzed_at
            ) VALUES (?, ?, ?, ?)
        """, (game_id, profile_id, _serialize_analysis(analysis),
              datetime.now(timezone.utc).isoformat()))
        self._connection.commit()

    def load_player_analyses(self, player: str, profile_id: str) -> list[GameAnalysis]:
        player_key = _normalize_player(player)
        rows = self._connection.execute("""
            SELECT ga.game_id, ga.analysis FROM game_analyses ga
            JOIN games g ON g.game_id = ga.game_id
            WHERE ga.profile_id=? AND (g.white_key=? OR g.black_key=?)
            ORDER BY g.played_date, g.game_id
        """, (profile_id, player_key, player_key))
        analyses = []
        for game_id, payload in rows:
            try:
                analyses.append(_deserialize_analysis(payload))
            except HistoryDataError as exc:
                raise HistoryDataError(
                    f"Invalid cached analysis for game {game_id}") from exc
        return analyses

    def load_person_analyses(self, person_id: str,
                             profile_id: str) -> list[GameAnalysis]:
        rows = self._connection.execute("""
            SELECT ga.game_id, ga.analysis FROM game_analyses ga
            JOIN game_participants gp ON gp.game_id=ga.game_id
            JOIN games g ON g.game_id=ga.game_id
            WHERE ga.profile_id=? AND gp.person_id=?
            ORDER BY g.played_date, g.game_id
        """, (profile_id, person_id))
        analyses = []
        for game_id, payload in rows:
            try:
                analyses.append(_deserialize_analysis(payload))
            except HistoryDataError as exc:
                raise HistoryDataError(
                    f"Invalid cached analysis for game {game_id}") from exc
        return analyses

    def latest_profile_id(self, player: str) -> Optional[str]:
        player_key = _normalize_player(player)
        row = self._connection.execute("""
            SELECT ga.profile_id, MAX(ga.analyzed_at) AS latest
            FROM game_analyses ga
            JOIN games g ON g.game_id = ga.game_id
            WHERE g.white_key=? OR g.black_key=?
            GROUP BY ga.profile_id
            ORDER BY latest DESC
            LIMIT 1
        """, (player_key, player_key)).fetchone()
        return row[0] if row else None

    def latest_person_profile_id(self, person_id: str) -> Optional[str]:
        row = self._connection.execute("""
            SELECT ga.profile_id, MAX(ga.analyzed_at) AS latest
            FROM game_analyses ga
            JOIN game_participants gp ON gp.game_id=ga.game_id
            WHERE gp.person_id=?
            GROUP BY ga.profile_id
            ORDER BY latest DESC
            LIMIT 1
        """, (person_id,)).fetchone()
        return row[0] if row else None

    def coach_students(self) -> list[dict]:
        """Return lightweight coverage and sync facts for the coach dashboard."""
        students = []
        for person in self.people():
            person_id = person["person_id"]
            game_row = self._connection.execute("""
                SELECT COUNT(*), MIN(NULLIF(g.played_date, '')),
                       MAX(NULLIF(g.played_date, ''))
                FROM games g
                JOIN game_participants gp ON gp.game_id=g.game_id
                WHERE gp.person_id=?
            """, (person_id,)).fetchone()
            profile_id = self.latest_person_profile_id(person_id)
            analyzed = 0
            if profile_id is not None:
                analyzed = self._connection.execute("""
                    SELECT COUNT(*) FROM game_analyses ga
                    JOIN game_participants gp ON gp.game_id=ga.game_id
                    WHERE gp.person_id=? AND ga.profile_id=?
                """, (person_id, profile_id)).fetchone()[0]
            accounts = []
            for account in person["accounts"]:
                accounts.append({**account, "sync": self.sync_state(account["account_id"])})
            games = game_row[0]
            students.append({
                **person,
                "accounts": accounts,
                "games": games,
                "analyzed_games": analyzed,
                "analysis_coverage": analyzed / games if games else 0.0,
                "first_game_date": game_row[1],
                "last_game_date": game_row[2],
                "latest_profile_id": profile_id,
            })
        return students

    def info(self, player: Optional[str] = None) -> dict:
        game_filter = ""
        params: tuple = ()
        if player:
            game_filter = " WHERE white_key=? OR black_key=?"
            player_key = _normalize_player(player)
            params = (player_key, player_key)
        games = self._connection.execute(
            "SELECT COUNT(*) FROM games" + game_filter, params).fetchone()[0]
        if player:
            analyses = self._connection.execute("""
                SELECT COUNT(*) FROM game_analyses ga
                JOIN games g ON g.game_id=ga.game_id
                WHERE g.white_key=? OR g.black_key=?
            """, params).fetchone()[0]
        else:
            analyses = self._connection.execute(
                "SELECT COUNT(*) FROM game_analyses").fetchone()[0]
        if player:
            profile_rows = self._connection.execute("""
                SELECT ap.profile_id, ap.profile_json, ap.created_at,
                       COUNT(ga.game_id)
                FROM analysis_profiles ap
                JOIN game_analyses ga ON ga.profile_id=ap.profile_id
                JOIN games g ON g.game_id=ga.game_id
                WHERE g.white_key=? OR g.black_key=?
                GROUP BY ap.profile_id, ap.profile_json, ap.created_at
                ORDER BY ap.created_at DESC
            """, params)
        else:
            profile_rows = self._connection.execute("""
                SELECT ap.profile_id, ap.profile_json, ap.created_at,
                       COUNT(ga.game_id)
                FROM analysis_profiles ap
                LEFT JOIN game_analyses ga ON ga.profile_id=ap.profile_id
                GROUP BY ap.profile_id, ap.profile_json, ap.created_at
                ORDER BY ap.created_at DESC
            """)
        profile_details = []
        for profile_id, profile_json, created_at, cached_games in profile_rows:
            profile = json.loads(profile_json)
            profile_details.append({
                "profile_id": profile_id,
                "created_at": created_at,
                "cached_games": cached_games,
                "engine": profile.get("engine"),
                "master_database": profile.get("master_database"),
            })
        return {
            "schema_version": SCHEMA_VERSION,
            "path": self.path,
            "player": player,
            "games": games,
            "analyses": analyses,
            "profile_count": len(profile_details),
            "profiles": profile_details,
            "latest_profile_id": self.latest_profile_id(player) if player else None,
        }

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "PlayerHistoryStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()