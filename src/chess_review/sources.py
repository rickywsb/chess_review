"""Official public-game source connectors for player history."""
from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional

import chess.pgn

from .history import PlayerHistoryStore, history_game_id


class SourceSyncError(RuntimeError):
    """A public source could not be read without risking partial state."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]


@dataclass
class SyncResult:
    account_id: str
    source: str
    requests: int = 0
    games_seen: int = 0
    games_new: int = 0
    games_duplicate: int = 0
    games_skipped: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


HttpGet = Callable[[str, Mapping[str, str], int], HttpResponse]


def _http_get(url: str, headers: Mapping[str, str], timeout: int) -> HttpResponse:
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return HttpResponse(
                status=response.status,
                body=response.read(),
                headers={key.casefold(): value for key, value in response.headers.items()},
            )
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return HttpResponse(304, b"", {
                key.casefold(): value for key, value in exc.headers.items()})
        detail = exc.read(500).decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise SourceSyncError(f"HTTP {exc.code} from {url}{suffix}") from exc
    except urllib.error.URLError as exc:
        raise SourceSyncError(f"Could not reach {url}: {exc.reason}") from exc


def _json(response: HttpResponse, url: str) -> dict:
    try:
        data = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceSyncError(f"Invalid JSON from {url}") from exc
    if not isinstance(data, dict):
        raise SourceSyncError(f"Unexpected JSON from {url}")
    return data


def _parse_games(payload: bytes, label: str) -> list[chess.pgn.Game]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceSyncError(f"Invalid UTF-8 PGN from {label}") from exc
    stream = io.StringIO(text)
    games = []
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        if game.errors:
            raise SourceSyncError(f"Invalid PGN from {label}: {game.errors[0]}")
        games.append(game)
    return games


def _lichess_game_id(game: chess.pgn.Game) -> str:
    site = game.headers.get("Site", "")
    match = re.search(r"lichess\.org/(?:game/export/)?([A-Za-z0-9]{8,12})", site)
    return match.group(1) if match else history_game_id(game)


def _played_at_ms(game: chess.pgn.Game) -> Optional[int]:
    date = game.headers.get("UTCDate") or game.headers.get("Date")
    if not date or "?" in date:
        return None
    time = game.headers.get("UTCTime", "00:00:00")
    try:
        value = datetime.strptime(f"{date} {time}", "%Y.%m.%d %H:%M:%S")
    except ValueError:
        return None
    return int(value.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _sync_lichess(store: PlayerHistoryStore, account: dict, cursor: dict,
                  user_agent: str, token: Optional[str], timeout: int,
                  http_get: HttpGet) -> tuple[SyncResult, dict]:
    username = account.get("username") or account["external_id"]
    params = {
        "moves": "true", "tags": "true", "clocks": "true",
        "evals": "false", "opening": "true",
    }
    if cursor.get("since_ms") is not None:
        params["since"] = str(cursor["since_ms"])
    url = (
        "https://lichess.org/api/games/user/"
        f"{urllib.parse.quote(username, safe='')}?{urllib.parse.urlencode(params)}"
    )
    headers = {
        "Accept": "application/x-chess-pgn",
        "User-Agent": user_agent,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = http_get(url, headers, timeout)
    if response.status != 200:
        raise SourceSyncError(f"Unexpected HTTP {response.status} from Lichess")
    result = SyncResult(account["account_id"], "lichess", requests=1)
    latest_ms = cursor.get("since_ms")
    for game in _parse_games(response.body, "Lichess"):
        result.games_seen += 1
        played_at = _played_at_ms(game)
        if played_at is not None:
            latest_ms = max(latest_ms or played_at, played_at)
        source_id = _lichess_game_id(game)
        _, inserted = store.ingest(
            game, source="lichess", source_game_id=source_id,
            source_url=game.headers.get("Site"), account_id=account["account_id"],
            source_metadata={"username": username},
        )
        result.games_new += int(inserted)
        result.games_duplicate += int(not inserted)
    return result, {"since_ms": latest_ms} if latest_ms is not None else cursor


def _sync_chesscom(store: PlayerHistoryStore, account: dict, cursor: dict,
                   user_agent: str, timeout: int,
                   http_get: HttpGet) -> tuple[SyncResult, dict]:
    username = account.get("username") or account["external_id"]
    base = f"https://api.chess.com/pub/player/{urllib.parse.quote(username, safe='')}"
    archives_url = f"{base}/games/archives"
    base_headers = {"Accept": "application/json", "User-Agent": user_agent}
    archive_list_response = http_get(archives_url, base_headers, timeout)
    if archive_list_response.status != 200:
        raise SourceSyncError(
            f"Unexpected HTTP {archive_list_response.status} from Chess.com")
    archive_list = _json(archive_list_response, archives_url).get("archives")
    if not isinstance(archive_list, list) or not all(
            isinstance(url, str) for url in archive_list):
        raise SourceSyncError("Invalid Chess.com archive list")

    result = SyncResult(account["account_id"], "chess.com", requests=1)
    etags = dict(cursor.get("archive_etags", {}))
    candidates = [url for url in archive_list if url not in etags]
    if archive_list and archive_list[-1] not in candidates:
        candidates.append(archive_list[-1])
    for archive_url in candidates:
        headers = dict(base_headers)
        if etags.get(archive_url):
            headers["If-None-Match"] = etags[archive_url]
        response = http_get(archive_url, headers, timeout)
        result.requests += 1
        if response.status == 304:
            continue
        if response.status != 200:
            raise SourceSyncError(
                f"Unexpected HTTP {response.status} from Chess.com archive")
        games = _json(response, archive_url).get("games")
        if not isinstance(games, list):
            raise SourceSyncError(f"Invalid Chess.com games archive: {archive_url}")
        for source_game in games:
            if not isinstance(source_game, dict):
                raise SourceSyncError(f"Invalid Chess.com game: {archive_url}")
            if source_game.get("rules") not in (None, "chess"):
                result.games_skipped += 1
                continue
            if not isinstance(source_game.get("pgn"), str):
                raise SourceSyncError(f"Invalid Chess.com game: {archive_url}")
            parsed = _parse_games(
                source_game["pgn"].encode("utf-8"), archive_url)
            if len(parsed) != 1:
                raise SourceSyncError(f"Expected one Chess.com game: {archive_url}")
            game = parsed[0]
            source_url = source_game.get("url") or game.headers.get("Site")
            source_id = source_url or history_game_id(game)
            _, inserted = store.ingest(
                game, source="chess.com", source_game_id=source_id,
                source_url=source_url, account_id=account["account_id"],
                source_metadata={
                    key: source_game.get(key)
                    for key in ("end_time", "rated", "rules", "time_class",
                                "time_control")
                    if source_game.get(key) is not None
                },
            )
            result.games_seen += 1
            result.games_new += int(inserted)
            result.games_duplicate += int(not inserted)
        etags[archive_url] = response.headers.get("etag", "")
    return result, {"archive_etags": etags}


def sync_external_account(store: PlayerHistoryStore, account_id: str, *,
                          user_agent: str = "chess-review/0.1",
                          lichess_token: Optional[str] = None,
                          timeout: int = 60,
                          http_get: Optional[HttpGet] = None) -> SyncResult:
    """Fetch new public games for one registered source account."""
    account = store.external_account(account_id)
    state = store.sync_state(account_id)
    cursor = dict(state["cursor"]) if state else {}
    get = http_get or _http_get
    try:
        if account["source"] == "lichess":
            result, next_cursor = _sync_lichess(
                store, account, cursor, user_agent, lichess_token, timeout, get)
        elif account["source"] in ("chess.com", "chesscom"):
            result, next_cursor = _sync_chesscom(
                store, account, cursor, user_agent, timeout, get)
        else:
            raise SourceSyncError(
                f"Source does not support automatic sync: {account['source']}")
    except Exception as exc:
        message = str(exc)
        store.save_sync_state(account_id, cursor, error=message)
        if isinstance(exc, SourceSyncError):
            raise
        raise SourceSyncError(message) from exc
    store.save_sync_state(account_id, next_cursor)
    return result