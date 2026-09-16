"""Tests for persistent player history and incremental analysis caching."""
import base64
import io
import sqlite3
from argparse import Namespace
from dataclasses import asdict

import chess
import chess.pgn
import pytest

from chess_review.history import (
    HistoryDataError,
    PlayerHistoryStore,
    analysis_profile,
    analysis_profile_id,
    history_game_id,
)
from chess_review.models import GameAnalysis, MoveAnalysis
from chess_review.opening_book import OpeningBook
from chess_review.sources import HttpResponse, sync_external_account


def _game(white="Alice", black="Bob", site="game-1"):
    return chess.pgn.read_game(io.StringIO(
        f'[Event "Test"]\n[Site "{site}"]\n[Date "2026.09.13"]\n'
        f'[White "{white}"]\n[Black "{black}"]\n[Result "1-0"]\n\n'
        "1. e4 e5 2. Nf3 Nc6 1-0"
    ))


def _analysis(white="Alice", black="Bob", date="2026.09.13", site="game-1",
              cp_loss=5):
    move = MoveAnalysis(
        ply=1, move_number=1, color=chess.WHITE, san="e4", uci="e2e4",
        fen_before=chess.STARTING_FEN,
        fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        eval_before_mover=20, eval_after_mover=15,
        eval_before_white=20, eval_after_white=15, cp_loss=cp_loss,
        best_move_uci="e2e4", best_move_san="e4", phase="opening",
        classification="best", best_is_capture=False, best_is_check=False,
        played_is_capture=False, in_book=True,
        best_line_san=["e4", "e5"], master_context={"total_games": 100},
    )
    return GameAnalysis(
        white=white, black=black, result="1-0", date=date,
        event="Test", site=site, headers={"WhiteElo": "2100"},
        eco="C20", opening_name="King's Pawn Game", moves=[move],
        final_eval_white=15,
    )


def test_history_game_id_deduplicates_imports_but_not_distinct_games():
    assert history_game_id(_game()) == history_game_id(_game())
    assert history_game_id(_game(site="game-1")) != history_game_id(_game(site="game-2"))


def test_store_round_trips_analysis_and_keeps_profiles_separate(tmp_path):
    database = tmp_path / "history.sqlite"
    analysis = _analysis()
    with PlayerHistoryStore(str(database)) as store:
        game_id, inserted = store.ingest(_game())
        assert inserted
        assert store.ingest(_game()) == (game_id, False)

        first_profile = {"engine": {"depth": 13}}
        second_profile = {"engine": {"depth": 18}}
        first_id = store.register_profile(first_profile)
        second_id = store.register_profile(second_profile)
        assert first_id != second_id
        assert not store.has_analysis(game_id, first_id)

        store.save_analysis(game_id, first_id, analysis)
        assert store.has_analysis(game_id, first_id)
        assert not store.has_analysis(game_id, second_id)
        loaded = store.load_player_analyses("  ALICE  ", first_id)
        assert len(loaded) == 1
        assert asdict(loaded[0]) == asdict(analysis)
        assert store.latest_profile_id("alice") == first_id
        info = store.info("Alice")
        assert info["games"] == 1
        assert info["profile_count"] == 1
        assert info["profiles"][0]["cached_games"] == 1


def test_analysis_profile_changes_with_engine_settings(tmp_path):
    engine_path = tmp_path / "stockfish"
    engine_path.write_bytes(b"engine")
    opening_path = tmp_path / "openings.tsv"
    opening_path.write_text("eco\tname\tpgn\n", encoding="utf-8")
    book = OpeningBook()
    book.path = str(opening_path)
    metadata = {
        "name": "Stockfish Test", "author": "Test", "path": str(engine_path),
        "depth": 13, "movetime_ms": None, "threads": 1, "hash_mb": 16,
    }
    first = analysis_profile(metadata, book)
    second = analysis_profile({**metadata, "depth": 18}, book)
    assert analysis_profile_id(first) != analysis_profile_id(second)
    assert first["python_chess_version"] != "unknown"
    assert first["engine"]["binary"]["sha256"]
    assert "path" not in first["engine"]


def test_history_analyze_reuses_matching_cached_profile(tmp_path, monkeypatch):
    from chess_review import cli

    database = tmp_path / "history.sqlite"
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(str(_game()) + "\n\n", encoding="utf-8")
    assert cli.cmd_history_ingest(Namespace(
        pgn=[str(pgn_path)], db=str(database))) == 0

    engine_path = tmp_path / "stockfish"
    engine_path.write_bytes(b"fake-engine")

    class FakeEngine:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def metadata(self):
            return {
                "name": "Stockfish Test", "author": "Test",
                "path": str(engine_path), "depth": 18,
                "movetime_ms": None, "threads": 1, "hash_mb": 16,
            }

    calls = []
    monkeypatch.setattr(cli, "Engine", FakeEngine)
    monkeypatch.setattr(cli, "analyze_game",
                        lambda *_args, **_kwargs: calls.append(True) or _analysis())
    book = OpeningBook()
    opening_path = tmp_path / "openings.tsv"
    opening_path.write_text("eco\tname\tpgn\n", encoding="utf-8")
    book.path = str(opening_path)
    monkeypatch.setattr(cli.OpeningBook, "load", lambda: book)
    args = Namespace(
        db=str(database), player="Alice", limit=None, engine=None,
        depth=18, movetime=None, threads=1, hash_mb=16, master_db=None,
    )
    assert cli.cmd_history_analyze(args) == 0
    assert cli.cmd_history_analyze(args) == 0
    assert len(calls) == 1

    output = tmp_path / "reports"
    assert cli.cmd_history_report(Namespace(
        db=str(database), player="Alice", profile=None,
        out=str(output), format="md")) == 0
    assert (output / "Alice-history-report.md").is_file()


def test_history_parser_defaults():
    from chess_review.cli import build_parser

    parser = build_parser()
    ingest = parser.parse_args(["history", "ingest", "games.pgn"])
    assert ingest.db.endswith("player-history.sqlite")
    analyze = parser.parse_args(["history", "analyze", "--player", "Alice"])
    assert analyze.depth == 18
    assert analyze.threads == 1
    analyze_person = parser.parse_args([
        "history", "analyze", "--person-id", "person-1"])
    assert analyze_person.person_id == "person-1"
    report = parser.parse_args(["history", "report", "--player", "Alice"])
    assert report.profile is None
    report_person = parser.parse_args([
        "history", "report", "--person-id", "person-1"])
    assert report_person.person_id == "person-1"
    person = parser.parse_args(["history", "person", "add", "--name", "Alice"])
    assert person.alias == []
    account = parser.parse_args([
        "history", "account", "add", "--person", "person-1",
        "--source", "lichess", "--username", "alice",
    ])
    assert account.external_id is None
    sync = parser.parse_args([
        "history", "sync", "--account", "account-1",
    ])
    assert sync.timeout == 60


def test_person_identity_links_aliases_accounts_games_and_sync_state(tmp_path):
    database = tmp_path / "history.sqlite"
    with PlayerHistoryStore(str(database)) as store:
        game_id, _ = store.ingest(_game(white="Wu, Sibo"))
        person_id = store.create_person("Sibo Wu", ("Wu, Sibo",))
        account_id = store.add_external_account(
            person_id, "Lichess", "sibo-account", username="SiboOnline",
            profile_url="https://lichess.org/@/SiboOnline",
            metadata={"title": "FM"},
        )
        assert store.add_external_account(
            person_id, "lichess", "sibo-account", username="SiboOnline",
            profile_url="https://lichess.org/@/SiboOnline",
            metadata={"title": "FM"},
        ) == account_id
        assert [item[0] for item in store.person_games(person_id)] == [game_id]

        store.save_sync_state(account_id, {"since_ms": 123}, etag="first")
        successful = store.sync_state(account_id)
        assert successful["cursor"] == {"since_ms": 123}
        assert successful["last_success_at"]
        store.save_sync_state(account_id, {"since_ms": 123}, error="rate limited")
        failed = store.sync_state(account_id)
        assert failed["last_success_at"] == successful["last_success_at"]
        assert failed["last_error"] == "rate limited"

        person = store.people()[0]
        assert person["person_id"] == person_id
        assert person["aliases"] == ["Sibo Wu", "SiboOnline", "Wu, Sibo"]
        assert person["accounts"][0]["metadata"] == {"title": "FM"}
        with pytest.raises(ValueError, match="alias is already assigned"):
            store.create_person("Wu, Sibo")


def test_person_report_combines_exact_aliases_without_substring_matches(tmp_path):
    from chess_review.metrics import build_player_report

    database = tmp_path / "history.sqlite"
    with PlayerHistoryStore(str(database)) as store:
        first_game_id, _ = store.ingest(_game(white="Alice", site="game-1"))
        second_game_id, _ = store.ingest(
            _game(white="Carol", black="AliceOnline", site="game-2"))
        store.ingest(_game(white="Malice", black="Dan", site="game-3"))
        person_id = store.create_person("Alice", ("AliceOnline",))
        identity = store.person_identity(person_id)
        assert len(list(store.person_games(person_id))) == 2

        profile_id = store.register_profile({"engine": {"depth": 18}})
        store.save_analysis(first_game_id, profile_id, _analysis())
        store.save_analysis(
            second_game_id, profile_id,
            _analysis(white="Carol", black="AliceOnline", site="game-2"),
        )
        analyses = store.load_person_analyses(person_id, profile_id)
        report = build_player_report(
            analyses, identity["display_name"], aliases=identity["aliases"])
        assert report["n_games"] == 2
        assert report["as_white"] == 1
        assert report["as_black"] == 1
        assert store.latest_person_profile_id(person_id) == profile_id


def test_progress_compares_equal_windows_and_detects_improvement():
    from chess_review.metrics import build_player_report
    from chess_review.render import render_player_html, render_player_markdown

    analyses = [
        _analysis(date=f"2026.08.{index + 1:02d}", site=f"old-{index}", cp_loss=250)
        for index in range(10)
    ] + [
        _analysis(date=f"2026.09.{index + 1:02d}", site=f"new-{index}", cp_loss=10)
        for index in range(10)
    ]
    report = build_player_report(analyses, "Alice")
    progress = report["progress"]
    assert progress["window_games"] == 10
    assert progress["confidence"] == "medium"
    assert progress["status"] == "improving"
    assert progress["metrics"]["acpl_median"]["status"] == "improving"
    assert progress["metrics"]["blunders_per_100"]["status"] == "improving"
    assert "previous 10 vs recent 10" in render_player_html(report)
    assert "| Median ACPL |" in render_player_markdown(report)


def test_progress_refuses_to_infer_from_small_sample():
    from chess_review.metrics import build_player_report

    report = build_player_report([_analysis() for _ in range(19)], "Alice")
    assert report["progress"]["status"] == "insufficient_data"
    assert report["progress"]["required_games"] == 20


def test_progress_orders_by_date_and_excludes_undated_games():
    from chess_review.metrics import build_player_report

    analyses = [
        _analysis(date=f"2026.09.{index + 1:02d}", cp_loss=10)
        for index in range(10)
    ] + [
        _analysis(date="????.??.??", cp_loss=250)
    ] + [
        _analysis(date=f"2026.08.{index + 1:02d}", cp_loss=250)
        for index in range(10)
    ]
    progress = build_player_report(analyses, "Alice")["progress"]
    assert progress["status"] == "improving"
    assert progress["excluded_undated"] == 1
    assert progress["baseline"]["first_date"] == "2026.08.01"
    assert progress["recent"]["last_date"] == "2026.09.10"


def test_coach_students_reports_coverage_and_sync_health(tmp_path):
    database = tmp_path / "history.sqlite"
    with PlayerHistoryStore(str(database)) as store:
        person_id = store.create_person("Alice", ("AliceOnline",))
        game_id, _ = store.ingest(_game(white="AliceOnline"))
        account_id = store.add_external_account(
            person_id, "lichess", "alice", username="AliceOnline")
        store.save_sync_state(account_id, {"since_ms": 123})
        profile_id = store.register_profile({"engine": {"depth": 18}})
        store.save_analysis(game_id, profile_id, _analysis(white="AliceOnline"))

        student = store.coach_students()[0]
        assert student["games"] == 1
        assert student["analyzed_games"] == 1
        assert student["analysis_coverage"] == 1.0
        assert student["latest_profile_id"] == profile_id
        assert student["accounts"][0]["sync"]["last_success_at"]


def test_coach_dashboard_routes_are_read_only_and_remote_protected(
        tmp_path, monkeypatch):
    from chess_review import webapp

    database = tmp_path / "history.sqlite"
    with PlayerHistoryStore(str(database)) as store:
        person_id = store.create_person("Alice")
        store.ingest(_game())

    monkeypatch.setattr(webapp, "_HISTORY_DB", str(database))
    monkeypatch.delenv("CHESS_REVIEW_COACH_TOKEN", raising=False)
    client = webapp.create_app().test_client()
    assert client.get("/coach").status_code == 200
    listing = client.get("/api/coach/students").get_json()
    assert listing["students"][0]["person_id"] == person_id
    assert client.get(f"/api/coach/students/{person_id}").status_code == 200
    assert client.get("/api/coach/students/missing").status_code == 404

    remote = {"REMOTE_ADDR": "10.0.0.2", "HTTP_FLY_CLIENT_IP": "203.0.113.8"}
    assert client.get("/coach", environ_overrides=remote).status_code == 401
    monkeypatch.setenv("CHESS_REVIEW_COACH_TOKEN", "test-secret")
    credentials = base64.b64encode(b"coach:test-secret").decode("ascii")
    response = client.get(
        "/coach", environ_overrides=remote,
        headers={"Authorization": f"Basic {credentials}"},
    )
    assert response.status_code == 200
    assert "教练工作台" in response.get_data(as_text=True)


def test_game_source_provenance_is_idempotent_and_fails_on_changed_game(tmp_path):
    database = tmp_path / "history.sqlite"
    with PlayerHistoryStore(str(database)) as store:
        game_id, inserted = store.ingest(
            _game(), source="lichess", source_game_id="abc123",
            source_url="https://lichess.org/abc123",
        )
        assert inserted
        assert store.ingest(
            _game(), source="LICHESS", source_game_id="abc123",
            source_url="https://lichess.org/abc123",
        ) == (game_id, False)
        with pytest.raises(HistoryDataError, match="Source game changed"):
            store.ingest(
                _game(site="changed"), source="lichess",
                source_game_id="abc123",
            )


def test_schema_v1_database_migrates_without_data_loss(tmp_path):
    database = tmp_path / "history.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES ('schema_version', '1')")
    connection.commit()
    connection.close()

    with PlayerHistoryStore(str(database)) as store:
        assert store.info()["schema_version"] == "2"
        assert store.create_person("Alice").startswith("person_")


def test_lichess_sync_ingests_games_and_advances_cursor(tmp_path):
    database = tmp_path / "history.sqlite"
    pgn = (
        '[Event "Rated Rapid game"]\n[Site "https://lichess.org/abcd1234"]\n'
        '[UTCDate "2026.09.15"]\n[UTCTime "12:34:56"]\n'
        '[White "AliceOnline"]\n[Black "Bob"]\n[Result "1-0"]\n\n'
        '1. e4 e5 2. Nf3 Nc6 1-0\n'
    ).encode()
    requests = []

    def fake_get(url, headers, timeout):
        requests.append((url, headers, timeout))
        return HttpResponse(200, pgn, {})

    with PlayerHistoryStore(str(database)) as store:
        person_id = store.create_person("Alice")
        account_id = store.add_external_account(
            person_id, "lichess", "aliceonline", username="AliceOnline")
        first = sync_external_account(store, account_id, http_get=fake_get)
        second = sync_external_account(store, account_id, http_get=fake_get)
        assert first.games_new == 1
        assert second.games_duplicate == 1
        assert len(list(store.person_games(person_id))) == 1
        assert store.sync_state(account_id)["cursor"]["since_ms"] > 0
    assert "since=" in requests[1][0]


def test_chesscom_sync_uses_archive_etag_and_skips_unchanged_month(tmp_path):
    database = tmp_path / "history.sqlite"
    archive_url = "https://api.chess.com/pub/player/alice/games/2026/09"
    source_game = {
        "url": "https://www.chess.com/game/live/123",
        "pgn": str(_game(white="AliceOnline")),
        "end_time": 1_789_000_000,
        "rated": True,
        "rules": "chess",
        "time_class": "rapid",
        "time_control": "600+5",
    }
    month_requests = []

    def fake_get(url, headers, _timeout):
        if url.endswith("/games/archives"):
            return HttpResponse(
                200, json_bytes({"archives": [archive_url]}), {})
        month_requests.append(dict(headers))
        if headers.get("If-None-Match") == '"month-v1"':
            return HttpResponse(304, b"", {"etag": '"month-v1"'})
        return HttpResponse(
            200, json_bytes({"games": [source_game]}),
            {"etag": '"month-v1"'},
        )

    with PlayerHistoryStore(str(database)) as store:
        person_id = store.create_person("Alice")
        account_id = store.add_external_account(
            person_id, "chess.com", "42", username="AliceOnline")
        first = sync_external_account(store, account_id, http_get=fake_get)
        second = sync_external_account(store, account_id, http_get=fake_get)
        assert first.games_new == 1
        assert second.games_seen == 0
        assert len(list(store.person_games(person_id))) == 1
    assert month_requests[1]["If-None-Match"] == '"month-v1"'


def json_bytes(value):
    import json

    return json.dumps(value).encode("utf-8")


def test_corrupt_cached_analysis_fails_closed(tmp_path):
    database = tmp_path / "history.sqlite"
    with PlayerHistoryStore(str(database)) as store:
        game_id, _ = store.ingest(_game())
        profile_id = store.register_profile({"engine": {"depth": 18}})
        store.save_analysis(game_id, profile_id, _analysis())

    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE game_analyses SET analysis=? WHERE game_id=?",
        (b"not-a-valid-cache", game_id),
    )
    connection.commit()
    connection.close()

    with PlayerHistoryStore(str(database)) as store:
        with pytest.raises(HistoryDataError, match=game_id):
            store.load_player_analyses("Alice", profile_id)


def test_history_cli_reports_invalid_database_without_traceback(tmp_path, capsys):
    from chess_review.cli import main

    database = tmp_path / "invalid.sqlite"
    database.write_bytes(b"not sqlite")
    assert main(["history", "info", "--db", str(database)]) == 2
    assert "history failed:" in capsys.readouterr().err