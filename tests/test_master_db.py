"""Tests for the local master opening database."""
from __future__ import annotations

import zipfile

import chess

from chess_review.master_db import MasterOpeningDatabase, build_master_database, position_key
from chess_review.models import GameAnalysis, MoveAnalysis
from chess_review.render import build_opening_section, render_game_markdown, build_game_view


PGN = """[Event "Elite One"]
[Date "2025.01.02"]
[White "A"]
[Black "B"]
[WhiteElo "2550"]
[BlackElo "2450"]
[Result "1-0"]

1. Nf3 d5 2. g3 Nf6 3. Bg2 1-0

[Event "Elite Two"]
[Date "2024.03.04"]
[White "C"]
[Black "D"]
[WhiteElo "2500"]
[BlackElo "2400"]
[Result "1/2-1/2"]

1. g3 d5 2. Nf3 Nf6 3. Bg2 1/2-1/2

[Event "Below Filter"]
[Date "2023.01.01"]
[White "E"]
[Black "F"]
[WhiteElo "2100"]
[BlackElo "2050"]
[Result "0-1"]

1. e4 e5 0-1
"""


def _transposed_board(first: str) -> chess.Board:
    board = chess.Board()
    sequence = ["Nf3", "d5", "g3"] if first == "knight" else ["g3", "d5", "Nf3"]
    for san in sequence:
        board.push_san(san)
    return board


def test_position_key_merges_transpositions_and_ignores_clocks():
    first = _transposed_board("knight")
    second = _transposed_board("pawn")
    assert first.fen() != second.fen()
    assert position_key(first) == position_key(second)


def test_build_and_query_aggregates_transposed_games(tmp_path):
    source = tmp_path / "elite.pgn"
    source.write_text(PGN, encoding="utf-8")
    output = tmp_path / "master.sqlite"
    stats = build_master_database(
        [str(source), str(source)], str(output), max_ply=6, min_rating=2300)
    assert stats.games_seen == 6
    assert stats.games_indexed == 2
    assert stats.duplicate_games == 2
    assert stats.skipped_rating == 2

    with MasterOpeningDatabase(str(output)) as database:
        result = database.lookup(_transposed_board("knight"))
        assert result is not None
        assert result.total_games == 2
        assert result.moves[0].san == "Nf6"
        assert result.moves[0].games == 2
        assert result.moves[0].white_wins == 1
        assert result.moves[0].draws == 1
        assert result.moves[0].play_rate == 100.0
        assert result.moves[0].avg_rating == 2475
        assert result.moves[0].first_year == 2024
        assert result.moves[0].last_year == 2025
        assert database.metadata()["max_ply"] == 6


def test_builder_streams_elite_style_zip(tmp_path):
    archive_path = tmp_path / "lichess_elite.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("lichess_elite.pgn", PGN)
    output = tmp_path / "master.sqlite"
    stats = build_master_database([str(archive_path)], str(output), min_rating=2300)
    assert stats.games_indexed == 2
    with MasterOpeningDatabase(str(output)) as database:
        start = database.lookup(chess.Board(), top=3)
        assert start is not None
        assert start.total_games == 2
        assert {move.san for move in start.moves} == {"Nf3", "g3"}


def test_identical_moves_by_different_players_are_distinct_games(tmp_path):
    source = tmp_path / "same-line.pgn"
    source.write_text("""[Event "One"]
[Site "A"]
[Date "2025.01.01"]
[Round "1"]
[White "Player One"]
[Black "Player Two"]
[WhiteElo "2500"]
[BlackElo "2450"]
[Result "1-0"]

1. e4 e5 1-0

[Event "Two"]
[Site "B"]
[Date "2025.01.02"]
[Round "2"]
[White "Player Three"]
[Black "Player Four"]
[WhiteElo "2550"]
[BlackElo "2400"]
[Result "0-1"]

1. e4 e5 0-1
""", encoding="utf-8")
    output = tmp_path / "master.sqlite"
    stats = build_master_database([str(source)], str(output))
    assert stats.games_indexed == 2
    assert stats.duplicate_games == 0
    with MasterOpeningDatabase(str(output)) as database:
        result = database.lookup(chess.Board())
        assert result is not None
        assert result.total_games == 2


def test_opening_report_keeps_master_popularity_separate_from_engine_choice():
    move = MoveAnalysis(
        ply=1, move_number=1, color=chess.WHITE, san="e4", uci="e2e4",
        fen_before=chess.STARTING_FEN,
        fen_after="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        eval_before_mover=20, eval_after_mover=15,
        eval_before_white=20, eval_after_white=15, cp_loss=5,
        best_move_uci="d2d4", best_move_san="d4",
        phase="opening", classification="best",
        best_is_capture=False, best_is_check=False, played_is_capture=False,
        in_book=True,
        master_context={
            "position_key": "abc", "total_games": 120,
            "source": "local-sqlite", "database_version": "2026-09-12",
            "moves": [
                {"uci": "e2e4", "san": "e4", "games": 72,
                 "white_score_pct": 55.0, "play_rate": 60.0},
                {"uci": "d2d4", "san": "d4", "games": 48,
                 "white_score_pct": 53.0, "play_rate": 40.0},
            ],
        },
    )
    game = GameAnalysis(
        white="A", black="B", result="*", date="2025.01.01",
        event="Test", site="", eco="A00", opening_name="Test Opening",
        moves=[move], final_eval_white=15,
    )
    section = build_opening_section(game)
    evidence = section["master_evidence"]
    assert evidence is not None and evidence["low_sample"] is True
    assert evidence["played_rate"] == 60.0
    assert evidence["master_top"] == "e4"
    assert evidence["engine_best"] == "d4"
    assert evidence["agrees_with_engine"] is False

    markdown = render_game_markdown(build_game_view(game, with_svg=False))
    assert "大师实战验证（样本较少，仅供参考）" in markdown
    assert "热门程度只代表实战经验，不替代引擎评价" in markdown