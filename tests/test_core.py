"""Unit tests that do not require an engine binary."""
import io

import chess
import chess.pgn

from chess_review.classify import classify_loss, classify_phase, non_pawn_material
from chess_review.models import GameAnalysis, MoveAnalysis


def test_classify_loss_thresholds():
    assert classify_loss(0) == "best"
    assert classify_loss(10) == "best"
    assert classify_loss(60) == "inaccuracy"
    assert classify_loss(120) == "mistake"
    assert classify_loss(250) == "blunder"


def test_phase_opening_by_move_number():
    board = chess.Board()
    assert classify_phase(board) == "opening"


def test_phase_endgame_by_material():
    # King + rook vs king + rook, well past move 15.
    board = chess.Board("8/8/4k3/8/8/4K3/8/R6r w - - 0 40")
    assert non_pawn_material(board) == 10
    assert classify_phase(board) == "endgame"


def test_phase_middlegame():
    board = chess.Board()
    # Advance the full-move counter past the opening without trading material.
    board.set_fen("r1bqkb1r/pppppppp/2n2n2/8/8/2N2N2/PPPPPPPP/R1BQKB1R w KQkq - 0 20")
    assert classify_phase(board) == "middlegame"


def test_game_analysis_helpers():
    ga = GameAnalysis(
        white="Alice", black="Bob", result="1-0", date="2025.01.01",
        event="Test", site="", eco="", opening_name="",
    )
    assert ga.player_color("alice") == chess.WHITE
    assert ga.player_color("bob") == chess.BLACK
    assert ga.player_color("carol") is None
    assert ga.result_score(chess.WHITE) == 1.0
    assert ga.result_score(chess.BLACK) == 0.0


def test_pgn_reads():
    pgn = "[White \"A\"]\n[Black \"B\"]\n[Result \"1-0\"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    assert len(list(game.mainline_moves())) == 7
