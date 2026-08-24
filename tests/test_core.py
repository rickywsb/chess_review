"""Unit tests that do not require an engine binary."""
import io

import chess
import chess.pgn

from chess_review.classify import classify_loss, classify_phase, non_pawn_material, has_queens
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


def test_phase_endgame_by_no_queens():
    # Queens already off the board early -> endgame regardless of move number.
    board = chess.Board("r1b1kb1r/pppp1ppp/2n2n2/8/8/2N2N2/PPPP1PPP/R1B1KB1R w KQkq - 0 10")
    assert not has_queens(board)
    assert classify_phase(board) == "endgame"


def test_phase_opening_only_when_in_book():
    # Move <=15 with a loaded book: opening only while still following theory.
    board = chess.Board()
    board.set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    assert classify_phase(board, in_book=True, book_loaded=True) == "opening"
    # Same position but already out of book -> middlegame (queens still on board).
    assert classify_phase(board, in_book=False, book_loaded=True) == "middlegame"


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


def test_polyglot_book_lookup():
    from chess_review.polyglot_book import PolyglotBook
    book = PolyglotBook()
    assert book.available  # bundled data/performance.bin
    moves = book.lookup(chess.STARTING_FEN, top=3)
    assert moves, "start position should be in book"
    assert all({"san", "weight", "pct"} <= set(m) for m in moves)
    assert moves[0]["san"] in {"e4", "d4", "c4", "Nf3", "g3", "b3"}
    # An out-of-book / bogus position yields no moves, never raises.
    assert book.lookup("8/8/8/8/8/8/8/K6k w - - 0 1") == []
