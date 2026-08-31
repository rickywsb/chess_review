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
    assert book.available  # bundled data/komodo.bin
    moves = book.lookup(chess.STARTING_FEN, top=3)
    assert moves, "start position should be in book"
    assert all({"san", "weight", "pct"} <= set(m) for m in moves)
    assert moves[0]["san"] in {"e4", "d4", "c4", "Nf3", "g3", "b3"}
    # An out-of-book / bogus position yields no moves, never raises.
    assert book.lookup("8/8/8/8/8/8/8/K6k w - - 0 1") == []


def _mk_move(before, after, cp_loss, mate_before=None, mate_after=None):
    """Minimal MoveAnalysis for exercising the selection layer."""
    return MoveAnalysis(
        ply=1, move_number=1, color=chess.WHITE, san="Xx", uci="a1a2",
        fen_before="", fen_after="",
        eval_before_mover=before, eval_after_mover=after,
        eval_before_white=before, eval_after_white=after, cp_loss=cp_loss,
        best_move_uci="b1b2", best_move_san="Yy",
        phase="middlegame", classification="mistake",
        best_is_capture=False, best_is_check=False, played_is_capture=False,
        in_book=False, mate_before=mate_before, mate_after=mate_after,
    )


def test_significance_keeps_outcome_changing_moves():
    from chess_review.render import _significance
    # 均势 -> 劣势: threw the game into trouble.
    assert _significance(_mk_move(-34, -176, 142), 100) == (True, "threw_game")
    # 优势 -> 均势: let the win slip.
    assert _significance(_mk_move(128, 5, 123), 100) == (True, "lost_win")
    # Had a forced mate and let the decisive edge go with it: missed mate.
    keep, tag = _significance(_mk_move(900, 150, 750, mate_before=4), 100)
    assert keep and tag == "missed_mate"
    # Huge blunder that still leaves a winning game is worth a note.
    assert _significance(_mk_move(1121, 228, 893), 100) == (True, "big_error")


def test_significance_drops_noise_while_crushing():
    from chess_review.render import _significance
    # +9.45 -> +7.28: both completely winning, don't nitpick.
    assert _significance(_mk_move(945, 728, 217), 100) == (False, "still_winning")
    # +8.34 -> +6.31: a 200cp "blunder" that changes nothing.
    assert _significance(_mk_move(834, 631, 203), 100) == (False, "still_winning")
    # Already lost, staying lost: not actionable.
    assert _significance(_mk_move(-403, -554, 151), 100) == (False, "already_lost")
    # Small slip inside the same balanced zone.
    assert _significance(_mk_move(-28, -68, 40), 100)[0] is False


def test_classify_delta_frames_honestly():
    from chess_review.classify import classify_delta
    # Threw a forced mate but is still crushing: category threw_mate, framed as
    # still winning — NOT "变差".
    cat, state = classify_delta(30000, 234, 18, None, material_swing=0, cp_loss=2766)
    assert cat == "threw_mate"
    assert state == "仍占优势"
    # The refutation line actually wins material off the mover.
    cat, state = classify_delta(-34, -176, None, None, material_swing=-3, cp_loss=142)
    assert cat == "lost_material"
    # Had a real edge, now only equal, no material lost: the win slipped away.
    cat, state = classify_delta(128, 5, None, None, material_swing=0, cp_loss=123)
    assert cat == "lost_the_win"
    assert state == "已回到均势"
    # Small drop, no material, no mate: a quiet positional slip we admit to.
    cat, _ = classify_delta(20, -40, None, None, material_swing=0, cp_loss=60)
    assert cat == "positional_slip"
    # Threw a mate and is no longer winning -> it cost the game.
    cat, _ = classify_delta(30000, -50, 5, None, material_swing=0, cp_loss=3050)
    assert cat == "lost_the_win"


def test_material_swing_counts_whole_board_not_mid_exchange():
    from chess_review.render import _line_material_swing
    # White trades rooks: Rxd8+ Kxd8. Measured against the whole board BEFORE the
    # move, an even recapture nets 0 — not -5 as a mid-exchange snapshot would say.
    fen = "3rk3/8/8/8/8/8/8/3RK3 w - - 0 1"
    assert _line_material_swing(fen, "d1d8", ["Kxd8"], chess.WHITE) == 0
    # A genuine hang still registers: White plays a quiet move, Black wins the rook.
    fen2 = "3rk3/8/8/8/8/8/6P1/3R1K2 w - - 0 1"
    assert _line_material_swing(fen2, "g2g3", ["Rxd1+"], chess.WHITE) == -5


def test_positional_diff_names_king_airiness():
    from chess_review.render import _positional_diff
    # Same position except White's g-pawn is shoved to g4, thinning the king's
    # shelter — the diff should name king airiness, not stay silent.
    best = chess.Board("6k1/8/8/8/8/8/5PPP/6K1 b - - 0 1")
    played = chess.Board("6k1/8/8/8/6P1/8/5P1P/6K1 b - - 0 1")
    msg = _positional_diff(best, played, chess.WHITE)
    assert msg is not None and "王" in msg
    # No structural change -> no fabricated feature.
    assert _positional_diff(best, best, chess.WHITE) is None


def test_choice_fact_distinguishes_only_move_and_many_options():
    from chess_review.render import _choice_fact
    m = _mk_move(200, 50, cp_loss=150)
    m.best_move_san = "Nf3"
    # A dominant best move with a big gap to the second choice -> only-move.
    m.alt_gap_cp = 260
    m.alt_count = 1
    assert "唯一解" in (_choice_fact(m) or "")
    # Several near-equal options -> the slip was avoidable.
    m.alt_gap_cp = 15
    m.alt_count = 4
    assert "都不错" in (_choice_fact(m) or "")
    # Small gap, only two close moves -> inconclusive, say nothing.
    m.alt_gap_cp = 20
    m.alt_count = 2
    assert _choice_fact(m) is None


def test_best_line_gain_confirms_won_material():
    from chess_review.render import _best_line_material_gain
    fen = "3r4/6k1/8/8/8/8/8/3R1K2 w - - 0 1"
    assert _best_line_material_gain(fen, ["Rxd8"], chess.WHITE) == 5
    # An even trade nets zero, so no 'won material' claim is made.
    fen2 = "3rk3/8/8/8/8/8/8/3RK3 w - - 0 1"
    assert _best_line_material_gain(fen2, ["Rxd8+", "Kxd8"], chess.WHITE) == 0


def test_judge_block_formats_diagnosis():
    from chess_review.coach_llm import _judge_block, _TWO_PASS
    assert _TWO_PASS is True  # two-pass on by default
    block = _judge_block({"primary": "王翼漏风",
                          "use_facts": ["【位置】X", "【选择】Y"],
                          "honest_state": "已处于下风", "avoid": "别夸大"})
    assert "核心主题：王翼漏风" in block
    assert "【位置】X" in block and "【选择】Y" in block
    assert "已处于下风" in block
    # No selected facts -> tell the writer to follow the line, don't invent.
    assert "顺着变化" in _judge_block({"primary": "x"})


def test_rate_limiter_blocks_after_max():
    from chess_review.webapp import _RateLimiter
    lim = _RateLimiter(max_hits=2, window_s=100)
    assert lim.allow("1.2.3.4")[0] is True
    assert lim.allow("1.2.3.4")[0] is True
    allowed, retry = lim.allow("1.2.3.4")
    assert allowed is False and retry >= 1
    # A different client is unaffected.
    assert lim.allow("5.6.7.8")[0] is True
    # max_hits <= 0 disables limiting entirely.
    off = _RateLimiter(max_hits=0, window_s=100)
    assert all(off.allow("x")[0] for _ in range(50))





