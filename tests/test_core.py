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


def test_opened_line_fact_names_the_file_onto_the_king():
    from chess_review.render import _opened_line_fact
    # Black king on e8; White pawn e5 blocks the e-file from the Re1. Black plays
    # a quiet ...a6 and White refutes with exd6, vacating e5 -> the e-file opens
    # straight onto the Black king. The real cost is the line, not the knight.
    fen = "4k3/p7/3n4/4P3/8/8/8/4R1K1 b - - 0 1"
    fact = _opened_line_fact(fen, "a7a6", ["exd6"], chess.BLACK)
    assert fact is not None
    assert "e 线" in fact and "王" in fact
    # The exposure often peaks mid-line: even if the king is chased off the file
    # by the end of the sequence, scanning every position still catches it.
    chased = _opened_line_fact(fen, "a7a6", ["exd6", "Kd7"], chess.BLACK)
    assert chased is not None and "e 线" in chased
    # A file that was already open is not credited to this move.
    quiet = "4k3/pppp1ppp/8/8/8/8/PPPP1PPP/4K3 w - - 0 1"
    assert _opened_line_fact(quiet, "a2a3", ["a7a6"], chess.WHITE) is None


def test_mate_threat_fact_surfaces_forced_mate():
    from chess_review.render import _mate_threat_fact
    # The played move leaves the mover getting mated: engine reports mate_after<0.
    m = _mk_move(120, -29000, cp_loss=29000, mate_after=-3)
    m.refutation_line_san = ["Qh4+", "g3", "Qxg3#"]
    fact = _mate_threat_fact(m)
    assert fact is not None and "杀" in fact and "3 步" in fact
    # No mate anywhere -> no mate-threat claim invented.
    assert _mate_threat_fact(_mk_move(120, -80, cp_loss=200)) is None
    # Already being mated at least as fast before the move -> not caused here.
    already = _mk_move(-29000, -29000, cp_loss=0, mate_before=-2, mate_after=-3)
    assert _mate_threat_fact(already) is None
    # A refutation line that itself ends in checkmate also triggers it.
    vialine = _mk_move(120, -400, cp_loss=520)
    vialine.refutation_line_san = ["Qe7#"]
    assert _mate_threat_fact(vialine) is not None


def test_refutation_fork_fact_names_the_double_attack():
    from chess_review.render import _refutation_fork_fact
    # After the blunder it is Black to move; ...Nc2+ forks the White king (e1)
    # and rook (a1). This is the real reason, not a hanging pawn.
    fen_after = "4k3/8/8/8/8/n7/8/R3K3 b - - 0 1"
    fact = _refutation_fork_fact(fen_after, ["Nc2+"], chess.WHITE)
    assert fact is not None and "叉子" in fact and "王" in fact and "车" in fact
    # A quiet reply that attacks nothing valuable -> no fork claim.
    quiet = "4k3/8/8/8/8/8/P7/4K3 b - - 0 1"
    assert _refutation_fork_fact(quiet, ["Kd7"], chess.WHITE) is None


def test_pin_fact_detects_absolute_and_relative_pins():
    from chess_review.render import _pin_fact
    # Absolute pin: the White knight on e2 is pinned to its king (e1) by the rook
    # on e8 and cannot move.
    absolute = "4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1"
    fact = _pin_fact(absolute, chess.WHITE)
    assert fact is not None and "牵制" in fact and "王" in fact
    # Relative pin: the knight on c4 is pinned by the bishop on a6 to the more
    # valuable rook on e2 behind it.
    relative = "6k1/8/b7/8/2N5/8/4R3/6K1 w - - 0 1"
    fact = _pin_fact(relative, chess.WHITE)
    assert fact is not None and "牵制" in fact and "车" in fact
    # No pin present -> nothing fabricated.
    assert _pin_fact("6k1/8/8/8/8/8/8/6K1 w - - 0 1", chess.WHITE) is None


def test_skewer_fact_wins_the_piece_behind():
    from chess_review.render import _skewer_fact
    # White rook on e5 (front) is attacked by the Black rook on e8; the White
    # bishop on e3 stands directly behind it — a skewer: when the rook moves the
    # bishop falls.
    fen = "4r1k1/8/8/4R3/8/4B3/8/6K1 b - - 0 1"
    fact = _skewer_fact(fen, chess.WHITE)
    assert fact is not None and "串" in fact and "象" in fact
    # Nothing lined up -> no skewer invented.
    assert _skewer_fact("6k1/8/8/8/8/8/8/6K1 b - - 0 1", chess.WHITE) is None


def test_discovered_attack_fact_reads_the_unveiled_line():
    from chess_review.render import _discovered_attack_fact
    # Black's ...Bd6 steps the bishop off the e-file, unveiling the Black rook on
    # e8 which now attacks the White queen on e2 — a discovered attack.
    fen_after = "4r1k1/8/8/4b3/8/8/4Q3/6K1 b - - 0 1"
    fact = _discovered_attack_fact(fen_after, ["Bd6"], chess.WHITE)
    assert fact is not None and "闪击" in fact and "后" in fact
    # A quiet king step reveals nothing -> no discovered attack.
    assert _discovered_attack_fact(fen_after, ["Kf8"], chess.WHITE) is None


def test_passed_pawn_fact_flags_an_allowed_passer():
    from chess_review.render import _passed_pawn_fact
    # After the best move a White pawn on e2 still holds back Black's d-pawn; the
    # played move let that pawn advance to e4, so Black's d4-pawn becomes passed.
    best = chess.Board("k7/8/8/8/3p4/8/4P3/K7 b - - 0 1")
    played = chess.Board("k7/8/8/8/3pP3/8/8/K7 b - - 0 1")
    fact = _passed_pawn_fact(best, played, chess.WHITE, endgame=True)
    assert fact is not None and "通路兵" in fact and "d4" in fact
    # Outside the endgame we stay quiet about passers.
    assert _passed_pawn_fact(best, played, chess.WHITE, endgame=False) is None


def test_weak_square_fact_names_a_new_hole():
    from chess_review.render import _weak_square_fact
    # With the pawn on e4 it still guards d5; pushing it to e5 leaves d5 a
    # permanent hole that Black's c6-pawn controls.
    best = chess.Board("6k1/8/2p5/8/4P3/8/8/6K1 w - - 0 1")
    played = chess.Board("6k1/8/2p5/4P3/8/8/8/6K1 w - - 0 1")
    fact = _weak_square_fact(best, played, chess.WHITE)
    assert fact is not None and "弱格" in fact and "d5" in fact
    # No new hole -> nothing to report.
    assert _weak_square_fact(best, best, chess.WHITE) is None


def test_outpost_fact_flags_a_planted_knight():
    from chess_review.render import _outpost_fact
    # The best move keeps Black's knight on f6; the played move let it hop to d5,
    # a hole supported by the c6-pawn — a classic outpost against White.
    best = chess.Board("6k1/8/2p2n2/8/8/8/8/6K1 w - - 0 1")
    played = chess.Board("6k1/8/2p5/3n4/8/8/8/6K1 w - - 0 1")
    fact = _outpost_fact(best, played, chess.WHITE)
    assert fact is not None and "前哨" in fact and "d5" in fact
    assert _outpost_fact(best, best, chess.WHITE) is None


def test_zugzwang_fact_only_in_a_quiet_pawn_endgame():
    from chess_review.render import _zugzwang_fact
    # Pure K+P endgame, White to move with only quiet king moves; the top move is
    # near-best yet still loses ground -> zugzwang.
    m = _mk_move(120, 20, cp_loss=30)
    m.fen_before = "4k3/8/4K3/4P3/8/8/8/8 w - - 0 1"
    assert "逼移" in (_zugzwang_fact(m) or "")
    # A piece on the board -> not a pawn endgame, so we do not claim zugzwang.
    m2 = _mk_move(120, 20, cp_loss=30)
    m2.fen_before = "4k3/8/4K3/4R3/8/8/8/8 w - - 0 1"
    assert _zugzwang_fact(m2) is None
    # A big single-move loss is an ordinary blunder, not zugzwang.
    m3 = _mk_move(120, -200, cp_loss=320)
    m3.fen_before = "4k3/8/4K3/4P3/8/8/8/8 w - - 0 1"
    assert _zugzwang_fact(m3) is None


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





