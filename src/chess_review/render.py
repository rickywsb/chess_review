"""Render analyses into Markdown and HTML reports (with board diagrams)."""
from __future__ import annotations

import os
from typing import Optional

import chess
import chess.svg
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import coach_llm
from .classify import (
    BLUNDER, MISTAKE, TAG_ZH, CATEGORY_ZH, CATEGORY_FRAME,
    classify_delta, outcome_zone, significance, state_word,
)
from .metrics import PHASES
from .models import GameAnalysis, MoveAnalysis
from .polyglot_book import get_default_book
from .summary import build_summary_zh

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

# Eval at/below which we consider a side "lost control" for turning-point calc.
TURNING_THRESHOLD = -150
RECOVERY_THRESHOLD = -100

PHASE_ZH = {"opening": "开局", "middlegame": "中局", "endgame": "残局"}
CLASS_ZH = {"blunder": "漏着", "mistake": "错着", "inaccuracy": "不精确",
            "good": "尚可", "best": "最佳"}


def _state_zh(cp: int) -> str:
    if cp >= 200:
        return "胜势"
    if cp >= 70:
        return "略优"
    if cp > -70:
        return "均势"
    if cp > -200:
        return "略差"
    return "劣势"


# Selection layer lives in classify.py so the summary shares one source of truth.
_significance = significance
_outcome_zone = outcome_zone


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def format_eval(cp: int, mate: Optional[int] = None) -> str:
    if mate is not None:
        return f"#{'+' if mate > 0 else '-'}{abs(mate)}"
    pawns = cp / 100.0
    return f"{pawns:+.2f}"


def lichess_url(fen: str) -> str:
    return "https://lichess.org/analysis/" + fen.replace(" ", "_")


def _pov(eval_white: int, color: bool) -> int:
    return eval_white if color == chess.WHITE else -eval_white


def _side_name(ga: GameAnalysis, color: bool) -> str:
    return ga.white if color == chess.WHITE else ga.black


def _comment_for(m: MoveAnalysis) -> str:
    if m.is_forcing_miss:
        kind = "将军" if m.best_is_check else "吃子"
        return f"漏掉强制{kind}：{m.best_move_san}。"
    return f"需要一步安静的着法（计划／预防／王的安全）：{m.best_move_san}。"


def _explain_move_zh(m: MoveAnalysis) -> dict:
    """Build a detailed Chinese explanation for one problematic move."""
    before = m.eval_before_mover
    after = m.eval_after_mover
    s_before = _state_zh(before)
    s_after = _state_zh(after)
    line = " ".join(m.best_line_san) if m.best_line_san else m.best_move_san

    # 为什么错
    if m.is_forcing_miss and m.best_is_check:
        why = (f"这里有一步强制的将军 {m.best_move_san}，能先手逼迫对手、抢占主动；"
               f"你走的 {m.san} 放过了这个机会。")
    elif m.is_forcing_miss and m.best_is_capture:
        why = (f"这里有一步直接的吃子 {m.best_move_san}，能占到实惠（子力或位置）；"
               f"你走的 {m.san} 没有抓住它。")
    else:
        why = (f"这是一步判断／计划上的失误：最佳是安静的 {m.best_move_san}"
               f"（改善子力、预防对手的计划或照顾王的安全），而 {m.san} 忽视了局面真正的需要。")

    # 造成了什么后果
    delta = f"评估从 {format_eval(before, m.mate_before)}（{s_before}）" \
            f"降到 {format_eval(after, m.mate_after)}（{s_after}），单步损失约 {m.cp_loss}cp。"
    if before >= 70 and after < -70:
        extra = "局面由占优直接滑向被动。"
    elif before > -70 and after <= -200:
        extra = "从大致均势一步落入明显劣势。"
    elif before >= 200 and after < 200:
        extra = "本可到手的胜势被放走了。"
    elif after <= -200:
        extra = "局面已经明显吃亏，翻盘将变得困难。"
    else:
        extra = "优势／均势被削弱，给了对手喘息的机会。"
    consequence = delta + extra

    # 应该做什么
    idea = f"正解主线：{line}。" if m.best_line_san else f"应走 {m.best_move_san}。"
    if m.is_forcing_miss:
        habit = "养成每步先把所有吃子和将军算一遍、再考虑其它着法的习惯。"
    else:
        habit = "先问三件事：对手想干什么？我最差的子是哪个？这一步在改善什么？"
    what_to_do = f"应走 {m.best_move_san}。{idea}{habit}"

    return {"why": why, "consequence": consequence, "what_to_do": what_to_do}


def _king_flight_squares(board: chess.Board, king_color: bool) -> int:
    """Rough count of ``king_color``'s king escape squares: adjacent squares not
    occupied by a friendly piece and not attacked by the opponent."""
    ksq = board.king(king_color)
    if ksq is None:
        return 0
    other = not king_color
    count = 0
    for sq in chess.SquareSet(chess.BB_KING_ATTACKS[ksq]):
        piece = board.piece_at(sq)
        if piece is not None and piece.color == king_color:
            continue
        if board.is_attacked_by(other, sq):
            continue
        count += 1
    return count


# Piece values / names used to describe material facts to the LLM.
_PIECE_VAL = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}
_PIECE_ZH = {
    chess.PAWN: "兵", chess.KNIGHT: "马", chess.BISHOP: "象",
    chess.ROOK: "车", chess.QUEEN: "后", chess.KING: "王",
}


def _material(board: chess.Board, color: bool) -> int:
    return sum(_PIECE_VAL[p.piece_type]
               for p in board.piece_map().values() if p.color == color)


def _pts_zh(pts: int) -> str:
    """Approximate a material-point swing as a piece in words."""
    v = abs(pts)
    if v >= 9:
        return "一个后左右的子力"
    if v >= 5:
        return "一个车左右的子力"
    if v >= 3:
        return "一个轻子（马/象）左右"
    if v >= 1:
        return f"约 {v} 个兵的子力"
    return "少量子力"


def _loose_pieces(board: chess.Board, color: bool) -> list[tuple[int, chess.Piece]]:
    """``color``'s pieces that are attacked and either undefended or attacked by
    a cheaper piece. Kept as a helper but no longer used for narration facts —
    consequence claims are now read off the engine's actual refutation line so
    we never assert a 'hanging piece' the line does not truly win."""
    opp = not color
    out: list[tuple[int, chess.Piece]] = []
    for sq, p in board.piece_map().items():
        if p.color != color or p.piece_type == chess.KING:
            continue
        attackers = board.attackers(opp, sq)
        if not attackers:
            continue
        defenders = board.attackers(color, sq)
        min_atk = min(_PIECE_VAL[board.piece_at(a).piece_type] for a in attackers)
        if not defenders or min_atk < _PIECE_VAL[p.piece_type]:
            out.append((sq, p))
    return out


def _fork_targets(board_after: chess.Board, to_square: int) -> Optional[str]:
    """If the piece now on ``to_square`` attacks 2+ valuable enemy pieces
    (king or >= minor), return their names — a fork / double attack."""
    p = board_after.piece_at(to_square)
    if p is None:
        return None
    names: list[str] = []
    for sq in board_after.attacks(to_square):
        q = board_after.piece_at(sq)
        if q is not None and q.color != p.color and (
                q.piece_type == chess.KING or _PIECE_VAL[q.piece_type] >= 3):
            names.append(_PIECE_ZH[q.piece_type])
    if len(names) >= 2:
        return "、".join(dict.fromkeys(names))
    return None


def _line_material_swing(fen_before: str, played_uci: str,
                         san_line: list[str], mover: bool) -> int:
    """Net material change for ``mover`` caused by the played move *and* the
    opponent's best reply, measured against the WHOLE board before the move.

    Starting from ``fen_before`` (not the mid-exchange position after the move)
    is what makes recaptures and even trades net to ~0: if the mover captures a
    piece and the opponent captures back, the balance returns to where it
    started, so we no longer mis-report an exchange as '丢子'. A negative result
    means the mover genuinely ends down material relative to before the move."""
    try:
        b = chess.Board(fen_before)
    except (ValueError, TypeError):
        return 0
    opp = not mover
    base = _material(b, mover) - _material(b, opp)
    try:
        b.push(chess.Move.from_uci(played_uci))
    except (ValueError, TypeError, AssertionError):
        return 0
    for san in san_line:
        try:
            b.push(b.parse_san(san))
        except ValueError:
            break
    end = _material(b, mover) - _material(b, opp)
    return end - base


def _best_line_material_gain(fen_before: str, best_line_san: list[str],
                            mover: bool) -> int:
    """Net material the mover ends up ahead by over the engine's best line (which
    starts with the best move). Positive => the best move's tactics genuinely win
    material; ~0 => the point is positional / initiative, not a won piece. Used to
    validate tactical claims against the actual line instead of 1-ply geometry."""
    if not best_line_san:
        return 0
    try:
        b = chess.Board(fen_before)
    except (ValueError, TypeError):
        return 0
    opp = not mover
    base = _material(b, mover) - _material(b, opp)
    for san in best_line_san:
        try:
            b.push(b.parse_san(san))
        except ValueError:
            break
    return (_material(b, mover) - _material(b, opp)) - base


def _pivot_capture(fen_before: str, played_uci: str, san_line: list[str],
                   mover: bool) -> Optional[tuple[str, str]]:
    """The opponent capture that drives ``mover``'s material to its lowest point
    *below where it started* (before the played move). Measuring from
    ``fen_before`` means an even recapture never registers — only a genuine net
    loss produces a pivot. Returns ``(move_san, captured_piece_zh)`` or ``None``."""
    try:
        b = chess.Board(fen_before)
    except (ValueError, TypeError):
        return None
    opp = not mover
    base = _material(b, mover) - _material(b, opp)
    try:
        b.push(chess.Move.from_uci(played_uci))
    except (ValueError, TypeError, AssertionError):
        return None
    low = base
    result: Optional[tuple[str, str]] = None
    for san in san_line:
        try:
            mv = b.parse_san(san)
        except ValueError:
            break
        side = b.turn
        captured = None
        if b.is_capture(mv):
            victim = b.piece_at(mv.to_square)
            captured = _PIECE_ZH.get(victim.piece_type) if victim else None
        b.push(mv)
        rel = _material(b, mover) - _material(b, opp)
        if side == opp and rel < low and captured:
            low = rel
            result = (san, captured)
    return result


# Direction vectors (file-delta, rank-delta) for the two slider families.
_ROOK_DIRS = ((0, 1, "file"), (0, -1, "file"), (1, 0, "rank"), (-1, 0, "rank"))
_BISHOP_DIRS = ((1, 1, "diag"), (1, -1, "diag"), (-1, 1, "diag"), (-1, -1, "diag"))


def _slider_pressure(board: chess.Board, target_sq: int, attacker_color: bool):
    """Enemy sliders of ``attacker_color`` that bear on ``target_sq`` along an
    open or single-blocker line. Returns a list of
    ``(piece_type, kind, blockers, dvec, dist)`` where ``kind`` is
    'file'/'rank'/'diag', ``blockers`` counts pieces standing between the slider
    and the target (0 = direct/open line, 1 = x-ray through one blocker, e.g. a
    pin), ``dvec`` is the direction vector, and ``dist`` is how many squares away
    the slider stands. Walking ray by ray lets us tell an open line from a blocked
    one, which is what distinguishes 'lost a pawn' from 'opened a file onto the
    king'; ``dist`` lets us ignore a piece sitting right next to the king (an
    ordinary check) versus one striking down a genuinely opened line."""
    tf, tr = chess.square_file(target_sq), chess.square_rank(target_sq)
    out = []
    for df, dr, kind in _ROOK_DIRS + _BISHOP_DIRS:
        rook_like = kind != "diag"
        blockers = 0
        steps = 0
        f, r = tf + df, tr + dr
        while 0 <= f < 8 and 0 <= r < 8:
            steps += 1
            p = board.piece_at(chess.square(f, r))
            if p is not None:
                match = (p.piece_type == chess.QUEEN
                         or (rook_like and p.piece_type == chess.ROOK)
                         or (not rook_like and p.piece_type == chess.BISHOP))
                if p.color == attacker_color and match:
                    out.append((p.piece_type, kind, blockers, (df, dr), steps))
                    break
                blockers += 1
                if blockers > 1:
                    break
            f += df
            r += dr
    return out


def _opened_line_fact(fen_before: str, played_uci: str, san_line: list[str],
                      mover: bool) -> Optional[str]:
    """Detect that the played move plus the opponent's refutation open a file or
    diagonal onto the mover's king or queen, handing an enemy rook/queen/bishop
    direct pressure (or a pin through a single blocker). This is the true reason
    many 'you just lost a pawn' moves are actually serious: the *line*, not the
    pawn, is the point. We only report a line as opened when it was blocked/quiet
    before the move and is open (or pins the king) after the forcing sequence, so
    a file that was already open is never mis-credited to this move.

    The exposure often peaks *inside* the line rather than at its end (e.g. a
    rook checks along the newly opened file and the king is then chased off it),
    so we scan every position along the sequence and report the first one where a
    previously closed line opens onto the mover's king or queen."""
    try:
        before = chess.Board(fen_before)
    except (ValueError, TypeError):
        return None
    opp = not mover
    board = before.copy(stack=False)
    try:
        board.push(chess.Move.from_uci(played_uci))
    except (ValueError, TypeError, AssertionError):
        return None
    positions = [board.copy(stack=False)]
    for san in san_line:
        try:
            board.push(board.parse_san(san))
        except ValueError:
            break
        positions.append(board.copy(stack=False))

    # Collect every newly-opened line onto the mover's king/queen across the
    # sequence, then keep the most telling one. Priority: a check delivered along
    # the opened line (the king is hit right now) > a pin of the king > the king
    # simply standing on an open line > the queen exposed. This stops a fleeting
    # attack on the queen from hiding the real point (the king on an opened file).
    best: Optional[tuple[int, str]] = None
    for pos in positions:
        in_check = pos.is_check()
        targets: list[tuple[int, str]] = []
        ksq = pos.king(mover)
        if ksq is not None:
            targets.append((ksq, "王"))
        for sq, p in pos.piece_map().items():
            if p.color == mover and p.piece_type == chess.QUEEN:
                targets.append((sq, "后"))

        for tsq, tname in targets:
            after_p = _slider_pressure(pos, tsq, opp)
            if not after_p:
                continue
            before_by_dir = {dvec: bl for (_pt, _k, bl, dvec, _d) in
                             _slider_pressure(before, tsq, opp)}
            for pt, kind, blockers, dvec, dist in after_p:
                was = before_by_dir.get(dvec)
                newly_open = blockers == 0 and (was is None or was >= 1)
                pin = tname == "王" and blockers == 1 and (was is None or was >= 2)
                # A slider right next to the king is an ordinary check, not an
                # opened line; require a genuine distance down the line.
                if dist < 2 or not (newly_open or pin):
                    continue
                if tname == "王":
                    score = 5 if (in_check and blockers == 0) else (4 if pin else 3)
                else:
                    score = 1
                if best is not None and score <= best[0]:
                    continue
                pzh = _PIECE_ZH.get(pt, "子")
                if kind == "file":
                    label = f"{chess.FILE_NAMES[chess.square_file(tsq)]} 线"
                elif kind == "rank":
                    label = f"第 {chess.RANK_NAMES[chess.square_rank(tsq)]} 横线"
                else:
                    label = "斜线"
                if pin:
                    fact = (f"【线路】这一步暴露了{label}：你的王被对方的{pzh}沿这条"
                            "线牵制（中间的子被别住形成钉子）——真正的隐患是线路，"
                            "不只是丢子。")
                elif tname == "王" and in_check:
                    fact = (f"【线路】实走之后经这段变化打开了{label}，对方的{pzh}"
                            "顺着这条线直接将军你的王——真正的代价是被打开的线路，"
                            "而不只是那个兵/子。")
                else:
                    fact = (f"【线路】实走之后经这段变化打开了{label}，你的{tname}正"
                            f"落在这条线上，对方的{pzh}由此获得直接压制——真正的代价"
                            "是被打开的线路，而不只是那个兵/子。")
                best = (score, fact)
    return best[1] if best else None


# Central squares whose control we track for the positional diff.
_CENTER = (chess.D4, chess.E4, chess.D5, chess.E5)


def _minor_major_mobility(board: chess.Board, color: bool,
                          piece_types: tuple[int, ...]) -> int:
    """Total pseudo-mobility (squares attacked that are empty or hold an enemy)
    of ``color``'s pieces of the given types."""
    total = 0
    for sq, p in board.piece_map().items():
        if p.color != color or p.piece_type not in piece_types:
            continue
        for t in board.attacks(sq):
            q = board.piece_at(t)
            if q is None or q.color != color:
                total += 1
    return total


def _center_control(board: chess.Board, color: bool) -> int:
    return sum(1 for s in _CENTER if board.is_attacked_by(color, s))


def _king_shelter(board: chess.Board, color: bool) -> int:
    """Count ``color``'s own pawns shielding its king: same or adjacent file and
    1-2 ranks in front of the king (in the direction that color advances)."""
    ksq = board.king(color)
    if ksq is None:
        return 0
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    cnt = 0
    for sq, p in board.piece_map().items():
        if p.color != color or p.piece_type != chess.PAWN:
            continue
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if abs(f - kf) > 1:
            continue
        ahead = (r - kr) if color == chess.WHITE else (kr - r)
        if 1 <= ahead <= 2:
            cnt += 1
    return cnt


def _pawn_weaknesses(board: chess.Board, color: bool) -> int:
    """Number of doubled + isolated pawns for ``color`` (a rough structural
    weakness count; higher is worse)."""
    files: dict[int, int] = {}
    for sq, p in board.piece_map().items():
        if p.color == color and p.piece_type == chess.PAWN:
            f = chess.square_file(sq)
            files[f] = files.get(f, 0) + 1
    doubled = sum(c - 1 for c in files.values() if c > 1)
    isolated = sum(c for f, c in files.items()
                   if (f - 1) not in files and (f + 1) not in files)
    return doubled + isolated


def _positional_diff(after_best: chess.Board, after_played: chess.Board,
                     mover: bool) -> Optional[str]:
    """Name the single positional feature the played move degraded most compared
    with the best move. Both positions are one ply deep (opponent to move), so we
    isolate exactly what choosing this move - instead of the best one - cost
    ``mover`` structurally. Checked in priority order (most concrete first);
    returns a Chinese phrase or ``None`` when nothing meaningful changed."""
    if _king_shelter(after_played, mover) - _king_shelter(after_best, mover) <= -1:
        return "把自己王前的兵盾走薄了，王的安全下降（王翼更漏风）"
    if _pawn_weaknesses(after_played, mover) - _pawn_weaknesses(after_best, mover) >= 1:
        return "让自己的兵形留下长期弱点（多出孤兵/叠兵）"
    bishops = (chess.BISHOP,)
    if _minor_major_mobility(after_played, mover, bishops) \
            - _minor_major_mobility(after_best, mover, bishops) <= -3:
        return "让自己的象活动空间明显变小（有变成坏象的趋势）"
    if _center_control(after_played, mover) - _center_control(after_best, mover) <= -2:
        return "削弱了自己对中心要点（d4/e4/d5/e5）的控制"
    pieces = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    if _minor_major_mobility(after_played, mover, pieces) \
            - _minor_major_mobility(after_best, mover, pieces) <= -4:
        return "让整体子力更被动、活动空间变小，少了先手"
    return None


def _choice_fact(m: "MoveAnalysis") -> Optional[str]:
    """A 【选择】 observation drawn from the MultiPV context: was the best move the
    only move, or were there several equally good options? This lets the coach
    match tone to how forced the position was — an only-move is hard to find and
    forgivable, whereas ignoring several easy good moves is more avoidable.
    Returns a phrase, or ``None`` when the move was not measured / inconclusive."""
    if m.alt_count >= 3:
        return (f"【选择】这里其实有大约 {m.alt_count} 步都不错（评估接近），"
                "实走这手偏差较明显——不是没有选择，而是选偏了。")
    if m.alt_gap_cp >= 200:
        if m.alt_gap_cp >= 600:
            lead = "次选也要差出一大截"
        else:
            pawns = max(1, round(m.alt_gap_cp / 100))
            lead = f"次选也要差约 {pawns} 个兵"
        return (f"【选择】{m.best_move_san} 几乎是这里的唯一解，{lead}，"
                "属于不易找到的一步。")
    if m.alt_gap_cp >= 80:
        pawns = max(1, round(m.alt_gap_cp / 100))
        return (f"【选择】{m.best_move_san} 明显强于次选（约 {pawns} 个兵），"
                "是这里的关键正着。")
    return None


def _move_facts(m: "MoveAnalysis") -> list[str]:
    """A battery of concrete, verifiable observations about a flagged move, each
    tagged by category. Every consequence claim is read off the engine's actual
    refutation line (ground truth), not guessed one ply deep — so the set varies
    by move and never asserts a 'hanging piece' the line does not truly win."""
    facts: list[str] = []
    try:
        board = chess.Board(m.fen_before)
    except (ValueError, TypeError):
        return facts
    mover = board.turn
    opp = not mover
    swing = (_line_material_swing(m.fen_before, m.uci, m.refutation_line_san, mover)
             if m.refutation_line_san else 0)

    best = None
    try:
        best = chess.Move.from_uci(m.best_move_uci)
    except (ValueError, TypeError):
        best = None
    after_best = None
    if best is not None and best in board.legal_moves:
        after_best = board.copy(stack=False)
        after_best.push(best)

    # --- what the best move achieved (the missed resource) ------------------
    # Tactical claims are validated against the engine's own best line: we only
    # say the best move 'wins material' or 'forks' when that line actually nets
    # the mover material (or the fork hits the king), never from 1-ply geometry.
    best_gain = _best_line_material_gain(m.fen_before, m.best_line_san, mover)
    if after_best is not None:
        if m.best_is_check:
            facts.append(f"【正解】最佳着法 {m.best_move_san} 是将军，能抢到先手。")
        elif board.is_capture(best) and best_gain >= 1:
            cap = board.piece_at(best.to_square)
            capname = _PIECE_ZH.get(cap.piece_type, "子") if cap else "子"
            facts.append(
                f"【子力】最佳着法 {m.best_move_san} 吃掉对方的{capname}，"
                f"按主变走下去最终净得{_pts_zh(best_gain)}。")
        fork = _fork_targets(after_best, best.to_square)
        if fork and (best_gain >= 1 or "王" in fork):
            tail = (f"，主变里最终净得{_pts_zh(best_gain)}" if best_gain >= 1
                    else "，形成强有力的双重攻击")
            facts.append(
                f"【战术】最佳着法 {m.best_move_san} 同时攻击对方的{fork}"
                f"（叉子 / 双重攻击）{tail}。")

    # --- the concrete consequence, read off the actual refutation line ------
    if m.refutation_line_san:
        line = " ".join(m.refutation_line_san)
        if swing <= -1:
            pivot = _pivot_capture(m.fen_before, m.uci, m.refutation_line_san, mover)
            if pivot:
                pv_san, cap_zh = pivot
                facts.append(
                    f"【对手回应】实走之后对手的最强回应：{line}——"
                    f"关键在 {pv_san} 吃掉你的{cap_zh}，你大约净丢{_pts_zh(swing)}。")
            else:
                facts.append(
                    f"【对手回应】实走之后对手的最强回应：{line}——"
                    f"这条变化里你大约会净丢{_pts_zh(swing)}。")
        else:
            facts.append(f"【对手回应】实走之后对手的最强回应：{line}。")

    # --- lines: did the move open a file/diagonal onto our king or queen? ----
    # Frequently the real cost of a pawn move/capture is not the pawn but the
    # line it vacates. Only for genuine mistakes, and verified against the actual
    # forcing sequence.
    if m.cp_loss >= 100:
        line_fact = _opened_line_fact(m.fen_before, m.uci, m.refutation_line_san,
                                      mover)
        if line_fact:
            facts.append(line_fact)

    # --- the dominant positional feature the played move degraded -----------
    # Only for non-material slips: when nothing is genuinely hung, the eval drop
    # is structural, so name the concrete feature instead of calling it 'subtle'.
    if after_best is not None and swing > -1:
        try:
            after_played = chess.Board(m.fen_after)
        except (ValueError, TypeError):
            after_played = None
        if after_played is not None:
            feature = _positional_diff(after_best, after_played, mover)
            if feature:
                facts.append(
                    f"【位置】和最佳着法 {m.best_move_san} 相比，这一步{feature}。")

    # --- king safety: only when a king hunt is genuinely the theme ----------
    king_theme = bool(m.best_is_check or (m.mate_before and m.mate_before > 0))
    if king_theme and after_best is not None:
        flight_before = _king_flight_squares(board, opp)
        flight_best = _king_flight_squares(after_best, opp)
        if flight_best < flight_before:
            facts.append(
                f"【王的安全】对方王本有约 {flight_before} 个逃格；"
                f"最佳着法 {m.best_move_san} 后只剩约 {flight_best} 个，杀网收紧。")
    if m.mate_before and m.mate_before > 0:
        facts.append(f"【强制】最佳着法可导向约 {m.mate_before} 步的强制杀。")

    choice = _choice_fact(m)
    if choice:
        facts.append(choice)

    return facts


def _move_verdict(m: "MoveAnalysis") -> dict:
    """Classify *why* the eval dropped and how to frame it honestly, using the
    material swing the engine's refutation line actually produces."""
    try:
        mover = chess.Board(m.fen_before).turn
    except (ValueError, TypeError):
        mover = m.color
    swing = (_line_material_swing(m.fen_before, m.uci, m.refutation_line_san, mover)
             if m.refutation_line_san else 0)
    category, rstate = classify_delta(
        m.eval_before_mover, m.eval_after_mover,
        m.mate_before, m.mate_after, swing, m.cp_loss)
    return {
        "category": category,
        "category_zh": CATEGORY_ZH.get(category, ""),
        "resulting_state": rstate,
        "framing": CATEGORY_FRAME.get(category, ""),
        "subtle": category == "positional_slip",
        "material_swing": swing,
    }


def _explain_for(m: MoveAnalysis, tag: str) -> dict:
    """Explanation for a flagged move: LLM-polished when a key is configured,
    otherwise the deterministic template. Grounded strictly on engine facts."""
    template = _explain_move_zh(m)
    if not coach_llm.available():
        return template
    verdict = _move_verdict(m)
    ctx = {
        "phase": PHASE_ZH.get(m.phase, m.phase),
        "side": "白方" if m.color == chess.WHITE else "黑方",
        "move_number": m.move_number,
        "played": m.san,
        "best": m.best_move_san,
        "pv": " ".join(m.best_line_san) if m.best_line_san else m.best_move_san,
        "refutation": " ".join(m.refutation_line_san),
        "eval_before": format_eval(m.eval_before_mover, m.mate_before),
        "eval_after": format_eval(m.eval_after_mover, m.mate_after),
        "state_before": _state_zh(m.eval_before_mover),
        "state_after": _state_zh(m.eval_after_mover),
        "cp_loss": m.cp_loss,
        "reason_tag": tag,
        "category": verdict["category"],
        "category_zh": verdict["category_zh"],
        "resulting_state": verdict["resulting_state"],
        "framing": verdict["framing"],
        "subtle": verdict["subtle"],
        "best_is_check": m.best_is_check,
        "best_is_capture": m.best_is_capture,
        "best_leads_to_mate_in": m.mate_before if (m.mate_before and m.mate_before > 0) else None,
        "facts": _move_facts(m),
    }
    polished = coach_llm.polish_explanation(ctx)
    return polished or template


# ---------------------------------------------------------------------------
# view builders
# ---------------------------------------------------------------------------
def _board_svg(fen: str, played_uci: str, best_uci: str, color: bool, size: int = 340) -> str:
    board = chess.Board(fen)
    arrows = []
    if best_uci and len(best_uci) >= 4:
        arrows.append(chess.svg.Arrow(
            chess.parse_square(best_uci[0:2]), chess.parse_square(best_uci[2:4]), color="#2e7d32"))
    if played_uci and len(played_uci) >= 4 and played_uci != best_uci:
        arrows.append(chess.svg.Arrow(
            chess.parse_square(played_uci[0:2]), chess.parse_square(played_uci[2:4]), color="#c62828"))
    return chess.svg.board(board, arrows=arrows, size=size, orientation=color)


def _critical_moments(ga: GameAnalysis, color: Optional[bool], threshold: int,
                      with_svg: bool) -> list[dict]:
    moments = []
    for m in ga.moves:
        if color is not None and m.color != color:
            continue
        keep, tag = _significance(m, threshold)
        if not keep:
            continue
        moments.append({
            "move_number": m.move_number,
            "side": "白方" if m.color == chess.WHITE else "黑方",
            "played": m.san,
            "best": m.best_move_san,
            "best_line": " ".join(m.best_line_san) if m.best_line_san else m.best_move_san,
            "cp_loss": m.cp_loss,
            "eval_before": format_eval(m.eval_before_mover, m.mate_before),
            "eval_after": format_eval(m.eval_after_mover, m.mate_after),
            "phase": PHASE_ZH.get(m.phase, m.phase),
            "phase_key": m.phase,
            "classification": m.classification,
            "class_zh": CLASS_ZH.get(m.classification, m.classification),
            "forcing_miss": m.is_forcing_miss,
            "tag": tag,
            "tag_zh": TAG_ZH.get(tag, ""),
            "comment": _comment_for(m),
            "explain": _explain_for(m, tag),
            "fen": m.fen_before,
            "lichess": lichess_url(m.fen_before),
            "svg": _board_svg(m.fen_before, m.uci, m.best_move_uci, m.color) if with_svg else "",
        })
    return moments


def _turning_point(ga: GameAnalysis, color: bool) -> Optional[dict]:
    """First move after which `color`'s eval collapses and never recovers."""
    traj = ga.eval_trajectory_white()
    povs = [_pov(v, color) for v in traj]
    for i in range(1, len(povs)):
        if povs[i] <= TURNING_THRESHOLD and max(povs[i:]) <= RECOVERY_THRESHOLD:
            m = ga.moves[i - 1]
            return {
                "move_number": m.move_number,
                "side": "白方" if m.color == chess.WHITE else "黑方",
                "played": m.san,
                "best": m.best_move_san,
                "eval_before": format_eval(m.eval_before_mover, m.mate_before),
                "eval_after": format_eval(m.eval_after_mover, m.mate_after),
                "phase": PHASE_ZH.get(m.phase, m.phase),
                "fen": m.fen_before,
                "lichess": lichess_url(m.fen_before),
            }
    return None


def _side_summary(ga: GameAnalysis, color: bool) -> dict:
    moves = ga.player_moves(color)
    n = len(moves)
    acpl = round(sum(m.cp_loss for m in moves) / n, 1) if n else 0.0
    biggest = max(moves, key=lambda m: m.cp_loss, default=None)
    phase_stats = {}
    for phase in PHASES:
        pm = [m for m in moves if m.phase == phase]
        phase_stats[phase] = {
            "moves": len(pm),
            "acpl": round(sum(x.cp_loss for x in pm) / len(pm), 1) if pm else 0.0,
            "blunders": sum(1 for x in pm if x.cp_loss >= BLUNDER),
            "mistakes": sum(1 for x in pm if MISTAKE <= x.cp_loss < BLUNDER),
        }
    return {
        "name": _side_name(ga, color),
        "acpl": acpl,
        "blunders": sum(1 for m in moves if m.cp_loss >= BLUNDER),
        "mistakes": sum(1 for m in moves if MISTAKE <= m.cp_loss < BLUNDER),
        "inaccuracies": sum(1 for m in moves if 50 <= m.cp_loss < MISTAKE),
        "biggest": None if biggest is None else {
            "move_number": biggest.move_number,
            "played": biggest.san,
            "best": biggest.best_move_san,
            "cp_loss": biggest.cp_loss,
            "lichess": lichess_url(biggest.fen_before),
        },
        "phases": phase_stats,
    }


def _san_line(moves: list, plies: int = 10) -> str:
    """Render the first `plies` half-moves as '1. e4 e5 2. Nf3 ...'."""
    out: list[str] = []
    for m in moves[:plies]:
        if m.color == chess.WHITE:
            out.append(f"{m.move_number}. {m.san}")
        elif not out:
            out.append(f"{m.move_number}... {m.san}")
        else:
            out.append(m.san)
    return " ".join(out)


def _book_line_from(start_fen: str, first_uci: str, plies: int = 10) -> tuple[str, str]:
    """Follow the opening book from ``start_fen``, playing ``first_uci`` then the
    single most popular book move for up to ``plies`` half-moves. Returns the
    rendered SAN line ("3... d5 4. exd5 ...") and the FEN of the final position.
    """
    book = get_default_book()
    board = chess.Board(start_fen)
    sans: list[str] = []
    uci: Optional[str] = first_uci
    for _ in range(plies):
        if not uci:
            break
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            break
        if mv not in board.legal_moves:
            break
        if board.turn == chess.WHITE:
            sans.append(f"{board.fullmove_number}. {board.san(mv)}")
        elif not sans:
            sans.append(f"{board.fullmove_number}... {board.san(mv)}")
        else:
            sans.append(board.san(mv))
        board.push(mv)
        nxt = book.lookup(board.fen(), top=1)
        uci = nxt[0]["uci"] if nxt else None
    return " ".join(sans), board.fen()


def build_opening_section(ga: GameAnalysis) -> dict:
    """Assemble the opening breakdown: ECO, the played line, the first
    out-of-book move, and the opening book's top-two continuations (each
    followed ~5 moves) with a lichess analysis link."""
    moves = ga.moves
    line_played = _san_line(moves, plies=10)

    dev_move = None
    if ga.deviation_ply is not None:
        dev_move = next((m for m in moves if m.ply == ga.deviation_ply), None)

    book = get_default_book()

    # Anchor the continuation choices at the *last real choice point*: the latest
    # opening position (up to and including the deviation) where the book still
    # offered at least two moves. The position right before leaving book is
    # usually near-exhausted (only one book move), so anchoring there shows a
    # single line; walking back to where the player genuinely had alternatives
    # lets us present one or two real variations.
    anchor_fen = None
    anchor_move_number = None
    anchor_turn = None
    for m in moves:
        if ga.deviation_ply is not None:
            if m.ply > ga.deviation_ply:
                break
        elif m.phase != "opening":
            break
        if len(book.lookup(m.fen_before, top=2)) >= 2:
            anchor_fen = m.fen_before
            anchor_move_number = m.move_number
            anchor_turn = m.color

    # Fallbacks when no branching point was found (very narrow book line):
    # the deviation position, else the last opening-phase position.
    if anchor_fen is None:
        if dev_move is not None:
            anchor_fen = dev_move.fen_before
            anchor_move_number = dev_move.move_number
            anchor_turn = dev_move.color
        else:
            opening_moves = [m for m in moves if m.phase == "opening"]
            if opening_moves:
                anchor_fen = opening_moves[-1].fen_after
            elif moves:
                anchor_fen = moves[min(len(moves) - 1, 9)].fen_after
    query_fen = anchor_fen

    # Top-2 book choices at the anchor position, each extended into a short line.
    book_choices: list[dict] = []
    pg = book.lookup(query_fen, top=2) if query_fen else []
    for mv in pg:
        line, end_fen = _book_line_from(query_fen, mv["uci"], plies=10)
        book_choices.append({
            "san": mv["san"],
            "pct": mv["pct"],
            "line": line,
            "lichess": lichess_url(end_fen),
        })

    deviation = None
    if dev_move is not None:
        deviation = {
            "move_number": dev_move.move_number,
            "side": "白方" if dev_move.color == chess.WHITE else "黑方",
            "played": dev_move.san,
            "cp_loss": dev_move.cp_loss,
            "lichess": lichess_url(dev_move.fen_before),
        }

    return {
        "eco": ga.eco or "",
        "name": ga.opening_name or "",
        "line_played": line_played,
        "deviation": deviation,
        "in_book_full": ga.deviation_ply is None and bool(ga.opening_name),
        "book_choices": book_choices,
        "book_available": bool(book_choices),
        "anchor_move_number": anchor_move_number,
        "anchor_side": ("白方" if anchor_turn == chess.WHITE else "黑方")
                        if anchor_turn is not None else None,
    }


# ---------------------------------------------------------------------------
# phase grouping
# ---------------------------------------------------------------------------
def _phase_moments(moments: list, phase_key: str) -> list:
    return [m for m in moments if m.get("phase_key") == phase_key]


def _phase_summary_zh(phase_zh: str, group: list) -> str:
    if not group:
        return f"{phase_zh}没有达到阈值的问题着法，走得比较稳。"
    n_bl = sum(1 for m in group if m["classification"] == "blunder")
    n_mi = sum(1 for m in group if m["classification"] == "mistake")
    n_in = sum(1 for m in group if m["classification"] == "inaccuracy")
    parts = []
    if n_bl:
        parts.append(f"{n_bl} 个漏着")
    if n_mi:
        parts.append(f"{n_mi} 个错着")
    if n_in:
        parts.append(f"{n_in} 个不精确")
    return f"{phase_zh}出现 " + "、".join(parts) + "，下面逐条来看。"


def _perspective_block(ga: GameAnalysis, color: bool, threshold: int,
                       with_svg: bool) -> dict:
    """Full single-side perspective: narrative + critical moments + turning point."""
    turning = _turning_point(ga, color)
    if turning:
        turning["for"] = "白方" if color == chess.WHITE else "黑方"
    return {
        "color": "white" if color == chess.WHITE else "black",
        "side": "白方" if color == chess.WHITE else "黑方",
        "name": _side_name(ga, color),
        "summary_zh": build_summary_zh(ga, color),
        "stats": _side_summary(ga, color),
        "turning_point": turning,
        "critical_moments": _critical_moments(ga, color, threshold, with_svg),
    }


def build_game_view(ga: GameAnalysis, player: Optional[str] = None,
                    threshold: int = MISTAKE, with_svg: bool = True,
                    dual: bool = False) -> dict:
    focus_color = ga.player_color(player) if player else None

    losers = []
    if ga.result == "1-0":
        losers = [chess.BLACK]
    elif ga.result == "0-1":
        losers = [chess.WHITE]

    turning = None
    for c in (losers or [chess.WHITE, chess.BLACK]):
        turning = _turning_point(ga, c)
        if turning:
            turning["for"] = "白方" if c == chess.WHITE else "黑方"
            break

    deviation = None
    if ga.deviation_ply is not None:
        deviation = {
            "ply": ga.deviation_ply,
            "move_number": (ga.deviation_ply + 1) // 2,
            "side": "白方" if ga.deviation_side == chess.WHITE else "黑方",
            "played": ga.deviation_move_san,
            "book": ga.deviation_book_san,
        }

    view = {
        "white": ga.white,
        "black": ga.black,
        "result": ga.result,
        "date": ga.date,
        "event": ga.event,
        "site": ga.site,
        "eco": ga.eco,
        "opening_name": ga.opening_name,
        "deviation": deviation,
        "dual": dual,
        "focus": None if focus_color is None else ("白方" if focus_color == chess.WHITE else "黑方"),
        "summary_zh": build_summary_zh(ga, focus_color),
        "white_summary": _side_summary(ga, chess.WHITE),
        "black_summary": _side_summary(ga, chess.BLACK),
        "turning_point": turning,
        "opening_section": build_opening_section(ga),
        "critical_moments": _critical_moments(ga, focus_color, threshold, with_svg),
        "n_moves": len(ga.moves),
    }
    if dual:
        view["perspectives"] = [
            _perspective_block(ga, chess.WHITE, threshold, with_svg),
            _perspective_block(ga, chess.BLACK, threshold, with_svg),
        ]
    return view


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _md_critical(L: list, moments: list, empty_msg: str = "_没有超过阈值的问题着法。_") -> None:
    if not moments:
        L.append(empty_msg)
        L.append("")
        return
    for m in moments:
        e = m["explain"]
        L.append(f"#### 第 {m['move_number']} 回合 · {m['side']} · {m['class_zh']}"
                 f"（损失 -{m['cp_loss']}cp）")
        L.append(f"- 实走 `{m['played']}` → 应走 `{m['best']}`"
                 f"（{m['eval_before']} → {m['eval_after']}）")
        L.append(f"- **为什么错：** {e['why']}")
        L.append(f"- **造成的后果：** {e['consequence']}")
        L.append(f"- **应该怎么做：** {e['what_to_do']}")
        L.append(f"- [在 lichess 上打开这个局面]({m['lichess']})")
        L.append("")


def _md_opening(L: list, sec: dict) -> None:
    if sec.get("eco") or sec.get("name"):
        L.append(f"- **开局定式：** {(sec['eco'] + ' ' + sec['name']).strip()}")
    if sec.get("line_played"):
        L.append(f"- **实战着法：** {sec['line_played']}")
    if sec.get("deviation"):
        d = sec["deviation"]
        L.append(f"- **首个脱谱点：** 第 {d['move_number']} 回合 {d['side']} 走了 "
                 f"`{d['played']}`。")
    elif sec.get("in_book_full"):
        L.append("- **脱谱点：** 全程跟随理论主线，没有脱谱。")
    if sec.get("book_choices"):
        anchor = ""
        if sec.get("anchor_move_number"):
            side = sec.get("anchor_side") or ""
            anchor = f"（第 {sec['anchor_move_number']} 回合{side}的选择点）"
        L.append(f"- **开局库续法{anchor}（前两个选择，各续约 5 回合）：**")
        labels = ["首选", "次选"]
        for i, bc in enumerate(sec["book_choices"]):
            lbl = labels[i] if i < len(labels) else f"选择{i + 1}"
            pct = f"（{bc['pct']}% 权重）" if bc.get("pct") is not None else ""
            line = f"{bc['line']}" if bc.get("line") else f"`{bc['san']}`"
            link = f" · [在 lichess 上打开]({bc['lichess']})" if bc.get("lichess") else ""
            L.append(f"  - {lbl} `{bc['san']}`{pct}：{line}{link}")
    L.append("")


def _md_phase(L: list, title: str, phase_zh: str, moments: list) -> None:
    if not moments:
        return  # hide phase sections with nothing above threshold
    L.append(f"## {title}")
    L.append(_phase_summary_zh(phase_zh, moments))
    L.append("")
    _md_critical(L, moments)


def _md_turning(L: list, t: dict) -> None:
    L.append("**转折点：** "
             f"第 {t['move_number']} 回合 {t['side']} `{t['played']}`"
             f"（{t['eval_before']} → {t['eval_after']}），{t['for']}此后再没回来；"
             f"应走 `{t['best']}`。[打开分析]({t['lichess']})")
    L.append("")


def render_game_markdown(view: dict) -> str:
    L = []
    L.append(f"# 对局分析 — {view['white']} vs {view['black']}")
    meta = " · ".join(filter(None, [
        view["event"], view["date"], f"结果 {view['result']}",
        f"{view['eco']} {view['opening_name']}".strip(),
    ]))
    if meta:
        L.append(f"_{meta}_")
    L.append("")

    # ===================== 一、总结 =========================================
    L.append("## 一、总结")
    sm = view.get("summary_zh")
    if sm:
        L.append(f"> **{sm['headline']}**")
        L.append("")
        L.append("**做得好的：**")
        for item in sm["good"]:
            L.append(f"- {item}")
        L.append("")
        L.append("**做得不好的：**")
        for item in sm["bad"]:
            L.append(f"- {item}")
        L.append("")
        L.append("**训练建议：**")
        for item in sm["advice"]:
            L.append(f"- {item}")
        L.append("")

    for label, s in [("白方", view["white_summary"]), ("黑方", view["black_summary"])]:
        L.append(f"**{label} · {s['name']}** — 平均每步损失（ACPL）**{s['acpl']}**："
                 f"漏着 {s['blunders']} · 错着 {s['mistakes']} · 不精确 {s['inaccuracies']}")
        seg = []
        for ph in PHASES:
            p = s["phases"][ph]
            if p["moves"]:
                seg.append(f"{PHASE_ZH.get(ph, ph)} {p['acpl']}")
        if seg:
            L.append("- 分阶段 ACPL：" + " / ".join(seg))
    L.append("")
    if view.get("turning_point"):
        _md_turning(L, view["turning_point"])

    # ===================== 二、开局板块 =====================================
    L.append("## 二、开局板块")
    _md_opening(L, view["opening_section"])
    open_moments = _phase_moments(view["critical_moments"], "opening")
    if open_moments:
        L.append("**开局阶段问题着法：**")
        L.append("")
        _md_critical(L, open_moments)

    if view.get("dual") and view.get("perspectives"):
        # per-side middlegame / endgame breakdowns
        for p in view["perspectives"]:
            L.append(f"# 【{p['side']}视角】{p['name']}")
            L.append("")
            _md_phase(L, "中局板块", "中局", _phase_moments(p["critical_moments"], "middlegame"))
            _md_phase(L, "残局板块", "残局", _phase_moments(p["critical_moments"], "endgame"))
        return "\n".join(L)

    # ===================== 三、中局板块 =====================================
    _md_phase(L, "三、中局板块", "中局", _phase_moments(view["critical_moments"], "middlegame"))
    # ===================== 四、残局板块 =====================================
    _md_phase(L, "四、残局板块", "残局", _phase_moments(view["critical_moments"], "endgame"))
    return "\n".join(L)


def render_player_markdown(r: dict) -> str:
    L = []
    L.append(f"# Player Report — {r['player']}")
    if r.get("error"):
        L.append(f"\n> {r['error']}")
        return "\n".join(L)
    rec = r["record"]
    line = f"{r['n_games']} games · {rec['wins']}-{rec['draws']}-{rec['losses']} " \
           f"({_pct(rec['score_pct'])}) · white {r['as_white']} / black {r['as_black']}"
    if r.get("avg_opponent"):
        line += f" · avg opp {r['avg_opponent']}"
    L.append(line)
    L.append(f"Overall ACPL **{r['acpl']}** · {r['blunders_total']} blunders · "
             f"{r['mistakes_total']} mistakes")
    L.append("")

    L.append("## Where points are lost (by phase)")
    L.append("| Phase | Moves | Avg loss | Blunder % | Mistake % |")
    L.append("|---|---:|---:|---:|---:|")
    for p in r["phases"]:
        L.append(f"| {p['phase']} | {p['moves']} | {p['avg_loss']} | "
                 f"{_pct(p['blunder_rate'])} | {_pct(p['mistake_rate'])} |")
    L.append("")

    L.append("## Middlegame failure rate by move number")
    L.append("| Moves | N | Failure rate (>=1 pawn) |")
    L.append("|---|---:|---:|")
    for b in r["middlegame_by_move"]:
        L.append(f"| {b['bucket']} | {b['moves']} | {_pct(b['failure_rate'])} |")
    L.append("")

    c = r["conversion"]
    L.append("## Converting winning positions (peak >= +2)")
    L.append(f"- {c['n']} games · {c['wins']}-{c['draws']}-{c['losses']} · "
             f"win rate **{_pct(c['win_rate'])}** (score {_pct(c['score_pct'])})")
    if r.get("conversion_vs_2000plus") is not None:
        L.append(f"- vs 2000+: {_pct(r['conversion_vs_2000plus'])} · "
                 f"vs <2000: {_pct(r['conversion_vs_sub2000'])}")
    L.append("")

    L.append("## Outcome by evaluation when entering the endgame")
    L.append("| Entering endgame | N | W-D-L | Score |")
    L.append("|---|---:|:---:|---:|")
    for e in r["endgame_entry"]:
        L.append(f"| {e['label']} | {e['n']} | {e['wins']}-{e['draws']}-{e['losses']} | "
                 f"{_pct(e['score_pct'])} |")
    L.append("")

    res = r["resilience"]
    L.append("## Resilience when behind (<= -2)")
    L.append(f"- {res['n']} games · saved {res['saved']} · save rate **{_pct(res['save_rate'])}**")
    L.append("")

    L.append("## Biggest blunders")
    L.append("| Date | Opponent | Move | Played → Best | Loss | Phase | Link |")
    L.append("|---|---|---:|---|---:|---|---|")
    for b in r["top_blunders"]:
        opp = b["opponent"] + (f" ({b['opponent_rating']})" if b["opponent_rating"] else "")
        L.append(f"| {b['date']} | {opp} | {b['move_number']} | "
                 f"`{b['played']}` → `{b['best']}` | -{b['cp_loss']}cp | {b['phase']} | "
                 f"[analyse]({lichess_url(b['fen'])}) |")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# HTML (Jinja2)
# ---------------------------------------------------------------------------
def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["pct"] = _pct
    env.filters["evalfmt"] = lambda cp: format_eval(cp)
    return env


def render_game_html(view: dict) -> str:
    return _env().get_template("game.html.j2").render(v=view, phases=PHASES, phase_zh=PHASE_ZH)


def render_player_html(report: dict) -> str:
    return _env().get_template("player.html.j2").render(r=report, phases=PHASES,
                                                        lichess_url=lichess_url)
