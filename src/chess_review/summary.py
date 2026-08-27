"""Generate a plain-language Chinese summary of a game for a coach.

The summary is derived deterministically from the engine analysis (no LLM):
a one-line headline plus what the player did well, what went wrong, and
concrete training advice.
"""
from __future__ import annotations

from typing import Optional

import chess

from .classify import BLUNDER, MISTAKE, significance
from .models import GameAnalysis, MoveAnalysis


def _pawns(cp: int) -> str:
    return f"{cp / 100:+.1f}"


def _pov(eval_white: int, color: bool) -> int:
    return eval_white if color == chess.WHITE else -eval_white


def _side_zh(color: bool) -> str:
    return "白方" if color == chess.WHITE else "黑方"


def _choose_focus(ga: GameAnalysis) -> bool:
    """Pick whom to summarize when no player is specified."""
    if ga.result == "1-0":
        return chess.BLACK  # the side that lost is more instructive
    if ga.result == "0-1":
        return chess.WHITE
    # Draw: focus on the side that lost more centipawns.
    w = sum(m.cp_loss for m in ga.player_moves(chess.WHITE))
    b = sum(m.cp_loss for m in ga.player_moves(chess.BLACK))
    return chess.WHITE if w >= b else chess.BLACK


def _describe_move(m: MoveAnalysis) -> str:
    nature = ""
    if m.is_forcing_miss:
        kind = "将军" if m.best_is_check else "吃子"
        nature = f"漏掉强制{kind}"
    else:
        nature = "判断/计划失误"
    return (f"第{m.move_number}回合 {m.san}（应走 {m.best_move_san}，"
            f"损失 -{m.cp_loss}cp，{nature}）")


def build_summary_zh(ga: GameAnalysis, focus_color: Optional[bool] = None) -> dict:
    color = focus_color if focus_color is not None else _choose_focus(ga)
    name = ga.white if color == chess.WHITE else ga.black
    opp = ga.black if color == chess.WHITE else ga.white
    side = _side_zh(color)

    moves = ga.player_moves(color)
    n = len(moves)
    acpl = round(sum(m.cp_loss for m in moves) / n, 1) if n else 0.0
    score = ga.result_score(color)

    blunders = [m for m in moves if m.cp_loss >= BLUNDER]
    mistakes = [m for m in moves if MISTAKE <= m.cp_loss < BLUNDER]
    big_errors = sorted(blunders + mistakes, key=lambda m: m.cp_loss, reverse=True)
    # Errors that actually changed the outcome (win thrown away / fell into
    # trouble / missed a decisive mate) — used for the visible "关键失误" list so
    # the summary highlights the same moves as the detailed sections.
    key_errors = [m for m in big_errors if significance(m)[0]]
    forcing_misses = [m for m in big_errors if m.is_forcing_miss]
    quiet_errors = [m for m in big_errors if not m.is_forcing_miss]
    biggest = big_errors[0] if big_errors else None

    povs = [_pov(v, color) for v in ga.eval_trajectory_white()]
    peak = max(povs) if povs else 0
    low = min(povs) if povs else 0

    # phase ACPL
    phase_acpl = {}
    for ph in ("opening", "middlegame", "endgame"):
        pm = [m for m in moves if m.phase == ph]
        phase_acpl[ph] = round(sum(x.cp_loss for x in pm) / len(pm), 1) if pm else None

    deviated = (ga.deviation_side == color and ga.deviation_ply is not None)

    # ---- headline ----------------------------------------------------------
    headline = _headline(side, name, opp, score, peak, low, biggest, big_errors,
                         forcing_misses, quiet_errors)

    # ---- good --------------------------------------------------------------
    good: list[str] = []
    for ph, label in (("opening", "开局"), ("middlegame", "中局"), ("endgame", "残局")):
        val = phase_acpl[ph]
        if val is not None and val < 20 and any(m.phase == ph for m in moves):
            good.append(f"{label}走得干净（平均每步损失 {val}cp）。")
    if not deviated and ga.deviation_ply is not None:
        good.append("开局跟着理论走，没有过早脱谱。")
    if score == 1.0 and not blunders:
        good.append("全程没有漏着，把优势稳稳兑现成胜利。")
    if peak >= 200 and score == 1.0:
        good.append(f"拿到 {_pawns(peak)} 的优势并成功转化。")
    if low <= -200 and score and score > 0:
        good.append(f"一度落后到 {_pawns(low)}，最终顽强守下来。")
    if not good:
        good.append("局面判断基本合理，没有明显的战略性失误。")

    # ---- bad ---------------------------------------------------------------
    bad: list[str] = []
    if deviated:
        book = f"（理论主线 {ga.deviation_book_san}）" if ga.deviation_book_san else ""
        bad.append(f"第{(ga.deviation_ply + 1)//2}回合 {ga.deviation_move_san} 过早脱离开局理论{book}。")
    for m in key_errors[:4]:
        bad.append(_describe_move(m) + "。")
    if peak >= 200 and (score is None or score < 1.0):
        bad.append(f"曾手握 {_pawns(peak)} 的胜势，却没能拿下。")
    if not bad:
        bad.append("没有明显失误可挑，属于高质量的一盘。")

    # ---- advice ------------------------------------------------------------
    advice: list[str] = []
    if len(forcing_misses) >= 1 and len(forcing_misses) >= len(quiet_errors):
        advice.append("多练算杀与强制线：每步先把所有吃子和将军算一遍，再考虑别的。")
    if len(quiet_errors) >= 1 and len(quiet_errors) > len(forcing_misses):
        advice.append("多练局面判断：先问“对手想干什么、我最差的子是哪个、换哪对子有利”。")
    if peak >= 200 and (score is None or score < 1.0):
        advice.append("优势局面收尾：+2 以上时先找强制线，在15回合内把优势变成不可逆。")
    if low <= -200 and (score == 0.0):
        advice.append("落后后加强韧性：找最顽强的防守和反击，别急着放弃。")
    if not advice:
        advice.append("保持现有思路，继续用完整慢棋积累经验。")

    return {
        "focus_name": name,
        "side": side,
        "result_text": {1.0: "胜", 0.5: "和", 0.0: "负"}.get(score, "未结束"),
        "acpl": acpl,
        "headline": headline,
        "good": good,
        "bad": bad,
        "advice": advice,
    }


def _headline(side, name, opp, score, peak, low, biggest, big_errors,
              forcing_misses, quiet_errors) -> str:
    # Winning-but-not-won
    if peak >= 200 and (score is not None and score < 1.0):
        tail = "被翻盘告负" if score == 0.0 else "只走成和棋"
        return f"{side}曾手握 {_pawns(peak)} 的胜势，却因收尾不力{tail}。"

    if score == 1.0:
        if not big_errors:
            return f"{side}全程零失误，抓住对手的机会干净利落地拿下。"
        return f"{side}虽有小瑕疵，但抓住对手的漏着赢得对局。"

    if score == 0.0:
        if biggest and biggest.cp_loss >= 300:
            nature = "漏掉强制着" if biggest.is_forcing_miss else "判断失误"
            return (f"{side}整体接近均势，却在第{biggest.move_number}回合一步{nature}"
                    f"（{biggest.san}，损失 -{biggest.cp_loss}cp）断送全局。")
        return f"{side}没有致命的一步，是被一连串小失误慢慢磨垮。"

    # draw
    if peak >= 150:
        return f"{side}占到 {_pawns(peak)} 的优势但没能转化，最终握手言和。"
    return f"{side}与对手全程势均力敌，双方无重大失误，和棋合理。"
