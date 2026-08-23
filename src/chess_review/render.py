"""Render analyses into Markdown and HTML reports (with board diagrams)."""
from __future__ import annotations

import os
from typing import Optional

import chess
import chess.svg
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .classify import BLUNDER, MISTAKE
from .metrics import PHASES
from .models import GameAnalysis, MoveAnalysis
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
        if m.cp_loss < threshold:
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
            "classification": m.classification,
            "class_zh": CLASS_ZH.get(m.classification, m.classification),
            "forcing_miss": m.is_forcing_miss,
            "comment": _comment_for(m),
            "explain": _explain_move_zh(m),
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


def _md_summary(L: list, sm: dict) -> None:
    L.append(f"## 总评 · {sm['focus_name']}（{sm['side']}·{sm['result_text']}）")
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


def _md_critical(L: list, moments: list, heading: str = "## 关键时刻逐步解释") -> None:
    L.append(heading)
    if not moments:
        L.append("_没有超过阈值的问题着法。_")
    for m in moments:
        e = m["explain"]
        L.append(f"### 第 {m['move_number']} 回合 · {m['side']} · {m['class_zh']}"
                 f"（{m['phase']}，损失 -{m['cp_loss']}cp）")
        L.append(f"- 实走 `{m['played']}` → 应走 `{m['best']}`"
                 f"（{m['eval_before']} → {m['eval_after']}）")
        L.append(f"- **为什么错：** {e['why']}")
        L.append(f"- **造成的后果：** {e['consequence']}")
        L.append(f"- **应该怎么做：** {e['what_to_do']}")
        L.append(f"- [在 lichess 上打开这个局面]({m['lichess']})")
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

    if view["deviation"]:
        d = view["deviation"]
        book = f"（理论主线 {d['book']}）" if d["book"] else ""
        L.append(f"**开局脱谱：** 第 {d['move_number']} 回合 {d['side']} 走了 "
                 f"`{d['played']}`{book}。")
    else:
        L.append("**开局：** 全程在已知理论之内。")
    L.append("")

    # ---- stat cards for both sides (always) --------------------------------
    for label, s in [("白方", view["white_summary"]), ("黑方", view["black_summary"])]:
        L.append(f"## {label} — {s['name']}")
        L.append(f"- 平均每步损失（ACPL）：**{s['acpl']}** · 漏着 {s['blunders']} · "
                 f"错着 {s['mistakes']} · 不精确 {s['inaccuracies']}")
        L.append("- 分阶段（ACPL / 漏着 / 错着）：")
        for ph in PHASES:
            p = s["phases"][ph]
            L.append(f"    - {PHASE_ZH.get(ph, ph)}：{p['acpl']}（{p['moves']} 步），"
                     f"漏着 {p['blunders']} / 错着 {p['mistakes']}")
        if s["biggest"]:
            b = s["biggest"]
            L.append(f"- 最大失误：第 {b['move_number']} 回合 `{b['played']}` "
                     f"（应走 `{b['best']}`，损失 -{b['cp_loss']}cp）— [打开分析]({b['lichess']})")
        L.append("")

    if view.get("dual") and view.get("perspectives"):
        # ---- dual: full narrative + critical moments per side --------------
        for p in view["perspectives"]:
            L.append(f"# 【{p['side']}视角】{p['name']}")
            L.append("")
            _md_summary(L, p["summary_zh"])
            if p["turning_point"]:
                t = p["turning_point"]
                L.append("## 转折点")
                L.append(f"第 {t['move_number']} 回合 {t['side']} `{t['played']}`"
                         f"（{t['eval_before']} → {t['eval_after']}），{t['for']}此后再没回来；"
                         f"应走 `{t['best']}`。[打开分析]({t['lichess']})")
                L.append("")
            _md_critical(L, p["critical_moments"])
        return "\n".join(L)

    # ---- single focus mode --------------------------------------------------
    sm = view.get("summary_zh")
    if sm:
        _md_summary(L, sm)

    if view["turning_point"]:
        t = view["turning_point"]
        L.append("## 转折点")
        L.append(f"第 {t['move_number']} 回合 {t['side']} `{t['played']}`"
                 f"（{t['eval_before']} → {t['eval_after']}），{t['for']}此后再没回来；"
                 f"应走 `{t['best']}`。[打开分析]({t['lichess']})")
        L.append("")

    _md_critical(L, view["critical_moments"])
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
