"""Aggregate many game analyses into a player tracking report."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, median
from typing import Optional

import chess

from .classify import BLUNDER, MISTAKE
from .models import GameAnalysis, MoveAnalysis

PHASES = ["opening", "middlegame", "endgame"]

# Eval buckets (target's perspective, centipawns).
WINNING = 200
SLIGHT = 70

MOVE_BUCKETS = [
    ("1-10", 1, 10),
    ("11-20", 11, 20),
    ("21-30", 21, 30),
    ("31-40", 31, 40),
    ("41-50", 41, 50),
    ("51-60", 51, 60),
    ("61+", 61, 10_000),
]

PHASE_ZH = {
    "opening": "开局",
    "middlegame": "中局",
    "endgame": "残局",
}


def _pov(eval_white: int, color: bool) -> int:
    return eval_white if color == chess.WHITE else -eval_white


def _rating(headers: dict, color: bool) -> Optional[int]:
    key = "WhiteElo" if color == chess.WHITE else "BlackElo"
    val = headers.get(key, "").strip()
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _bucket_eval(cp: int) -> str:
    if cp >= WINNING:
        return "winning"
    if cp >= SLIGHT:
        return "slightly_better"
    if cp > -SLIGHT:
        return "equal"
    if cp > -WINNING:
        return "slightly_worse"
    return "losing"


@dataclass
class GameContext:
    """Per-game facts about the target player, used to build the report."""

    game: GameAnalysis
    color: bool
    opponent: str
    opponent_rating: Optional[int]
    score: Optional[float]         # 1/0.5/0/None
    moves: list[MoveAnalysis]
    peak_eval: int                 # best target-pov eval reached
    low_eval: int                  # worst target-pov eval reached
    endgame_entry_eval: Optional[int]
    reached_endgame: bool


def _name_key(name: str) -> str:
    return " ".join(name.casefold().split())


def _build_context(ga: GameAnalysis, player_names: set[str]) -> Optional[GameContext]:
    white_key = _name_key(ga.white)
    black_key = _name_key(ga.black)
    if white_key in player_names:
        color = chess.WHITE
    elif black_key in player_names:
        color = chess.BLACK
    else:
        color = None
    if color is None:
        return None
    opponent = ga.black if color == chess.WHITE else ga.white
    traj = ga.eval_trajectory_white()
    povs = [_pov(v, color) for v in traj]
    peak = max(povs) if povs else 0
    low = min(povs) if povs else 0

    endgame_entry = None
    reached_endgame = False
    for m in ga.moves:
        if m.phase == "endgame":
            reached_endgame = True
            endgame_entry = _pov(m.eval_before_white, color)
            break

    return GameContext(
        game=ga,
        color=color,
        opponent=opponent,
        opponent_rating=_rating(ga.headers, not color),
        score=ga.result_score(color),
        moves=ga.player_moves(color),
        peak_eval=peak,
        low_eval=low,
        endgame_entry_eval=endgame_entry,
        reached_endgame=reached_endgame,
    )


def _rate(hits: int, total: int) -> float:
    return (hits / total) if total else 0.0


def _wdl(contexts: list[GameContext]) -> tuple[int, int, int, float]:
    w = sum(1 for c in contexts if c.score == 1.0)
    d = sum(1 for c in contexts if c.score == 0.5)
    l = sum(1 for c in contexts if c.score == 0.0)
    total = w + d + l
    score_pct = ((w + 0.5 * d) / total) if total else 0.0
    return w, d, l, score_pct


def _window_metrics(contexts: list[GameContext]) -> dict:
    moves = [move for context in contexts for move in context.moves]
    blunders = sum(1 for move in moves if move.cp_loss >= BLUNDER)
    clean_games = sum(
        1 for context in contexts
        if all(move.cp_loss < BLUNDER for move in context.moves)
    )
    wins, draws, losses, score_pct = _wdl(contexts)
    winning_games = [
        context for context in contexts
        if context.peak_eval >= WINNING and context.score is not None
    ]
    converted = sum(1 for context in winning_games if context.score == 1.0)
    return {
        "games": len(contexts),
        "moves": len(moves),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "acpl_median": round(median(move.cp_loss for move in moves), 1) if moves else None,
        "blunders_per_100": round(100 * _rate(blunders, len(moves)), 1),
        "clean_game_rate": _rate(clean_games, len(contexts)),
        "score_pct": score_pct,
        "conversion_n": len(winning_games),
        "conversion_win_rate": (
            _rate(converted, len(winning_games)) if winning_games else None),
        "first_date": contexts[0].game.date if contexts else None,
        "last_date": contexts[-1].game.date if contexts else None,
    }


def _metric_trend(baseline: dict, recent: dict, key: str, *,
                  lower_is_better: bool, threshold: float) -> dict:
    before = baseline.get(key)
    after = recent.get(key)
    if before is None or after is None:
        return {"status": "insufficient_data", "before": before,
                "after": after, "change": None}
    change = round(after - before, 3)
    improvement = -change if lower_is_better else change
    if improvement >= threshold:
        status = "improving"
    elif improvement <= -threshold:
        status = "declining"
    else:
        status = "stable"
    return {"status": status, "before": before, "after": after,
            "change": change}


def _progress(contexts: list[GameContext]) -> dict:
    dated_contexts = []
    for index, context in enumerate(contexts):
        parsed = None
        for pattern in ("%Y.%m.%d", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(context.game.date, pattern).date()
                break
            except ValueError:
                continue
        if parsed is not None:
            dated_contexts.append((parsed, index, context))
    chronological = [
        context for _, _, context in sorted(dated_contexts)
    ]
    window = min(20, len(chronological) // 2)
    if window < 10:
        return {
            "status": "insufficient_data",
            "confidence": "insufficient",
            "window_games": window,
            "required_games": 20,
            "available_games": len(chronological),
            "excluded_undated": len(contexts) - len(chronological),
            "metrics": {},
        }
    baseline = _window_metrics(chronological[-2 * window:-window])
    recent = _window_metrics(chronological[-window:])
    metrics = {
        "acpl_median": _metric_trend(
            baseline, recent, "acpl_median", lower_is_better=True, threshold=5.0),
        "blunders_per_100": _metric_trend(
            baseline, recent, "blunders_per_100",
            lower_is_better=True, threshold=1.0),
        "clean_game_rate": _metric_trend(
            baseline, recent, "clean_game_rate",
            lower_is_better=False, threshold=0.08),
        "score_pct": _metric_trend(
            baseline, recent, "score_pct",
            lower_is_better=False, threshold=0.08),
        "conversion_win_rate": _metric_trend(
            baseline, recent, "conversion_win_rate",
            lower_is_better=False, threshold=0.10),
    }
    directions = {
        metric["status"] for metric in metrics.values()
        if metric["status"] in ("improving", "declining")
    }
    if directions == {"improving"}:
        status = "improving"
    elif directions == {"declining"}:
        status = "declining"
    elif directions:
        status = "mixed"
    else:
        status = "stable"
    return {
        "status": status,
        "confidence": "high" if window >= 20 else "medium",
        "window_games": window,
        "required_games": 20,
        "available_games": len(chronological),
        "excluded_undated": len(contexts) - len(chronological),
        "baseline": baseline,
        "recent": recent,
        "metrics": metrics,
    }


def _coach_diagnosis(contexts: list[GameContext], report: dict) -> dict:
    """Build cautious, evidence-backed strengths and training priorities."""
    if len(contexts) < 10:
        return {
            "status": "insufficient_data",
            "required_games": 10,
            "available_games": len(contexts),
            "strengths": [],
            "weaknesses": [],
            "priorities": [],
        }

    strengths = []
    weaknesses = []
    priorities = []
    eligible_phases = [row for row in report["phases"] if row["moves"] >= 30]
    if len(eligible_phases) >= 2:
        ordered = sorted(eligible_phases, key=lambda row: row["avg_loss"])
        strongest = ordered[0]
        weakest = ordered[-1]
        if ordered[1]["avg_loss"] - strongest["avg_loss"] >= 5:
            strengths.append({
                "key": f"phase_{strongest['phase']}",
                "title": f"{PHASE_ZH[strongest['phase']]}相对稳定",
                "evidence": (
                    f"平均每步损失 {strongest['avg_loss']}cp，"
                    f"严重失误率 {strongest['blunder_rate'] * 100:.1f}%，"
                    f"样本 {strongest['moves']} 步。"
                ),
                "confidence": "high" if strongest["moves"] >= 100 else "medium",
            })
        if weakest["avg_loss"] - ordered[-2]["avg_loss"] >= 5:
            phase_zh = PHASE_ZH[weakest["phase"]]
            weaknesses.append({
                "key": f"phase_{weakest['phase']}",
                "title": f"{phase_zh}损失最集中",
                "evidence": (
                    f"平均每步损失 {weakest['avg_loss']}cp，"
                    f"错着率 {weakest['mistake_rate'] * 100:.1f}%，"
                    f"样本 {weakest['moves']} 步。"
                ),
                "confidence": "high" if weakest["moves"] >= 100 else "medium",
            })
            priorities.append({
                "key": f"phase_{weakest['phase']}",
                "title": f"优先改善{phase_zh}决策",
                "action": "复盘该阶段损失最大的局面，按候选着、对手威胁、计算验证三步记录思路。",
            })

    conversion = report["conversion"]
    if conversion["n"] >= 5:
        item = {
            "key": "conversion",
            "title": "优势转化可靠" if conversion["win_rate"] >= 0.75 else "优势转化不足",
            "evidence": (
                f"取得至少 +2 优势的 {conversion['n']} 盘中，"
                f"赢下 {conversion['wins']} 盘，胜率 {conversion['win_rate'] * 100:.0f}%。"
            ),
            "confidence": "high" if conversion["n"] >= 10 else "medium",
        }
        if conversion["win_rate"] >= 0.75:
            strengths.append(item)
        elif conversion["win_rate"] < 0.5:
            weaknesses.append(item)
            priorities.append({
                "key": "conversion",
                "title": "训练优势局面的收束",
                "action": "针对 +2 以上未获胜对局，检查强制着、兑子选择和对手反击资源。",
            })

    resilience = report["resilience"]
    if resilience["n"] >= 5:
        item = {
            "key": "resilience",
            "title": "逆风韧性较强" if resilience["save_rate"] >= 0.35 else "逆风救回率偏低",
            "evidence": (
                f"曾落后至少 -2 的 {resilience['n']} 盘中，"
                f"救回 {resilience['saved']} 盘，挽救率 {resilience['save_rate'] * 100:.0f}%。"
            ),
            "confidence": "high" if resilience["n"] >= 10 else "medium",
        }
        if resilience["save_rate"] >= 0.35:
            strengths.append(item)
        elif resilience["save_rate"] < 0.15:
            weaknesses.append(item)
            priorities.append({
                "key": "resilience",
                "title": "增加困难局面的防守训练",
                "action": "从劣势局面练习寻找交换、堡垒、永将和制造实际难题的机会。",
            })

    moves = [move for context in contexts for move in context.moves]
    forcing_misses = [
        move for move in moves
        if move.cp_loss >= MISTAKE and move.is_forcing_miss
    ]
    forcing_rate = 100 * _rate(len(forcing_misses), len(moves))
    if len(forcing_misses) >= 3 and forcing_rate >= 1.0:
        weaknesses.append({
            "key": "forcing_moves",
            "title": "强制着扫描需要加强",
            "evidence": (
                f"共漏掉 {len(forcing_misses)} 次关键将军或吃子，"
                f"每百步 {forcing_rate:.1f} 次。"
            ),
            "confidence": "high" if len(forcing_misses) >= 6 else "medium",
        })
        priorities.append({
            "key": "forcing_moves",
            "title": "建立每步强制着检查习惯",
            "action": "落子前固定扫描双方的将军、吃子和直接威胁，再进入候选着计算。",
        })

    return {
        "status": "ready",
        "required_games": 10,
        "available_games": len(contexts),
        "strengths": strengths[:3],
        "weaknesses": weaknesses[:3],
        "priorities": priorities[:3],
    }


def build_player_report(analyses: list[GameAnalysis], player: str,
                        aliases: Optional[list[str]] = None) -> dict:
    player_names = {_name_key(name) for name in (aliases or [player]) if name.strip()}
    contexts = [
        context for context in (
            _build_context(analysis, player_names) for analysis in analyses)
        if context is not None
    ]
    report: dict = {"player": player, "n_games_total": len(analyses), "n_games": len(contexts)}
    if not contexts:
        report["error"] = f"No games found for player '{player}'."
        return report

    all_moves = [m for c in contexts for m in c.moves]

    # ---- headline / results -------------------------------------------------
    w, d, l, score_pct = _wdl(contexts)
    report["record"] = {"wins": w, "draws": d, "losses": l, "score_pct": score_pct}
    report["as_white"] = sum(1 for c in contexts if c.color == chess.WHITE)
    report["as_black"] = sum(1 for c in contexts if c.color == chess.BLACK)
    ratings = [c.opponent_rating for c in contexts if c.opponent_rating]
    report["avg_opponent"] = round(mean(ratings)) if ratings else None
    report["acpl"] = round(mean(m.cp_loss for m in all_moves), 1) if all_moves else 0.0
    report["blunders_total"] = sum(1 for m in all_moves if m.cp_loss >= BLUNDER)
    report["mistakes_total"] = sum(1 for m in all_moves if m.cp_loss >= MISTAKE)
    report["progress"] = _progress(contexts)

    # ---- phase table --------------------------------------------------------
    phase_rows = []
    for phase in PHASES:
        pm = [m for m in all_moves if m.phase == phase]
        n = len(pm)
        phase_rows.append({
            "phase": phase,
            "moves": n,
            "avg_loss": round(mean(m.cp_loss for m in pm), 1) if pm else 0.0,
            "blunder_rate": _rate(sum(1 for m in pm if m.cp_loss >= BLUNDER), n),
            "mistake_rate": _rate(sum(1 for m in pm if m.cp_loss >= MISTAKE), n),
        })
    report["phases"] = phase_rows

    # ---- middlegame failure by move-number bucket ---------------------------
    mid = [m for m in all_moves if m.phase == "middlegame"]
    bucket_rows = []
    for label, lo, hi in MOVE_BUCKETS:
        bm = [m for m in mid if lo <= m.move_number <= hi]
        n = len(bm)
        bucket_rows.append({
            "bucket": label,
            "moves": n,
            "failure_rate": _rate(sum(1 for m in bm if m.cp_loss >= MISTAKE), n),
        })
    report["middlegame_by_move"] = bucket_rows

    # ---- conversion of winning positions (peak >= +2) -----------------------
    winning_games = [c for c in contexts if c.peak_eval >= WINNING and c.score is not None]
    cw, cd, cl, cscore = _wdl(winning_games)
    report["conversion"] = {
        "n": len(winning_games),
        "wins": cw, "draws": cd, "losses": cl,
        "win_rate": _rate(cw, len(winning_games)),
        "score_pct": cscore,
    }
    strong = [c for c in winning_games if c.opponent_rating and c.opponent_rating >= 2000]
    weak = [c for c in winning_games if c.opponent_rating and c.opponent_rating < 2000]
    report["conversion_vs_2000plus"] = _rate(sum(1 for c in strong if c.score == 1.0), len(strong)) if strong else None
    report["conversion_vs_sub2000"] = _rate(sum(1 for c in weak if c.score == 1.0), len(weak)) if weak else None

    # ---- endgame entry buckets ---------------------------------------------
    eg_rows = []
    eg_games = [c for c in contexts if c.reached_endgame and c.endgame_entry_eval is not None]
    for key, label in [
        ("winning", "winning (>= +2)"),
        ("slightly_better", "slightly better (+0.7..+2)"),
        ("equal", "equal (+-0.7)"),
        ("slightly_worse", "slightly worse (-2..-0.7)"),
        ("losing", "losing (<= -2)"),
    ]:
        grp = [c for c in eg_games if _bucket_eval(c.endgame_entry_eval) == key]
        gw, gd, gl, gscore = _wdl(grp)
        eg_rows.append({
            "bucket": key, "label": label, "n": len(grp),
            "wins": gw, "draws": gd, "losses": gl, "score_pct": gscore,
        })
    report["endgame_entry"] = eg_rows

    # ---- resilience when behind (low <= -2) --------------------------------
    behind = [c for c in contexts if c.low_eval <= -WINNING and c.score is not None]
    saved = sum(1 for c in behind if c.score and c.score > 0.0)
    report["resilience"] = {
        "n": len(behind),
        "saved": saved,
        "save_rate": _rate(saved, len(behind)),
    }

    # ---- top blunders -------------------------------------------------------
    ranked = sorted(
        ((m, c) for c in contexts for m in c.moves if m.cp_loss >= BLUNDER),
        key=lambda pair: pair[0].cp_loss,
        reverse=True,
    )
    report["top_blunders"] = [
        {
            "date": c.game.date,
            "opponent": c.opponent,
            "opponent_rating": c.opponent_rating,
            "color": "White" if c.color == chess.WHITE else "Black",
            "move_number": m.move_number,
            "phase": m.phase,
            "played": m.san,
            "best": m.best_move_san,
            "cp_loss": m.cp_loss,
            "fen": m.fen_before,
        }
        for m, c in ranked[:20]
    ]

    report["diagnosis"] = _coach_diagnosis(contexts, report)

    return report
