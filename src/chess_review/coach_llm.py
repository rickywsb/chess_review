"""Optional LLM polishing of move explanations, *grounded* on engine facts.

The deterministic template in ``render.py`` is accurate but generic. This module
rephrases the explanation for the handful of moves that survive the significance
filter, using an LLM constrained to the engine-verified facts we pass in (played
move, engine best move, the principal variation, evals, and the reason tag). The
model is told to name the concrete idea (e.g. 限制对手王的逃跑, 织杀网) rather than
speak in generalities, and to never invent variations beyond the given PV.

Fail-soft by design: if no API key is configured or any call fails, the caller
keeps the template explanation. Only a few moves per game are sent, so cost and
latency stay small. Results are cached so a move shared across report views is
only sent once.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

_MODEL = os.environ.get("CHESS_REVIEW_LLM_MODEL", "gpt-4o-mini")

_SYSTEM = (
    "你是一位实力强劲、擅长讲解的国际象棋教练，用简体中文点评学生的对局。"
    "系统会给你一步棋的『引擎已核实的事实』：学生的实走着法、引擎的最佳着法、"
    "最佳着法之后的主变（principal variation，按 SAN 给出）、走子前后的评估分、"
    "以及这步棋为什么重要的标签。请据此写出具体、到位的点评。\n\n"
    "严格规则：\n"
    "1. 只能使用提供的事实和给出的主变，绝不可臆造其它变化、评估或棋子位置。\n"
    "2. 要讲出这步棋背后『具体的棋理』——比如：限制对手王的逃跑、织杀网、抢先手将军、"
    "多得子力、弃子抢攻、占据关键格/线、解除对方威胁——只要给出的『事实』或主变支持该说法。"
    "不要用『改善子力』『照顾局面需要』这类空泛套话。\n"
    "3. 只有在提供的事实明确支持时才下具体棋理结论；若事实不足以判断，就顺着主变"
    "逐步描述接下来会发生什么，绝不臆测或编造理由。\n"
    "4. 如果最佳着法能导向强制杀，请点明大致步数。\n"
    "4. 语气像教练一样鼓励、简洁：三段各 1-2 句短话即可。\n"
    "5. 只输出 JSON 对象，键为 why / consequence / what_to_do，值为中文字符串。\n"
    "   - why：为什么实走的着法不好、错过了什么关键点。\n"
    "   - consequence：造成的实际后果（结合评估变化）。\n"
    "   - what_to_do：应该怎么走、正解思路，以及一句可迁移的思考习惯。"
)


def _load_api_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()
    candidates = []
    override = os.environ.get("CHESS_REVIEW_OPENAI_KEY_FILE")
    if override:
        candidates.append(override)
    here = os.path.dirname(__file__)
    candidates.append(os.path.join(os.getcwd(), "chatgptapi.txt"))
    candidates.append(os.path.abspath(os.path.join(here, "..", "..", "chatgptapi.txt")))
    for path in candidates:
        try:
            if path and os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    val = fh.read().strip()
                if val:
                    return val
        except OSError:
            continue
    return None


@lru_cache(maxsize=1)
def _client():
    if os.environ.get("CHESS_REVIEW_LLM", "").strip() == "0":
        return None
    key = _load_api_key()
    if not key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=key)
    except Exception:
        return None


def available() -> bool:
    """True when an LLM client is configured and polishing is not disabled."""
    return _client() is not None


def _user_prompt(f: dict) -> str:
    lines = [
        f"阶段：{f.get('phase')}",
        f"走子方：{f.get('side')}，第 {f.get('move_number')} 回合",
        f"实走：{f.get('played')}",
        f"引擎最佳：{f.get('best')}",
        f"最佳着法之后的主变：{f.get('pv')}",
        f"走子前评估：{f.get('eval_before')}（{f.get('state_before')}）",
        f"走子后评估：{f.get('eval_after')}（{f.get('state_after')}）",
        f"单步损失：约 {f.get('cp_loss')} 厘兵（centipawn）",
        f"重要性标签：{f.get('reason_tag')}",
        f"最佳着法是否将军：{'是' if f.get('best_is_check') else '否'}",
        f"最佳着法是否吃子：{'是' if f.get('best_is_capture') else '否'}",
    ]
    mate = f.get("best_leads_to_mate_in")
    if mate:
        lines.append(f"最佳着法可导向强制杀：约 {mate} 步")
    board_facts = f.get("facts") or []
    if board_facts:
        lines.append("棋盘事实（务必据此判断，不得臆造）：")
        lines.extend(f"- {x}" for x in board_facts)
    return "\n".join(lines)


@lru_cache(maxsize=512)
def _polish_cached(payload_json: str) -> Optional[str]:
    client = _client()
    if client is None:
        return None
    facts = json.loads(payload_json)
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            temperature=0.4,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _user_prompt(facts)},
            ],
        )
        content = resp.choices[0].message.content or ""
    except Exception:
        return None
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not all(isinstance(data.get(k), str) and data.get(k) for k in
               ("why", "consequence", "what_to_do")):
        return None
    return json.dumps({k: data[k] for k in ("why", "consequence", "what_to_do")},
                      ensure_ascii=False)


def polish_explanation(context: dict) -> Optional[dict]:
    """Return an LLM-polished ``{why, consequence, what_to_do}`` dict, or None
    to signal the caller should keep the template explanation."""
    if _client() is None:
        return None
    payload = json.dumps(context, ensure_ascii=False, sort_keys=True)
    out = _polish_cached(payload)
    if not out:
        return None
    try:
        return json.loads(out)
    except (ValueError, TypeError):
        return None
