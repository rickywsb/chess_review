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
    "最佳着法之后的主变（正解思路）、实走之后对手的最强回应（错着的后果），"
    "走子前后的评估分，以及若干条按类别标注的『候选观察』。\n\n"
    "你的任务不是套模板，而是像教练一样先判断『这一步真正的问题是什么』，再讲给人听。\n\n"
    "严格规则：\n"
    "0. 系统已判定本步的『失误类型』和『走完之后的真实处境』，并给出『定调要求』——"
    "你必须严格据此定调。特别注意诚实框定：如果真实处境是『仍占优势/仍是胜势』，"
    "绝不能说『局面明显变差』或让学生以为要输了；反之亦然。\n"
    "1. 候选观察用【】标注了类别：【子力】丢子/得子、【战术】叉子/牵制/闪击等、"
    "【对手回应】错着之后对手怎么惩罚、【王的安全】王被攻击、【强制】将军或强制杀、"
    "【正解】最佳着法的好处。不同的错着原因不同——请你判断本步属于哪一类，"
    "只挑最能解释评估变化的 1-2 点来讲，其余不相关的观察一律不要提。\n"
    "2. 绝不要每步都套用同一种说法。尤其不要动不动就说『限制对手王的逃跑 / 安静的调整』——"
    "只有当【王的安全】或【强制】确实是本步主因时才谈王；若主因是丢子或战术，就讲子力和战术。\n"
    "3. 只能使用提供的事实、两条变化（正解主变、对手最强回应）里的着法，"
    "绝不可臆造其它变化、评估、棋子位置或战术。\n"
    "4. 讲后果时优先引用『对手最强回应』这条具体着法序列：对手接下来会怎么走、赢得什么，"
    "把因果说清楚，而不是空泛地说『局面变差』。\n"
    "5. 若候选观察不足以判断具体原因，就老实顺着变化描述接下来会发生什么，不要编造棋理。\n"
    "6. 语气像教练一样鼓励、简洁：三段各 1-2 句短话即可。\n"
    "7. 只输出 JSON 对象，键为 why / consequence / what_to_do，值为中文字符串。\n"
    "   - why：这步棋真正的问题是什么、错过了什么关键点（点明类别对应的具体棋理）。\n"
    "   - consequence：造成的实际后果（结合『对手最强回应』和评估变化）。\n"
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
        f"最佳着法之后的主变（正解思路）：{f.get('pv')}",
        f"实走之后对手的最强回应（错着的后果）：{f.get('refutation') or '（无）'}",
        f"走子前评估：{f.get('eval_before')}（{f.get('state_before')}）",
        f"走子后评估：{f.get('eval_after')}（{f.get('state_after')}）",
        f"单步损失：约 {f.get('cp_loss')} 厘兵（centipawn）",
        "",
        f"失误类型（已判定，必须据此定调）：{f.get('category_zh')}",
        f"走完之后的真实处境（务必如实框定，不得夸大或缩小）：{f.get('resulting_state')}",
        f"定调要求：{f.get('framing')}",
    ]
    if f.get("subtle"):
        lines.append("提示：这是一步细微的位置性选择，没有立刻的吃子或杀着。"
                     "请坦诚说明它只是小幅下滑，不要编造具体棋理或战术。")
    mate = f.get("best_leads_to_mate_in")
    if mate:
        lines.append(f"最佳着法可导向强制杀：约 {mate} 步")
    board_facts = f.get("facts") or []
    if board_facts:
        lines.append("")
        lines.append("候选观察（每条已核实，按【类别】标注；请只选最相关的 1-2 条来讲）：")
        lines.extend(f"- {x}" for x in board_facts)
    else:
        lines.append("（本步没有额外的战术/子力观察，请顺着上面的变化和处境如实解释。）")
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
