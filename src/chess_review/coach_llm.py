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

# Two-pass (judge -> write) is on by default; set to "0" for the single pass.
_TWO_PASS = os.environ.get("CHESS_REVIEW_LLM_TWO_PASS", "1").strip() != "0"

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
    "【线路】这步打开了某条竖线/横线/斜线，正好落在你自己的王或后上（对方车/象/后借线压制或牵制）——"
    "这往往比『丢一个兵』更关键，要讲清是线路而非那个兵、子的问题、"
    "【位置】没有丢子、但走坏了结构（坏象/弱格/王翼漏风/兵形弱点/子力变被动）、"
    "【选择】这里是唯一解还是有多手等价（用来拿捧语气轻重：唯一解难找、可多鼓励，多手等价则该提醒）、"
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

# ---- two-pass prompts -----------------------------------------------------
# Pass 1 (judge): decide the single core cause and pick the relevant facts.
_JUDGE_SYSTEM = (
    "你是国际象棋引擎结论的诊断器（不是写作者）。系统会给你一步棋的引擎已核实事实、"
    "两条变化（正解主变、对手最强回应）、评估分，以及若干按【类别】标注的候选观察，"
    "并已用规则预判了『失误类型』和『真实处境』。\n"
    "你的唯一职责：判断这步棋评估下降的『单一核心原因』，并从候选观察里挑出最能解释它的"
    "1-2 条。严格只依据给出的事实，绝不臆造变化、棋子或战术。\n"
    "注意区分：吃子之后被吃回只是兑换、不算丢子；仍占优势就不能当成要输。\n"
    "关键判据——子力与评估的落差：如果候选观察里的丢子很少（例如只有约 1 个兵、"
    "或只是兑换），可评估却下降很多（≈150cp 以上），那么核心原因几乎一定不是那个兵，"
    "而是位置/线路/王的安全——此时 use_facts 应以【线路】【位置】【王的安全】为主，"
    "不要再把【子力】或【对手回应】里『净得/净丢一个兵』一并列为核心。只有当丢子量"
    "与评估跌幅相称时，才把子力得失当作核心。\n"
    "primary 必须与你挑选的核心观察一致：选了【线路】就描述被打开的线路/王的暴露，"
    "选了【位置】就描述结构问题——绝不要写事实里没有出现的战术名词（如叉子、闪击）。\n"
    "只输出 JSON 对象，键：\n"
    "  primary：用一句话（不超过 20 字）概括核心问题，例如『错过强制得子』『王翼漏风』"
    "『把胜势走成均势』『忽视对手的叉子回击』。\n"
    "  use_facts：数组，逐字照抄你选中的 1-2 条候选观察原文；若确实没有合适观察就留空数组。\n"
    "  honest_state：照抄给定的『真实处境』。\n"
    "  avoid：一句话提醒写作时要避免的夸大或缩小（例如『仍是胜势，别说要输了』）。"
)

# Pass 2 (write): narrate strictly around the diagnosis pass 1 produced.
_WRITE_SYSTEM = _SYSTEM + (
    "\n\n【本次为写作环节】系统已完成诊断，并在用户消息末尾给出"
    "『核心主题 / 应使用的观察 / 真实处境 / 注意』。请严格围绕该诊断写作："
    "只讲这个主题、只用指定的观察，不要再另选其它候选观察；处境框定以诊断为准。\n"
    "『为什么错』这一段必须先点明核心主题：若主题是线路/王的暴露，就先说清是哪条线被"
    "打开、王或后如何被压制或将军，再把丢子当作这条线路带来的结果来讲——绝不能一上来"
    "只说『丢了一个兵/一个子』而把真正的原因（线路、王的安全）埋没或略去。"
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


def _judge_block(judge: dict) -> str:
    facts = judge.get("use_facts") or []
    return "\n".join([
        "",
        "【诊断（必须据此写作）】",
        f"核心主题：{judge.get('primary', '')}",
        "应使用的观察：" + ("；".join(facts) if facts else "（无，请顺着变化如实说明）"),
        f"真实处境：{judge.get('honest_state', '')}",
        f"注意：{judge.get('avoid', '')}",
    ])


@lru_cache(maxsize=512)
def _judge_cached(payload_json: str) -> Optional[str]:
    """Pass 1: diagnose the single core cause and select the relevant facts."""
    client = _client()
    if client is None:
        return None
    facts = json.loads(payload_json)
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
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
    primary = data.get("primary")
    honest = data.get("honest_state")
    use_facts = data.get("use_facts")
    if not isinstance(primary, str) or not primary:
        return None
    if not isinstance(use_facts, list):
        use_facts = []
    norm = {
        "primary": primary,
        "use_facts": [x for x in use_facts if isinstance(x, str) and x][:2],
        "honest_state": honest if isinstance(honest, str) else "",
        "avoid": data.get("avoid") if isinstance(data.get("avoid"), str) else "",
    }
    return json.dumps(norm, ensure_ascii=False, sort_keys=True)


@lru_cache(maxsize=512)
def _write_cached(payload_json: str, judge_json: str) -> Optional[str]:
    """Pass 2: write the coach explanation, constrained to the diagnosis."""
    client = _client()
    if client is None:
        return None
    facts = json.loads(payload_json)
    judge = json.loads(judge_json)
    user = _user_prompt(facts) + "\n" + _judge_block(judge)
    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            temperature=0.4,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _WRITE_SYSTEM},
                {"role": "user", "content": user},
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
    to signal the caller should keep the template explanation.

    Two-pass by default: a judge pass diagnoses the single core cause and picks
    the relevant facts, then a writer pass narrates strictly around it. Falls
    back to the single pass if the judge step is disabled or fails."""
    if _client() is None:
        return None
    payload = json.dumps(context, ensure_ascii=False, sort_keys=True)
    out = None
    if _TWO_PASS:
        judge = _judge_cached(payload)
        if judge:
            out = _write_cached(payload, judge)
    if not out:
        out = _polish_cached(payload)
    if not out:
        return None
    try:
        return json.loads(out)
    except (ValueError, TypeError):
        return None
