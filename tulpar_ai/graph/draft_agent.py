"""The draft step as a tool-using agent: the model decides what to look up, checks its own draft, then submits it.

  ┌─► LLM step ─► get_client_context | search_exercises | check_plan ─► tool ─► observation ─┐
  └─────────────────────────────────────────────────────────────────────────────────────────┘
      LLM step ─► final ─► check_plan ─┬─► ok, or the last step ─► draft (to the graph's validate node)
                                       └─► E_* errors and steps left ─► observation, next step

Why a JSON protocol and not native tool calling: the fallback chain mixes Ollama Cloud and Groq models, and all of
them have a JSON mode while tool-calling support differs from model to model. One protocol, any model of the chain.

Bounded: at most `draft_agent_max_steps` tool steps, then one more call that must return the final draft. The
graph's validate → draft loop (≤3 drafts) and the trainer's review stay outside and unchanged: the agent's own
checks only make its first draft better, the validate node still decides.
"""

from __future__ import annotations

import json
import os

from ..config import get_settings
from ..llm import LLMError, chat
from ..pii import mask
from ..prompts import prompt
from ..skill import instructions
from ..tools import check_plan, get_client_context, normalize_ops, search_exercises

AGENT_PROMPT = "v3"  # draft.v3.md: the agent protocol; draft.v2.md stays the prompt of DRAFT_MODE=candidates
TOOLS = ("get_client_context", "search_exercises", "check_plan")
MAX_BATCH = 3  # tool calls in one step ({"actions": [...]})
OBS_CHARS = 4000  # one observation in the next prompt


def system_prompt() -> str:
    version = os.environ.get("PROMPT_DRAFT_AGENT") or AGENT_PROMPT
    return prompt("draft", version) + "\n\n# Skill: tulpar-program-builder\n" + instructions()


def plan_text(plan: dict, equipment: bool = False) -> str:
    """The plan as the model sees it. The agent also gets each exercise's equipment: it has no candidate list
    pre-filtered by the client's place, so it must see which rows need a barbell or a machine."""
    lines = [f"План «{plan['title']}»"]
    for d in plan["days"]:
        lines.append(f"День day_index={d['day_index']} «{d['title']}»:")
        for e in d["exercises"]:
            lines.append(f"  - wex_id={e['id']} | {e['exercise_name']} | {e.get('muscle_group')} | "
                         f"{e.get('target_sets')}×{e.get('target_reps')}" + (f" | {e.get('equipment') or '—'}" if equipment else ""))
    return "\n".join(lines)


def _head(state: dict) -> str:
    client = state.get("client") or {}
    note = client.get("trainer_note") or {}
    has_note = "есть — подробности в get_client_context" if (note.get("body") or note.get("injury")) else "нет"
    parts = [
        f"Запрос ({'клиент' if state.get('source') == 'client' else 'тренер'}): {mask(state['request'])}",
        f"Клиент: уровень {client.get('level') or '—'}, место тренировок {client.get('place') or '—'}, "
        f"заметка тренера об ограничениях: {has_note}.",
        plan_text(state["plan"], equipment=True),
    ]
    errors = [v for v in state.get("violations", []) if v["severity"] == "error"]
    if errors:
        parts.append("Предыдущий черновик не прошёл проверку:\n" + "\n".join(f"- {v['code']}: {v['message']}" for v in errors))
    if state.get("feedback"):
        parts.append(f"Комментарий тренера к прошлому черновику: {mask(state['feedback'])}")
    return "\n\n".join(parts)


def _user(head: str, steps: list[str], step: int, max_steps: int) -> str:
    body = [head, "Шаги:\n" + ("\n".join(steps) if steps else "(пока нет)")]
    if step > max_steps:
        body.append('Это последний шаг: инструменты больше недоступны. Верни {"action": "final", ...} с лучшими '
                    "операциями, которые у тебя есть (по возможности уже проверенными).")
    else:
        body.append(f"Шаг {step} из {max_steps}. Верни JSON с действием.")
    return "\n\n".join(body)


def _dump(x) -> str:
    s = json.dumps(x, ensure_ascii=False)
    return s if len(s) <= OBS_CHARS else s[:OBS_CHARS] + "…(обрезано)"


def _name(a: dict) -> str:
    n = str(a.get("action") or a.get("tool") or a.get("name") or "").strip().lower()
    return n or ("final" if "ops" in a else "")


def _args(a: dict) -> dict:
    args = a.get("args") if isinstance(a.get("args"), dict) else {}
    top = {k: v for k, v in a.items() if k not in ("action", "tool", "name", "args")}
    return {**top, **args}


def parse_reply(data: dict) -> tuple[list[tuple[str, dict]], dict | None, str | None]:
    """(tool calls, final draft, protocol error). A reply without "action" but with "ops" is a final draft."""
    raw = data.get("actions") if isinstance(data.get("actions"), list) else [data]
    calls: list[tuple[str, dict]] = []
    for a in raw:
        if not isinstance(a, dict):
            return [], None, "каждое действие должно быть JSON-объектом"
        name = _name(a)
        if name == "final":
            return [], _args(a), None
        if name not in TOOLS:
            return [], None, (f"неизвестное действие {name!r}; допустимые: {', '.join(TOOLS)}, final" if name
                              else 'нет поля "action"')
        calls.append((name, _args(a)))
    if not calls:
        return [], None, "пустой список actions"
    return calls[:MAX_BATCH], None, None


def _search_rows(rows: list[dict]) -> str:
    if not rows:
        return "ничего не найдено — ослабь фильтры"
    return "exercise_id | название | группа | инвентарь | базовое\n" + "\n".join(
        f"{r['exercise_id']} | {r['name']} | {r['muscle_group']} | {r['equipment'] or '—'} | "
        f"{'да' if r['compound'] else 'нет'}" for r in rows)


async def _run_tool(name: str, args: dict, state: dict) -> tuple[str, dict]:
    """(observation for the next prompt, log row). A failing tool is an observation, not a crash."""
    try:
        if name == "search_exercises":
            q = {k: str(args.get(k) or "") for k in ("query", "muscle_group", "equipment")}
            rows = search_exercises(**q, client=state["client"])
            return _search_rows(rows), {"tool": name, "args": q, "found": len(rows)}
        if name == "get_client_context":
            ctx = await get_client_context(state["client_id"])
            return _dump(ctx), {"tool": name}
        ops, bad = normalize_ops(args.get("ops"))
        res = check_plan(state["plan"], ops, state["client"])
        if bad:
            res = {**res, "ok": False, "unreadable": bad}
        return _dump(res), {"tool": name, "ops": len(ops), "ok": res["ok"], "errors": [e["code"] for e in res["errors"]]}
    except Exception as e:  # noqa: BLE001
        return f"ошибка инструмента: {type(e).__name__}: {str(e)[:200]}", {"tool": name, "error": type(e).__name__}


def _draft(final: dict, ops: list[dict]) -> dict:
    return {"summary": str(final.get("summary") or ""), "rationale": str(final.get("rationale") or ""), "ops": ops}


async def run(state: dict) -> tuple[dict, list[dict]]:
    """One draft. Returns (draft {summary, rationale, ops}, log of steps for evals and the trace metadata)."""
    s = get_settings()
    max_steps = max(1, s.draft_agent_max_steps)
    system, head = system_prompt(), _head(state)
    steps: list[str] = []
    log: list[dict] = []
    checked: list[dict] | None = None  # ops the validator last passed — the fallback if no final comes
    for step in range(1, max_steps + 2):
        try:
            res = await chat("text", system, _user(head, steps, step, max_steps), json_mode=True,
                             temperature=s.draft_temperature, top_p=s.draft_top_p, max_tokens=s.draft_max_tokens)
        except LLMError as e:
            log.append({"step": step, "action": "llm_error"})
            return {"summary": "Модель недоступна, черновик не собран.", "rationale": str(e)[:200], "ops": []}, log
        data = res.data if isinstance(res.data, dict) else {}
        calls, final, problem = parse_reply(data) if data else ([], None, "ответ должен быть одним JSON-объектом")
        if final is not None:
            ops, bad = normalize_ops(final.get("ops"))
            if step <= max_steps and (ops or bad):
                chk = check_plan(state["plan"], ops, state["client"])  # the final is checked before it leaves
                log.append({"step": step, "action": "final", "ops": len(ops), "ok": chk["ok"] and not bad,
                            "errors": [e["code"] for e in chk["errors"]], "auto_check": True})
                if not chk["ok"] or bad:
                    steps.append(f"Шаг {step}: final {_dump({'ops': ops})}\nРезультат: черновик НЕ принят, проверка "
                                 "нашла ошибки — исправь и снова верни final. " + _dump({**chk, "unreadable": bad} if bad else chk))
                    continue
            else:
                log.append({"step": step, "action": "final", "ops": len(ops)})
            return _draft(final, ops), log
        if step > max_steps:
            log.append({"step": step, "action": "no_final"})
            break
        if problem:
            log.append({"step": step, "action": "invalid", "error": problem})
            steps.append(f"Шаг {step}: {(res.text or '').strip()[:300]}\nРезультат: ошибка протокола — {problem}.")
            continue
        for name, args in calls:
            obs, row = await _run_tool(name, args, state)
            log.append({"step": step, "action": name, **row})
            if name == "check_plan" and row.get("ok"):
                checked = normalize_ops(args.get("ops"))[0]
            steps.append(f"Шаг {step}: {name}{' ' + _dump(args) if args else ''}\nРезультат: {obs}")
    if checked:
        return {"summary": "Черновик по последним проверенным операциям агента.",
                "rationale": "Агент не вернул итоговый черновик за отведённые шаги; взяты операции, прошедшие проверку.",
                "ops": checked}, log
    return {"summary": "Агент не собрал черновик за отведённые шаги.", "rationale": "", "ops": []}, log


def stats(log: list[dict]) -> dict:
    """Numbers for the trace metadata and the eval: LLM steps, tool calls by name, self-corrections."""
    tools = [r["tool"] for r in log if r.get("tool")] + ["check_plan" for r in log if r.get("auto_check")]
    return {"llm_steps": len({r["step"] for r in log}), "tool_calls": len(tools),
            "tools": {t: tools.count(t) for t in sorted(set(tools))},
            "rejected_finals": sum(1 for r in log if r.get("auto_check") and not r.get("ok")),
            "protocol_errors": sum(1 for r in log if r.get("action") == "invalid")}
