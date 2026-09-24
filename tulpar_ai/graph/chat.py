"""Chat graph: one client turn. No human pause here — the long-lived pause lives in the program graph.

  ingest ─► precheck ─► route ─┬─► meal_photo ─────────────────────────────► END
                               ├─► meal_text ──────────────────────────────► END
                               ├─► retrieve ─┬─► answer ─┬────────────────────► END
                               │      ▲      │           └─► escalate ─────► END
                               │      └ rewrite (≤2) ◄──┘ (not enough)
                               ├─► program_request (starts the program graph) ► END
                               ├─► escalate ───────────────────────────────► END
                               ├─► refuse (prompt injection) ──────────────► END
                               └─► other ──────────────────────────────────► END
"""

from __future__ import annotations

import base64
import re
from datetime import date
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .. import notify, stt
from ..config import get_settings
from ..gateway import get_gateway
from ..gateway.base import MATCH_THRESHOLD, MealItem
from ..llm import LLMError, chat, json_call
from ..pii import mask
from ..prompts import prompt
from ..rag.retrieve import retrieve
from ..store import get_store

INTENTS = {"meal_text", "question", "program_request", "escalate", "other"}

# HARD markers force a human regardless of the model; SOFT markers are only a hint for the router.
HARD = re.compile(r"стероид|анабол|тестостерон|рвот|не ем(?:\s+уже)?\s+\d+\s*(?:дн|день|дня|дней|сут)|не ела?\s+\d+\s*(?:дн|день|дня|дней|сут)|"
                  r"голодаю|обморок|(?:по)?теря\w*\s+сознан|суицид|беремен|жүкті|кровь|кровотеч|давит в груди|боль в сердц", re.I)
SOFT = re.compile(r"\bбол(?:ит|ят|ь|ью|и|ела|ело|ел)\b|травм|хруст|\bот[её]к|\bнемеет|\bонемен|таблет|лекарств|препарат|ауырады", re.I)
INJECTION = re.compile(r"игнорируй (все )?(инструкц|правил)|ignore (all |previous )?instructions|системн\w* промпт|"
                       r"system prompt|покажи (телефон|номер|данные) клиент|you are now|ты теперь", re.I)
GRAMS = re.compile(r"(\d{2,4})\s*(?:г|гр|грамм\w*)\b", re.I)
DEFAULT_GRAMS = 150.0


class ChatState(TypedDict, total=False):
    client_id: str
    turn_id: str
    text: str
    image_path: str | None
    audio_path: str | None
    transcript: str | None
    flags: dict
    intent: str
    red_flag: bool
    reason: str
    query: str
    rewrites: int
    hits: list[dict]
    sufficient: bool
    reply: str
    kind: str
    citations: list[dict]
    meal: dict | None
    proposal_id: str | None
    escalation_id: str | None


def _text(state: ChatState) -> str:
    return " ".join(x for x in (state.get("text"), state.get("transcript")) if x).strip()


async def ingest(state: ChatState) -> dict:
    if state.get("audio_path"):
        p = Path(state["audio_path"])
        return {"transcript": await stt.transcribe(p.read_bytes(), filename=p.name)}
    return {}


async def precheck(state: ChatState) -> dict:
    t = _text(state)
    return {"flags": {"hard": bool(HARD.search(t)), "soft": bool(SOFT.search(t)), "injection": bool(INJECTION.search(t))}}


def _heuristic_intent(t: str, flags: dict) -> str:
    low = t.lower()
    if flags.get("hard"):
        return "escalate"
    if GRAMS.search(low) or re.search(r"\b(съел|съела|поел|поела|выпил|выпила|на (завтрак|обед|ужин))\b", low):
        return "meal_text"
    if re.search(r"замен|программ|план|добав\w* (день|кардио)|облегч", low):
        return "program_request"
    if flags.get("soft"):
        return "escalate"
    if re.search(r"^(привет|спасибо|здравствуй|салем|сәлем|ок|окей)\b", low):
        return "other"
    return "question"


async def route(state: ChatState) -> dict:
    flags, t = state.get("flags", {}), _text(state)
    if flags.get("injection"):
        return {"intent": "refuse", "reason": "prompt-injection pattern"}
    if state.get("image_path") and not flags.get("hard"):
        return {"intent": "meal_photo", "reason": "photo"}
    if not t:
        return {"intent": "other", "reason": "empty"}
    s = get_settings()
    hint = "\n\n(В сообщении есть слова-маркеры боли или лекарств — проверь внимательно.)" if flags.get("soft") else ""
    try:
        data, _ = await json_call("route", prompt("route"), mask(t) + hint,
                                  temperature=s.route_temperature, max_tokens=s.route_max_tokens)
        intent = data.get("intent") if data.get("intent") in INTENTS else "question"
        red = bool(data.get("red_flag")) or intent == "escalate"
        reason = str(data.get("reason") or "")[:200]
    except LLMError as e:  # no provider reachable → deterministic fallback, still safe
        intent = _heuristic_intent(t, flags)
        red, reason = intent == "escalate", f"heuristic ({type(e).__name__})"
    if flags.get("hard"):
        intent, red, reason = "escalate", True, (reason + "; hard red-flag marker").strip("; ")
    return {"intent": "escalate" if red else intent, "red_flag": red, "reason": reason}


def after_route(state: ChatState) -> str:
    return {"question": "retrieve"}.get(state["intent"], state["intent"])


# ── question branch: retrieve → (rewrite ≤2) → answer ───────────────────────
async def retrieve_node(state: ChatState) -> dict:
    s = get_settings()
    q = state.get("query") or _text(state)
    hits = await retrieve(q)
    top = hits[0]["rerank_score"] if hits else 0.0
    return {"query": q, "hits": hits, "sufficient": top >= s.rag_min_score}


def after_retrieve(state: ChatState) -> str:
    if state.get("sufficient"):
        return "answer"
    if state.get("rewrites", 0) < get_settings().rag_max_rewrites:
        return "rewrite"
    return "escalate"


async def rewrite(state: ChatState) -> dict:
    try:
        data, _ = await json_call("route", prompt("rewrite"), mask(_text(state)), temperature=0.0, max_tokens=120)
        q = str(data.get("query") or _text(state))
    except LLMError:
        q = _text(state)
    return {"query": q, "rewrites": state.get("rewrites", 0) + 1}


def _sources(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        where = h["title"] + (f", стр. {h['page']}" if h.get("page") else "")
        lines.append(f"[{i}] {where}\n{h['text']}")
    return "\n\n".join(lines)


async def answer(state: ChatState) -> dict:
    s = get_settings()
    hits = state.get("hits", [])
    user = f"Вопрос клиента: {mask(_text(state))}\n\nИсточники:\n{_sources(hits)}"
    try:
        data, _ = await json_call("text", prompt("answer"), user, temperature=s.answer_temperature,
                                  top_p=s.answer_top_p, max_tokens=s.answer_max_tokens)
    except LLMError:
        return {"sufficient": False, "reason": "answer model unavailable"}
    used = [int(n) for n in data.get("citations", []) if str(n).isdigit() and 1 <= int(n) <= len(hits)]
    cites = [{"n": n, "title": hits[n - 1]["title"], "page": hits[n - 1].get("page"), "source": hits[n - 1]["source"]}
             for n in used]
    if not data.get("sufficient", True):
        return {"sufficient": False, "reason": "model: sources do not answer"}
    disclaimer = "\n\nЭто общая информация, а не медицинская консультация."
    return {"reply": str(data.get("answer", "")).strip() + disclaimer, "citations": cites, "kind": "answer"}


def after_answer(state: ChatState) -> str:
    return END if state.get("reply") else "escalate"


# ── food ─────────────────────────────────────────────────────────────────────
async def _resolve_items(client_id: str, wanted: list[dict]) -> tuple[list[dict], list[dict]]:
    gw, items, unknown = get_gateway(), [], []
    for w in wanted:
        names = [w["name"], *w.get("alternatives", [])][:3]
        best = None
        for n in names:  # retry with the model's alternatives (≤2 extra lookups)
            found = await gw.search_foods(client_id, n, limit=3)
            if found and found[0].score >= MATCH_THRESHOLD:
                best = found[0]
                break
        if best is None:
            unknown.append({"name": w["name"], "alternatives": w.get("alternatives", [])})
            continue
        grams = float(w.get("grams") or DEFAULT_GRAMS)
        items.append({**MealItem(food_id=best.id, name=best.name, grams=grams, kcal=best.kcal, protein=best.protein,
                                 fat=best.fat, carbs=best.carbs).model_dump(),
                      "asked_as": w["name"], "grams_source": "user" if w.get("grams") else "default"})
    return items, unknown


_MEAL_VERBS = re.compile(r"\b(?:съел[аи]?|поел[аи]?|скушал[аи]?|выпил[аи]?|ел[аи]?|на (?:завтрак|обед|ужин|перекус)|сегодня|вчера)\b", re.I)


def parse_meal_text(text: str) -> list[dict]:
    """No-LLM fallback: «Съел 200 г плова и чай» → [{плова, 200}, {чай, None}]."""
    t = _MEAL_VERBS.sub(" ", text or "")
    parts = re.split(r",|;|\+|\bи\b|\bс\b|\bплюс\b", t, flags=re.I)
    out = []
    for part in parts:
        m = GRAMS.search(part)
        name = re.sub(r"\s+", " ", GRAMS.sub(" ", part)).strip(" .:-")
        if len(name) >= 3:
            out.append({"name": name, "grams": float(m.group(1)) if m else None})
    return out


def _card_text(items: list[dict], unknown: list[dict]) -> str:
    lines = []
    for it in items:
        kcal = round(it["kcal"] * it["grams"] / 100)
        tail = " — уточните граммы" if it["grams_source"] == "default" else ""
        lines.append(f"• {it['name']}: {int(it['grams'])} г ≈ {kcal} ккал{tail}")
    for u in unknown:
        lines.append(f"• «{u['name']}» — нет в справочнике, можно добавить вручную")
    total = round(sum(i["kcal"] * i["grams"] / 100 for i in items))
    head = f"Нашёл {len(items)} поз., всего ≈ {total} ккал:" if items else "Не нашёл продукты в справочнике:"
    return head + "\n" + "\n".join(lines) + ("\n\nПроверьте граммы и нажмите «Записать»." if items else "")


async def _meal_card(state: ChatState, wanted: list[dict]) -> dict:
    items, unknown = await _resolve_items(state["client_id"], wanted)
    card_id = await get_store().save_meal_card(state["client_id"], items, unknown)
    return {"meal": {"card_id": card_id, "items": items, "unknown": unknown}, "reply": _card_text(items, unknown),
            "kind": "meal_card"}


async def meal_photo(state: ChatState) -> dict:
    s = get_settings()
    b64 = base64.b64encode(Path(state["image_path"]).read_bytes()).decode()
    try:
        data, _ = await json_call("vision", prompt("vision"), "Что на фото?", images=[b64],
                                  temperature=s.vision_temperature, max_tokens=s.vision_max_tokens)
    except LLMError:
        return {"reply": "Не получилось распознать фото. Напишите, что вы съели, например: «плов 250 г».", "kind": "info"}
    wanted = [{"name": str(i.get("name", "")).strip(), "alternatives": [str(a) for a in i.get("alternatives", [])][:2]}
              for i in data.get("items", []) if i.get("name")]
    caption = GRAMS.search(state.get("text") or "")
    if caption and len(wanted) == 1:
        wanted[0]["grams"] = float(caption.group(1))
    if not wanted:
        return {"reply": "На фото не вижу еды. Если это блюдо — напишите его название и вес.", "kind": "info"}
    return await _meal_card(state, wanted)


async def meal_text(state: ChatState) -> dict:
    try:
        data, _ = await json_call("text", prompt("meal_text"), mask(_text(state)), temperature=0.0, max_tokens=300)
        wanted = [{"name": str(i["name"]), "grams": i.get("grams")} for i in data.get("items", []) if i.get("name")]
    except LLMError:
        wanted = parse_meal_text(_text(state))
    if not wanted:
        return {"reply": "Не понял, что записать. Пример: «гречка 200 г и курица 150 г».", "kind": "info"}
    return await _meal_card(state, wanted)


# ── hand-offs to people ──────────────────────────────────────────────────────
async def program_request(state: ChatState) -> dict:
    from .runner import new_program_proposal  # late import: runner imports this module

    trainer = await get_gateway().trainer_of(state["client_id"])
    if trainer is None:
        return {"reply": "У вас пока нет тренера в Tulpar, поэтому изменить программу некому.", "kind": "info"}
    p = await new_program_proposal(state["client_id"], trainer.id, _text(state), source="client")
    return {"proposal_id": p["id"], "kind": "proposal",
            "reply": "Передал тренеру. Я подготовлю черновик изменений, тренер его проверит, и вы увидите обновлённую "
                     "программу, как только он её примет."}


async def escalate(state: ChatState) -> dict:
    gw, store = get_gateway(), get_store()
    trainer = await gw.trainer_of(state["client_id"])
    reason = state.get("reason") or ("нет ответа в базе знаний" if state.get("intent") == "question" else "red flag")
    item = await store.create_proposal(kind="escalation", client_id=state["client_id"],
                                       trainer_id=trainer.id if trainer else None, source="client",
                                       request=_text(state), status="open")
    item = await store.update_proposal(item["id"], draft={"reason": reason, "intent": state.get("intent")})
    await notify.escalation(item)
    if state.get("red_flag"):
        msg = ("С этим лучше к тренеру и врачу, а не к боту. Я передал ваше сообщение тренеру. "
               "Если боль сильная, острая или не проходит — обратитесь к врачу, а при угрозе жизни звоните 103 или 112.")
    else:
        msg = "В моих материалах нет надёжного ответа на этот вопрос. Я передал его тренеру — он ответит здесь."
    return {"escalation_id": item["id"], "reply": msg, "kind": "escalated"}


async def refuse(state: ChatState) -> dict:
    return {"reply": "Я помогаю с тренировками и питанием по данным вашего клуба. С этим запросом помочь не могу.",
            "kind": "refusal"}


async def other(state: ChatState) -> dict:
    return {"kind": "info", "reply": (
        "Я AI-коуч вашего клуба. Могу:\n• посчитать калории по фото или сообщению «гречка 200 г» и записать в дневник;\n"
        "• ответить на вопросы по технике, питанию и нормам активности со ссылками на источники;\n"
        "• передать тренеру просьбу изменить программу.\nС болью, травмами и лекарствами — сразу к тренеру.")}


def build_chat_graph():
    g = StateGraph(ChatState)
    for name, fn in [("ingest", ingest), ("precheck", precheck), ("route", route), ("retrieve", retrieve_node),
                     ("rewrite", rewrite), ("answer", answer), ("meal_photo", meal_photo), ("meal_text", meal_text),
                     ("program_request", program_request), ("escalate", escalate), ("refuse", refuse), ("other", other)]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "precheck")
    g.add_edge("precheck", "route")
    g.add_conditional_edges("route", after_route, ["meal_photo", "meal_text", "retrieve", "program_request",
                                                   "escalate", "refuse", "other"])
    g.add_conditional_edges("retrieve", after_retrieve, ["answer", "rewrite", "escalate"])
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("answer", after_answer, [END, "escalate"])
    for leaf in ("meal_photo", "meal_text", "program_request", "escalate", "refuse", "other"):
        g.add_edge(leaf, END)
    return g


def today() -> str:
    return date.today().isoformat()
