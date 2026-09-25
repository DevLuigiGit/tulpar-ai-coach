"""One full defense scenario against a running service: client chat → trainer queue → the plan really changes.

    .venv/bin/python tools/demo_scenario.py [--base http://localhost:8089] [--no-traces]

Steps: client demo-login → RAG question → meal by text (+ confirm) → program request → trainer demo-login →
queue → accept the draft → the client's plan differs from the one before. Every step prints one or two lines.
With LANGSMITH_API_KEY set (env or .env) it then prints links to the LangSmith traces of exactly this run.
Exit code 1 if a step the next ones depend on fails.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUESTION = "Сколько минут активности в неделю рекомендует ВОЗ?"
MEAL = "Съел 200 г плова и чай"
PROGRAM = "Хочу добавить кардио"


class StepFailed(RuntimeError):
    pass


def short(text: str | None, n: int = 110) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def plan_lines(plan: dict | None) -> list[str]:
    """Flat, comparable view of a plan: one line per exercise with its day and volume."""
    out = []
    for d in (plan or {}).get("days", []):
        for e in d.get("exercises", []):
            out.append(f"{d.get('title')}: {e.get('exercise_name')} {e.get('target_sets')}×{e.get('target_reps')}")
    return out


def plan_diff(before: dict | None, after: dict | None) -> tuple[list[str], list[str]]:
    b, a = plan_lines(before), plan_lines(after)
    return [x for x in b if x not in a], [x for x in a if x not in b]


class Scenario:
    def __init__(self, http: httpx.Client, out: Callable[[str], None] = print, draft_timeout: float = 240,
                 poll_s: float = 2.0):
        self.http, self.out = http, out
        self.draft_timeout, self.poll_s = draft_timeout, poll_s
        self.t0 = time.perf_counter()
        self.n = 0
        self.warnings: list[str] = []
        self.sent: list[str] = []
        self.client_id = ""
        self.proposal_id = ""

    # ── plumbing ─────────────────────────────────────────────────────────────
    def step(self, title: str, *lines: str) -> None:
        self.n += 1
        self.out(f"[{self.n}] {time.perf_counter() - self.t0:6.1f}s  {title}")
        for line in lines:
            self.out(f"      {line}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.out(f"      ВНИМАНИЕ: {msg}")

    def call(self, method: str, path: str, headers: dict | None = None, **kw) -> dict | list:
        r = self.http.request(method, path, headers=headers, **kw)
        if r.status_code >= 400:
            raise StepFailed(f"{method} {path} → {r.status_code}: {short(r.text, 200)}")
        return r.json()

    def login(self, role: str) -> tuple[dict, dict]:
        d = self.call("POST", "/api/auth/demo-login", json={"role": role})
        return {"Authorization": f"Bearer {d['token']}"}, d["user"]

    def chat(self, headers: dict, text: str) -> dict:
        self.sent.append(text)
        t = time.perf_counter()
        r = self.call("POST", "/api/chat", headers, data={"text": text})
        r["_s"] = time.perf_counter() - t
        return r

    # ── the scenario ─────────────────────────────────────────────────────────
    def run(self) -> None:
        health = self.call("GET", "/health")
        cl, client = self.login("client")
        self.client_id = client["id"]
        before = self.call("GET", "/api/my/plan", cl)
        self.step("Клиент: demo-login", f"{client['name']} ({client['id'][:8]}), режим {health.get('mode')}, "
                                         f"план «{(before or {}).get('title')}», упражнений: {len(plan_lines(before))}")

        r = self.chat(cl, QUESTION)
        cites = "; ".join(f"[{c['n']}] {short(c['title'], 50)}" + (f", стр. {c['page']}" if c.get("page") else "")
                          for c in r.get("citations") or [])
        self.step(f"Клиент: вопрос «{QUESTION}»", f"intent={r.get('intent')} kind={r['kind']} за {r['_s']:.1f} с",
                  f"ответ: {short(r['reply'])}", f"источники: {cites or '—'}")
        if r["kind"] != "answer" or not r.get("citations"):
            self.warn("ожидался ответ со ссылками на источники")

        r = self.chat(cl, MEAL)
        items = (r.get("meal") or {}).get("items") or []
        lines = [f"intent={r.get('intent')} kind={r['kind']} за {r['_s']:.1f} с"]
        lines += [f"{it['name']}: {int(it['grams'])} г ≈ {round(it['kcal'] * it['grams'] / 100)} ккал" for it in items]
        if items:
            ok = self.call("POST", f"/api/meals/{r['meal']['card_id']}/confirm", cl, json={"meal": "lunch"})
            lines.append(f"записано в дневник: {ok['logged']} поз., ≈ {ok['total_kcal']} ккал")
        self.step(f"Клиент: еда текстом «{MEAL}»", *lines)
        if r["kind"] != "meal_card" or not items:
            self.warn("ожидалась карточка КБЖУ")

        r = self.chat(cl, PROGRAM)
        self.step(f"Клиент: «{PROGRAM}»", f"intent={r.get('intent')} kind={r['kind']} за {r['_s']:.1f} с",
                  f"ответ: {short(r['reply'])}")
        if not r.get("proposal_id"):
            raise StepFailed("запрос на изменение программы не создал черновик (proposal_id пуст)")
        self.proposal_id = r["proposal_id"]

        tr, trainer = self.login("trainer")
        self.step("Тренер: demo-login", f"{trainer['name']} ({trainer['id'][:8]})")

        p = self.wait_draft(tr)
        queue = self.call("GET", "/api/queue", tr)
        mine = next((x for x in queue if x["id"] == self.proposal_id), None)
        kinds = ", ".join(f"{x['kind']}:{x['status']}" for x in queue)
        changes = [f"{c['day']}: {c['was'] or '—'} → {c['becomes']}" for c in p.get("changes") or []]
        violations = p.get("violations") or []
        issues = [f"{'ошибка' if v.get('severity') == 'error' else 'предупреждение'} валидатора: {short(v['message'])}"
                  for v in violations]
        self.step("Тренер: очередь", f"в очереди {len(queue)} ({kinds}); черновик {self.proposal_id[:8]} "
                                     f"{'есть' if mine else 'НЕ найден'}",
                  f"summary: {short((p.get('draft') or {}).get('summary'))}", *changes, *issues)
        if mine is None:
            raise StepFailed("черновика нет в очереди тренера")
        if any(v.get("severity") == "error" for v in violations):
            self.warn("черновик ушёл тренеру с ошибками валидатора: попытки доработки исчерпаны")

        p = self.call("POST", f"/api/proposals/{self.proposal_id}/decision", tr, json={"action": "accept"})
        self.step("Тренер: принять черновик", f"статус: {p['status']}")
        if p["status"] != "applied":
            raise StepFailed(f"черновик не применён: {p['status']} {short(p.get('reply'))}")

        after = self.call("GET", "/api/my/plan", cl)
        removed, added = plan_diff(before, after)
        self.step("Клиент: план после решения тренера", *(f"- {x}" for x in removed), *(f"+ {x}" for x in added))
        if not (removed or added):
            raise StepFailed("план клиента не изменился")

    def wait_draft(self, headers: dict) -> dict:
        deadline = time.monotonic() + self.draft_timeout
        while True:
            p = self.call("GET", f"/api/proposals/{self.proposal_id}", headers)
            if p["status"] != "drafting":
                break
            if time.monotonic() > deadline:
                raise StepFailed(f"черновик не собран за {self.draft_timeout:.0f} с")
            time.sleep(self.poll_s)
        self.step("Агент: черновик изменений", f"статус {p['status']} через "
                                               f"{time.perf_counter() - self.t0:.1f} с от старта сценария")
        if p["status"] != "pending":
            raise StepFailed(f"черновик в статусе {p['status']}: {short(p.get('reply'))}")
        return p


# ── LangSmith links ─────────────────────────────────────────────────────────
def _meta(run) -> dict:
    return ((run.extra or {}).get("metadata") or {}) if hasattr(run, "extra") else {}


def match_runs(runs: list, texts: list[str], client_id: str, proposal_id: str) -> list:
    """Root runs of THIS scenario: chat turns by their exact text and client, the program graph by proposal."""
    out = []
    for r in runs:
        inputs = r.inputs or {}
        if r.name == "coach_turn" and inputs.get("text") in texts and inputs.get("client_id") == client_id:
            out.append(r)
        elif r.name == "program_change" and proposal_id and (
                inputs.get("proposal_id") == proposal_id or _meta(r).get("proposal") == proposal_id[:8]):
            out.append(r)
    return sorted(out, key=lambda r: r.start_time)


def trace_links(started: datetime, sc: Scenario, out: Callable[[str], None] = print, wait_s: float = 45) -> None:
    from tulpar_ai import config  # noqa: F401  (exports .env into os.environ for the LangSmith client)

    if not os.environ.get("LANGSMITH_API_KEY"):
        out("LangSmith не настроен (нет LANGSMITH_API_KEY) — ссылок на трейсы нет.")
        return
    from langsmith import Client

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    client, project = Client(), os.environ.get("LANGSMITH_PROJECT") or "default"
    want = len(sc.sent) + 1  # chat turns + at least the program graph's resume
    deadline, found = time.monotonic() + wait_s, []
    while True:  # ingestion is asynchronous: runs show up a few seconds after the fact
        runs = list(client.list_runs(project_name=project, start_time=started, is_root=True, limit=100))
        found = match_runs(runs, sc.sent, sc.client_id, sc.proposal_id)
        if len(found) >= want or time.monotonic() > deadline:
            break
        time.sleep(5)
    out(f"LangSmith, проект «{project}»: трейсов этого прогона — {len(found)}")
    for r in found:
        kids = list(client.list_runs(project_name=project, trace_id=r.trace_id, select=["name", "run_type"], limit=100))
        llm = [k for k in kids if k.run_type == "llm"]
        other = sorted({k.name for k in kids if k.run_type in ("retriever", "embedding", "tool")})
        secs = (r.end_time - r.start_time).total_seconds() if r.end_time else 0.0
        label = short((r.inputs or {}).get("text"), 40) or (_meta(r).get("proposal") and f"proposal {_meta(r)['proposal']}")
        url = r.url or client.get_run_url(run=r, project_name=project)
        out(f"  {r.name} «{label}» {secs:.1f} с, спанов {len(kids)}, LLM-вызовов {len(llm)}"
            + (f", {', '.join(other)}" if other else ""))
        out(f"    {url}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://localhost:8089", help="URL запущенного сервиса")
    ap.add_argument("--no-traces", action="store_true", help="не искать трейсы в LangSmith")
    ap.add_argument("--draft-timeout", type=float, default=240)
    args = ap.parse_args()

    started = datetime.now(timezone.utc) - timedelta(seconds=5)  # small margin for clock skew
    print(f"Сценарий защиты → {args.base}  ({started.astimezone():%Y-%m-%d %H:%M})")
    with httpx.Client(base_url=args.base, timeout=180) as http:
        sc = Scenario(http, draft_timeout=args.draft_timeout)
        try:
            sc.run()
        except (StepFailed, httpx.HTTPError) as e:
            print(f"ОШИБКА на шаге {sc.n + 1}: {e}")
            return 1
    print(f"Итог: {sc.n} шагов за {time.perf_counter() - sc.t0:.1f} с, "
          + ("без замечаний" if not sc.warnings else f"замечаний: {len(sc.warnings)}"))
    if not args.no_traces:
        try:
            trace_links(started, sc)
        except Exception as e:  # tracing is a bonus: never fail the demo because of it
            print(f"LangSmith недоступен: {type(e).__name__}: {short(str(e), 160)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
