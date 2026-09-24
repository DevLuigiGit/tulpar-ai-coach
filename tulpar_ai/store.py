"""This service's own state (SQLite): proposals and escalations queue, chat log, meal cards,
Telegram chat registry and — in demo mode — the clients' plans and food diary.

The LangGraph checkpointer lives in a separate file (graph.sqlite) and is owned by LangGraph.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS proposals (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, client_id TEXT NOT NULL, trainer_id TEXT,
  source TEXT, request TEXT, status TEXT NOT NULL,
  draft_json TEXT, violations_json TEXT, before_json TEXT, after_json TEXT, decision_json TEXT,
  reply TEXT, created_at TEXT, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_proposals_status ON proposals(status, updated_at);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, client_id TEXT, role TEXT, text TEXT, payload_json TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_messages_client ON messages(client_id, id);
CREATE TABLE IF NOT EXISTS meal_cards (
  id TEXT PRIMARY KEY, client_id TEXT, items_json TEXT, unknown_json TEXT, status TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS tg_chats (
  telegram_id TEXT PRIMARY KEY, chat_id INTEGER, user_id TEXT, role TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS demo_users (
  id TEXT PRIMARY KEY, role TEXT, name TEXT, telegram_id TEXT UNIQUE, trainer_id TEXT, profile_json TEXT
);
CREATE TABLE IF NOT EXISTS demo_plans (client_id TEXT PRIMARY KEY, plan_json TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS demo_diary (
  id INTEGER PRIMARY KEY AUTOINCREMENT, client_id TEXT, on_date TEXT, meal TEXT, food_id TEXT, name TEXT,
  grams REAL, kcal REAL, protein REAL, fat REAL, carbs REAL, idem TEXT UNIQUE
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _j(v: Any) -> str | None:
    return None if v is None else json.dumps(v, ensure_ascii=False)


def _l(v: str | None) -> Any:
    return None if v is None else json.loads(v)


PROPOSAL_JSON = ("draft", "violations", "before", "after", "decision")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.db: aiosqlite.Connection | None = None

    async def open(self) -> "Store":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = await aiosqlite.connect(self.path)
        self.db.row_factory = aiosqlite.Row
        await self.db.executescript(SCHEMA)
        await self.db.commit()
        return self

    async def close(self) -> None:
        if self.db is not None:
            await self.db.close()
            self.db = None

    async def _all(self, sql: str, *args) -> list[dict]:
        async with self.db.execute(sql, args) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def _one(self, sql: str, *args) -> dict | None:
        rows = await self._all(sql, *args)
        return rows[0] if rows else None

    async def _exec(self, sql: str, *args) -> int:
        cur = await self.db.execute(sql, args)
        await self.db.commit()
        return cur.lastrowid

    # ── proposals & escalations ────────────────────────────────────────────
    async def create_proposal(self, *, kind: str, client_id: str, trainer_id: str | None, source: str, request: str,
                              status: str, proposal_id: str | None = None) -> dict:
        pid = proposal_id or str(uuid.uuid4())
        await self._exec(
            "INSERT INTO proposals(id,kind,client_id,trainer_id,source,request,status,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
            pid, kind, client_id, trainer_id, source, request, status, now(), now())
        return await self.get_proposal(pid)

    async def update_proposal(self, pid: str, **fields) -> dict | None:
        sets, args = [], []
        for k, v in fields.items():
            if k in PROPOSAL_JSON:
                sets.append(f"{k}_json=?")
                args.append(_j(v))
            else:
                sets.append(f"{k}=?")
                args.append(v)
        sets.append("updated_at=?")
        args.append(now())
        await self._exec(f"UPDATE proposals SET {', '.join(sets)} WHERE id=?", *args, pid)
        return await self.get_proposal(pid)

    def _proposal(self, row: dict | None) -> dict | None:
        if row is None:
            return None
        for k in PROPOSAL_JSON:
            row[k] = _l(row.pop(f"{k}_json"))
        return row

    async def get_proposal(self, pid: str) -> dict | None:
        return self._proposal(await self._one("SELECT * FROM proposals WHERE id=?", pid))

    async def list_proposals(self, *, trainer_id: str | None = None, client_id: str | None = None,
                             statuses: list[str] | None = None, limit: int = 100) -> list[dict]:
        where, args = [], []
        if trainer_id:
            where.append("trainer_id=?")
            args.append(trainer_id)
        if client_id:
            where.append("client_id=?")
            args.append(client_id)
        if statuses:
            where.append(f"status IN ({','.join('?' * len(statuses))})")
            args.extend(statuses)
        sql = "SELECT * FROM proposals" + (f" WHERE {' AND '.join(where)}" if where else "") + " ORDER BY updated_at DESC LIMIT ?"
        return [self._proposal(r) for r in await self._all(sql, *args, limit)]

    # ── chat log ───────────────────────────────────────────────────────────
    async def add_message(self, client_id: str, role: str, text: str, payload: dict | None = None) -> int:
        return await self._exec("INSERT INTO messages(client_id,role,text,payload_json,created_at) VALUES(?,?,?,?,?)",
                                client_id, role, text, _j(payload), now())

    async def history(self, client_id: str, limit: int = 50) -> list[dict]:
        rows = await self._all("SELECT * FROM messages WHERE client_id=? ORDER BY id DESC LIMIT ?", client_id, limit)
        for r in rows:
            r["payload"] = _l(r.pop("payload_json"))
        return list(reversed(rows))

    # ── meal cards ─────────────────────────────────────────────────────────
    async def save_meal_card(self, client_id: str, items: list[dict], unknown: list[dict]) -> str:
        cid = str(uuid.uuid4())
        await self._exec("INSERT INTO meal_cards VALUES(?,?,?,?,?,?)", cid, client_id, _j(items), _j(unknown), "draft", now())
        return cid

    async def get_meal_card(self, card_id: str) -> dict | None:
        row = await self._one("SELECT * FROM meal_cards WHERE id=?", card_id)
        if row:
            row["items"], row["unknown"] = _l(row.pop("items_json")), _l(row.pop("unknown_json"))
        return row

    async def set_meal_card_status(self, card_id: str, status: str) -> None:
        await self._exec("UPDATE meal_cards SET status=? WHERE id=?", status, card_id)

    # ── telegram registry ──────────────────────────────────────────────────
    async def upsert_tg_chat(self, telegram_id: str, chat_id: int, user_id: str, role: str) -> None:
        await self._exec(
            "INSERT INTO tg_chats VALUES(?,?,?,?,?) ON CONFLICT(telegram_id) DO UPDATE SET chat_id=excluded.chat_id, "
            "user_id=excluded.user_id, role=excluded.role, updated_at=excluded.updated_at",
            telegram_id, chat_id, user_id, role, now())

    async def tg_chats_for_user(self, user_id: str) -> list[dict]:
        return await self._all("SELECT * FROM tg_chats WHERE user_id=?", user_id)

    async def tg_chat(self, telegram_id: str) -> dict | None:
        return await self._one("SELECT * FROM tg_chats WHERE telegram_id=?", telegram_id)

    # ── demo mode world ────────────────────────────────────────────────────
    async def demo_user_count(self) -> int:
        row = await self._one("SELECT COUNT(*) AS n FROM demo_users")
        return int(row["n"]) if row else 0

    async def upsert_demo_user(self, u: dict) -> None:
        await self._exec(
            "INSERT INTO demo_users VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name, "
            "profile_json=excluded.profile_json, trainer_id=excluded.trainer_id",
            u["id"], u["role"], u["name"], u.get("telegram_id"), u.get("trainer_id"), _j(u.get("profile")))

    def _user(self, row: dict | None) -> dict | None:
        if row:
            row["profile"] = _l(row.pop("profile_json")) or {}
        return row

    async def demo_user(self, user_id: str) -> dict | None:
        return self._user(await self._one("SELECT * FROM demo_users WHERE id=?", user_id))

    async def demo_user_by_tg(self, telegram_id: str) -> dict | None:
        return self._user(await self._one("SELECT * FROM demo_users WHERE telegram_id=?", telegram_id))

    async def demo_clients(self, trainer_id: str) -> list[dict]:
        return [self._user(r) for r in await self._all("SELECT * FROM demo_users WHERE role='client' AND trainer_id=? ORDER BY name", trainer_id)]

    async def get_demo_plan(self, client_id: str) -> dict | None:
        row = await self._one("SELECT plan_json FROM demo_plans WHERE client_id=?", client_id)
        return _l(row["plan_json"]) if row else None

    async def put_demo_plan(self, client_id: str, plan: dict) -> None:
        await self._exec("INSERT INTO demo_plans VALUES(?,?,?) ON CONFLICT(client_id) DO UPDATE SET plan_json=excluded.plan_json, "
                         "updated_at=excluded.updated_at", client_id, _j(plan), now())

    async def add_demo_diary(self, client_id: str, rows: list[dict], idem: str | None = None) -> int:
        n = 0
        for i, r in enumerate(rows):
            key = f"{idem}:{i}" if idem else None
            try:
                await self.db.execute(
                    "INSERT INTO demo_diary(client_id,on_date,meal,food_id,name,grams,kcal,protein,fat,carbs,idem) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (client_id, r["on_date"], r["meal"], r.get("food_id"), r["name"], r["grams"], r["kcal"],
                     r["protein"], r["fat"], r["carbs"], key))
                n += 1
            except aiosqlite.IntegrityError:
                pass  # same idempotency key → already logged
        await self.db.commit()
        return n

    async def demo_diary(self, client_id: str, since: str) -> list[dict]:
        return await self._all("SELECT * FROM demo_diary WHERE client_id=? AND on_date>=? ORDER BY on_date, id", client_id, since)


_store: Store | None = None


def set_store(store: Store | None) -> None:
    global _store
    _store = store


def get_store() -> Store:
    if _store is None:
        raise RuntimeError("store is not open (lifespan not started)")
    return _store
