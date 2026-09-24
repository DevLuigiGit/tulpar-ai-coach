"""Screenshots for README: runs the real UI against a running service (python -m tulpar_ai on :8089).

    .venv/bin/python tools/screenshots.py
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

BASE = "http://localhost:8089"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def token(role: str) -> dict:
    r = httpx.post(f"{BASE}/api/auth/demo-login", json={"role": role}, timeout=30).json()
    return r


def send(page, text: str, wait_for: str, timeout: int = 120_000) -> None:
    box = page.locator("textarea, input[type=text]").last
    box.fill(text)
    page.get_by_label("Отправить").click()
    page.get_by_text(wait_for).last.wait_for(timeout=timeout)
    time.sleep(0.8)


def main() -> None:
    client, trainer = token("client"), token("trainer")
    th = {"Authorization": f"Bearer {trainer['token']}"}
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 430, "height": 900}, device_scale_factor=2, locale="ru-RU")
        page = ctx.new_page()
        page.goto(BASE)
        page.get_by_text("Войти как клиент").wait_for()
        page.screenshot(path=OUT / "01-login.png")
        page.get_by_text("Войти как клиент").click()
        page.get_by_label("Отправить").wait_for()
        send(page, "Съел 200 г плова и чай", "Записать")
        page.screenshot(path=OUT / "02-meal-card.png")
        send(page, "Сколько минут активности в неделю рекомендует ВОЗ взрослым?", "стр.")
        page.screenshot(path=OUT / "03-answer-with-citations.png")
        send(page, "Колено болит при приседе, что делать?", "тренер")
        page.screenshot(path=OUT / "04-escalation.png")

        # trainer asks the agent for a change → wait for the draft
        p0 = httpx.post(f"{BASE}/api/trainer/proposals", headers=th, timeout=60,
                        json={"client_id": client["user"]["id"],
                              "request": "Колено беспокоит: замени приседания и выпады на щадящие упражнения"}).json()
        for _ in range(90):
            st = httpx.get(f"{BASE}/api/proposals/{p0['id']}", headers=th, timeout=30).json()
            if st["status"] != "drafting":
                break
            time.sleep(2)
        print("draft status:", st["status"], "|", (st.get("draft") or {}).get("summary"))

        tctx = b.new_context(viewport={"width": 1200, "height": 1000}, device_scale_factor=2, locale="ru-RU")
        tp = tctx.new_page()
        tp.goto(BASE)
        tp.get_by_text("Войти как тренер").click()
        tp.get_by_text("Принять").first.wait_for(timeout=60_000)
        time.sleep(1)
        tp.screenshot(path=OUT / "05-trainer-queue.png", full_page=True)
        tp.goto(f"{BASE}/#/clients/{client['user']['id']}")
        time.sleep(3)
        tp.screenshot(path=OUT / "06-client-detail.png", full_page=True)
        tp.goto(f"{BASE}/#/queue")
        tp.get_by_text("Принять").first.click()
        time.sleep(4)
        tp.screenshot(path=OUT / "07-applied.png", full_page=True)
        b.close()
    print("saved:", sorted(x.name for x in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
