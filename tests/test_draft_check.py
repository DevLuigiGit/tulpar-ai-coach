"""A draft whose summary promises more than its operations do goes back for another draft (screenshot of 30.09)."""

import pytest

from conftest import boot, shutdown
from test_program_resume import wait_status
from tulpar_ai.draft_check import summary_mismatches

PLAN = {"days": [{"day_index": 0, "title": "Ноги", "exercises": [
    {"id": "a", "exercise_name": "Приседания со штангой"}, {"id": "b", "exercise_name": "Жим ногами"},
    {"id": "c", "exercise_name": "Выпады с гантелями"}, {"id": "d", "exercise_name": "Жим штанги лёжа"}]}]}
OPS = [{"op": "replace_exercise", "day_index": 0, "wex_id": "a", "exercise_id": "x"}]


def test_the_screenshot_summary_is_caught():
    summary = ("У клиента колено после бега, тренер запретил глубокие приседания со штангой, прыжки и жим ногами в "
               "полной амплитуде. В плане есть приседания со штангой и жим ногами — нужно их заменить на безопасные "
               "альтернативы для ног; всё остальное менять не нужно.")
    [v] = summary_mismatches(PLAN, OPS, summary)
    assert v["code"] == "E_SUMMARY_MISMATCH" and v["severity"] == "error" and "«Жим ногами»" in v["message"]


@pytest.mark.parametrize("summary", [
    "Заменить приседания со штангой на зашагивания на платформу.",
    "Заменяем приседания со штангой на зашагивания; жим ногами оставляем в неполной амплитуде.",
    "Заменить жим на что-то щадящее.",  # «жим» is two exercises of the plan: not a promise about one of them
    "Тренер запретил жим ногами в полной амплитуде.",  # no change promised in this clause
])
def test_no_false_alarm(summary):
    assert summary_mismatches(PLAN, OPS, summary) == []


def test_a_unique_first_word_is_enough():
    [v] = summary_mismatches(PLAN, OPS, "Убираем выпады и приседания со штангой.")
    assert "«Выпады с гантелями»" in v["message"]


@pytest.mark.parametrize("mode", ["agent", "candidates"])
async def test_overclaiming_draft_is_sent_back(env, fake_llm, monkeypatch, mode):
    from tulpar_ai.config import get_settings
    from tulpar_ai.graph import runner

    monkeypatch.setenv("DRAFT_MODE", mode)
    get_settings.cache_clear()
    store, gw = await boot(env)
    try:
        fake_llm.draft_plan = ["overclaim", "valid"]
        client, trainer = await gw.demo_user("client"), await gw.demo_user("trainer")
        p = await runner.new_program_proposal(client.id, trainer.id, "Замени первое упражнение", "trainer")
        p = await wait_status(store, p["id"], "pending")
        assert fake_llm.drafts == 2 and "(valid)" in p["draft"]["summary"]
        assert not [v for v in p["violations"] if v["code"] == "E_SUMMARY_MISMATCH"]
    finally:
        await shutdown(store)
