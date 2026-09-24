import asyncio

from conftest import boot, shutdown


async def wait_status(store, pid, status, timeout=10.0):
    for _ in range(int(timeout * 20)):
        p = await store.get_proposal(pid)
        if p["status"] == status:
            return p
        await asyncio.sleep(0.05)
    raise AssertionError(f"proposal stayed {p['status']!r}, expected {status!r}")


async def test_pause_survives_restart_and_apply_is_idempotent(env, fake_llm):
    from tulpar_ai.graph import runner

    store, gw = await boot(env)
    client = await gw.demo_user("client")
    trainer = await gw.demo_user("trainer")
    before = await gw.active_plan(client.id)
    p = await runner.new_program_proposal(client.id, trainer.id, "Замени первое упражнение на щадящее", "trainer")
    p = await wait_status(store, p["id"], "pending")
    assert p["draft"]["ops"], "draft must contain ops"
    assert not [v for v in p["violations"] if v["severity"] == "error"]

    await shutdown(store)  # ── the service restarts while the trainer thinks ──
    store, gw = await boot(env)

    await runner.resume_program(p["id"], "accept")
    done = await store.get_proposal(p["id"])
    assert done["status"] == "applied"
    after = await gw.active_plan(client.id)
    op = done["draft"]["ops"][0]
    changed = next(e for e in after.days[0].exercises if e.id == op["wex_id"])
    assert changed.exercise_id == op["exercise_id"]
    assert before.days[0].exercises[0].exercise_id != changed.exercise_id

    # a second resume must not re-apply anything
    try:
        await runner.resume_program(p["id"], "accept")
        raise AssertionError("second resume should be refused")
    except RuntimeError:
        pass
    assert (await gw.active_plan(client.id)).model_dump() == after.model_dump()
    await shutdown(store)


async def test_validator_forces_redraft(app_state, fake_llm):
    from tulpar_ai.graph import runner

    store, gw = app_state
    fake_llm.draft_plan = ["invalid", "valid"]
    client, trainer = await gw.demo_user("client"), await gw.demo_user("trainer")
    p = await runner.new_program_proposal(client.id, trainer.id, "Замени первое упражнение", "trainer")
    p = await wait_status(store, p["id"], "pending")
    assert "(valid)" in p["draft"]["summary"]
    drafts = [c for c in fake_llm.calls if c["system"].startswith("Ты помогаешь тренеру")]
    assert len(drafts) == 2


async def test_trainer_edit_then_reject(app_state, fake_llm):
    from tulpar_ai.graph import runner

    store, gw = app_state
    fake_llm.draft_plan = ["valid", "valid"]
    client, trainer = await gw.demo_user("client"), await gw.demo_user("trainer")
    before = await gw.active_plan(client.id)
    p = await runner.new_program_proposal(client.id, trainer.id, "Замени первое упражнение", "client")
    await wait_status(store, p["id"], "pending")
    await runner.resume_program(p["id"], "edit", "Оставь 4 подхода")
    await wait_status(store, p["id"], "pending")
    await runner.resume_program(p["id"], "reject", "Пока без изменений")
    done = await store.get_proposal(p["id"])
    assert done["status"] == "rejected" and done["decision"]["comment"] == "Пока без изменений"
    assert (await gw.active_plan(client.id)).model_dump() == before.model_dump()
