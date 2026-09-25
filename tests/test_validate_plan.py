import json
from pathlib import Path

from tulpar_ai.skill import catalog, validator

GOLDEN = Path(__file__).resolve().parents[1] / "evals" / "golden" / "program.jsonl"


def test_catalog_loaded():
    assert len(catalog()) >= 150


def test_golden_program_cases():
    """Each golden case lists the violation codes the validator must raise (and nothing else as an error)."""
    cases = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(cases) >= 8
    v = validator()
    for c in cases:
        got = v.validate(c["plan"], c["ops"], c["client"], catalog())
        codes = {x["code"] for x in got if x["severity"] == "error"}
        assert set(c["expect_errors"]) == codes, (c["id"], sorted(codes), c["expect_errors"])


def test_replacement_must_keep_the_muscle_group():
    """Live draft 2abd60fb: the reason described a machine quad exercise, but the id was a cardio burpee."""
    v, cat = validator(), catalog()
    by_name = {e["name"]: e for e in cat.values()}
    plan = {"id": "p", "title": "t", "days": [{"day_index": 2, "title": "Legs", "exercises": [
        {"id": "w1", "exercise_id": by_name["Приседания со штангой"]["id"], "exercise_name": "Приседания со штангой",
         "muscle_group": "ноги", "target_sets": 4, "target_reps": 8}]}]}
    client = {"level": "inter", "place": "gym", "trainer_note": {"injury": True, "body": "колено"}}

    def codes(name: str) -> set[str]:
        ops = [{"op": "replace_exercise", "day_index": 2, "wex_id": "w1", "exercise_id": by_name[name]["id"]}]
        return {x["code"] for x in v.validate(plan, ops, client, cat) if x["severity"] == "error"}

    assert codes("Бёрпи с отжиманием") == {"E_MUSCLE_GROUP"}
    assert codes("Сгибания ног сидя") == set()


def test_strength_swap_across_groups_is_a_warning():
    """Live draft 59e67d98: RDL (ноги) → weighted back extension (спина) is the same hinge; the trainer decides."""
    v, cat = validator(), catalog()
    by_name = {e["name"]: e for e in cat.values()}
    rdl = by_name["Румынская тяга"]
    plan = {"id": "p", "title": "t", "days": [{"day_index": 2, "title": "Legs", "exercises": [
        {"id": "w1", "exercise_id": rdl["id"], "exercise_name": rdl["name"], "muscle_group": "ноги"}]}]}
    ops = [{"op": "replace_exercise", "day_index": 2, "wex_id": "w1", "exercise_id": by_name["Гиперэкстензия с весом"]["id"]}]
    got = {(x["code"], x["severity"]) for x in v.validate(plan, ops, {"level": "inter", "place": "gym"}, cat)}
    assert got == {("W_MUSCLE_GROUP", "warning")}


def test_core_and_abs_count_as_one_group():
    v, cat = validator(), catalog()
    core = next(e for e in cat.values() if e["muscle_group"] == "кор")
    abs_ = next(e for e in cat.values() if e["muscle_group"] == "пресс" and not e.get("contraindications"))
    plan = {"id": "p", "title": "t", "days": [{"day_index": 0, "title": "Кор", "exercises": [
        {"id": "w1", "exercise_id": core["id"], "exercise_name": core["name"], "muscle_group": "кор"},
        {"id": "w2", "exercise_id": "x", "exercise_name": "Другое", "muscle_group": "спина"}]}]}
    ops = [{"op": "replace_exercise", "day_index": 0, "wex_id": "w1", "exercise_id": abs_["id"]}]
    assert "E_MUSCLE_GROUP" not in {x["code"] for x in v.validate(plan, ops, {"level": "inter", "place": "gym"}, cat)}
