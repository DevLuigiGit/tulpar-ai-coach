"""evals/build_retrieval_synth.py: stratified plan, code checks, the relevance rule and the generate→verify loop.

The LLM is the deterministic fake of `tulpar_ai.llm.set_fake`; no network, no keys."""

import json

from evals import build_retrieval_synth as bs
from tulpar_ai.rag.index import Chunk

WHO_TXT = ("Adults should do at least 150–300 minutes of moderate-intensity aerobic physical activity throughout the week "
           "for substantial health benefits.")


def _ex(i, group, extra=""):
    return Chunk(id=f"ex:{i}", source="exercises", title=f"Упражнение номер {i}", muscle_group=group, file_type="jsonl",
                 text=f"Упражнение: Упражнение номер {i}. Группа мышц: {group}. Техника: держите спину прямой и "
                      f"опускайтесь медленно до угла {i} градусов.{extra}")


def _who(page, idx, text):
    return Chunk(id=f"who:400:{page}:{idx}", source="who2020", title="WHO", page=page, chunk_index=idx, text=text)


def _corpus():
    ex = [_ex(i, "ноги") for i in range(6)] + [_ex(10 + i, "грудь", " Польза: развивает грудные мышцы.") for i in range(3)]
    nut = [Chunk(id="nut:Питание:0", source="nutrition", title="Правила питания Tulpar — Питание", text="<!-- Источник: код -->"),
           Chunk(id="nut:Вода:0", source="nutrition", title="Правила питания Tulpar — Вода",
                 text="## Вода\n\n- Норма воды — 32 мл на каждый килограмм веса, с округлением до 100 мл.")]
    who = [_who(5, 0, "CONTENTS Acknowledgements iv Abbreviations v Glossary vi Executive summary 1"),
           _who(12, 0, "It is recommended that: " + WHO_TXT),
           _who(12, 1, WHO_TXT + " Adults should also do muscle-strengthening activities on 2 or more days a week."),
           _who(42, 0, "RECOMMENDATION " + WHO_TXT + " Strong recommendation, moderate certainty evidence."),
           _who(44, 0, "Physical activity reduces the risk of all-cause mortality and cardiovascular disease mortality."),
           _who(86, 0, "56. Fang K, Mu M, Liu K, He Y. Screen time and childhood overweight. Child Care Health Dev. "
                       "2019;45(5):744–53. 57. Smith AB, Jones C, Lee D. Sleep. Pediatrics. 2018;141:1–9.")]
    return ex + nut + who


def test_largest_remainder_and_plan():
    assert bs.largest_remainder({"a": 36, "b": 21, "c": 2}, 10) == {"a": 6, "b": 3, "c": 1}
    chunks = _corpus()
    plan = bs.sample_plan(chunks, seed=1, n_ex=4, n_who=2, n_nut=3)
    kinds = {s.key: (s.kind, s.quota) for s in plan}
    assert kinds["ex-ноги"] == ("exercises", 3) and kinds["ex-грудь"] == ("exercises", 1)
    nut = [s for s in plan if s.kind == "nutrition"]
    assert [s.candidates[0]["chunk"].id for s in nut] == ["nut:Вода:0"]  # the HTML source note is not sampled
    who_ids = {c["chunk"].id for s in plan if s.kind == "who" for c in s.candidates}
    assert "who:400:5:0" not in who_ids and "who:400:86:0" not in who_ids  # front matter and bibliography
    assert len([s for s in plan if s.kind == "who"]) == 2 and len(who_ids) == 4
    again = bs.sample_plan(chunks, seed=1, n_ex=4, n_who=2, n_nut=3)
    assert [[c["chunk"].id for c in s.candidates] for s in plan] == [[c["chunk"].id for c in s.candidates] for s in again]
    grudь = next(s for s in plan if s.key == "ex-грудь")
    assert {c["aspect"] for c in grudь.candidates} <= {"technique", "benefit"}


def test_text_checks():
    card = _ex(3, "ноги").text
    assert bs.copy_run("Как держать спину прямой и опускаться медленно?", card) == 3
    assert bs.copy_run("держите спину прямой и опускайтесь медленно до угла", card) == 8
    assert bs.copy_run("Упражнение номер 3: держите спину?", card, ignore="Упражнение номер 3") == 2  # title words free
    assert bs.cyrillic_share("Сколько минут в неделю?") == 1.0 and bs.cyrillic_share("How many minutes?") == 0.0
    assert bs.lex_overlap("Прямая спина при опускании?", card) == 0.333  # 5-letter stems, as the lexical reranker
    assert bs.names_exercise("как делать упражнение номер 3?", "Упражнение номер 3")
    assert not bs.names_exercise("как приседать глубже?", "Упражнение номер 3")
    assert bs.who_band(9) == "who-summary" and bs.who_band(30) == "who-methods" and bs.who_band(86) == "who-back"
    assert bs.not_client_content(_corpus()[-1].text) and not bs.not_client_content(WHO_TXT)


def test_relevance_rule():
    chunks = _corpus()
    src = next(c for c in chunks if c.id == "who:400:12:0")
    groups = bs.relevant_groups(src, WHO_TXT, chunks)
    why = {g["id"]: g["why"] for g in groups}
    assert why == {"who:400:12:0": "source", "who:400:12:1": "evidence-overlap", "who:400:42:0": "evidence-repeat"}
    assert all(g["text"] == WHO_TXT for g in groups)
    short = "150–300 minutes"  # too short to make other chunks relevant by the quote
    g44 = next(c for c in chunks if c.id == "who:400:44:0")
    assert [g["id"] for g in bs.relevant_groups(g44, short, chunks)] == ["who:400:44:0"]
    # the neighbour of 12:1 holds most of its text: a near-duplicate even without a usable quote
    got = {g["id"]: g["why"] for g in bs.relevant_groups(next(c for c in chunks if c.id == "who:400:12:1"), short, chunks)}
    assert got["who:400:12:1"] == "source" and set(got.values()) <= {"source", "near-duplicate"} and len(got) > 1
    # no evidence: only near-duplicates join the source
    dup = Chunk(id="ex:99", source="exercises", title="Упражнение номер 99", muscle_group="ноги", text=_ex(3, "ноги").text + " Ещё.")
    g = bs.relevant_groups(next(c for c in chunks if c.id == "ex:3"), None, chunks + [dup])
    assert [(x["id"], x["why"]) for x in g] == [("ex:3", "source"), ("ex:99", "near-duplicate")]
    assert bs.locate_evidence("держите спину  прямой", _ex(3, "ноги").text) == ("держите спину  прямой", True)
    assert bs.locate_evidence("совсем другой текст про бег", _ex(3, "ноги").text) == (None, False)


class FakeGenVerify:
    """Generator answers from the chunk; the verifier rejects the first exercise question it sees."""

    def __init__(self):
        self.rejected = 0
        self.calls = 0

    def __call__(self, *, role, system, user, images, json_mode):
        self.calls += 1
        if system == bs.VERIFY_SYSTEM:
            if "Упражнение номер" in user and self.rejected == 0:
                self.rejected += 1
                return json.dumps({"answerable": False, "specific": True, "reason": "нет"})
            return json.dumps({"answerable": True, "specific": True, "reason": "да"})
        if "CONTENTS" in user:
            return json.dumps({"skip": True, "reason": "оглавление"})
        if "правила питания" in user:
            return json.dumps({"skip": False, "questions": [
                {"question": "Сколько воды мне пить при весе 70 кг?", "evidence": "Норма воды — 32 мл на каждый килограмм веса"},
                {"question": "Сколько воды мне пить при весе 70 кг?", "evidence": "32 мл"}]})
        if "ВОЗ" in user and "Фрагмент — из рекомендаций" in user:
            return json.dumps({"skip": False, "question": f"Сколько минут в неделю мне нужно ходить, вариант {len(user)}?",
                               "evidence": WHO_TXT if "150–300" in user else "reduces the risk of all-cause mortality"})
        n = user.split("Упражнение номер ")[1].split(".")[0]
        return json.dumps({"skip": False, "question": f"Как правильно делать упражнение номер {n}, до какого угла опускаться?",
                           "evidence": f"опускайтесь медленно до угла {n} градусов"})


async def test_build_with_fake_llm(env, tmp_path):
    from tulpar_ai import llm

    fake = FakeGenVerify()
    llm.set_fake(fake)
    try:
        chunks = _corpus()
        plan = bs.sample_plan(chunks, seed=3, n_ex=3, n_who=2, n_nut=2)
        cache = bs.LLMCache(tmp_path / "llm.jsonl")
        rows, stats = await bs.build(chunks, plan, cache, concurrency=2)
        by = {}
        for r in rows:
            by.setdefault(r["source"], []).append(r)
        assert len(by["exercises"]) == 3  # the rejected question was replaced by the next card of its group
        assert stats.drops["verify_not_answerable"] == 1 and stats.drops["duplicate"] == 1
        assert len(by["nutrition"]) == 1 and len(by["who2020"]) == 2
        assert all(r["synthetic"] is True and r["gen_model"] == "ollama:deepseek-v4.1-flash"
                   and r["verify_model"] == "ollama:kimi-k2.6" for r in rows)
        assert [r["id"] for r in rows] == [f"s{i:03d}" for i in range(1, len(rows) + 1)]
        w = next(r for r in by["who2020"] if r["source_chunk"] in ("who:400:12:0", "who:400:12:1", "who:400:42:0"))
        assert "cross-lingual" in w["tags"] and len(w["relevant"]) >= 2 and w["evidence_verbatim"]
        m = bs.meta(rows, plan, stats, chunks, 3, cache)
        assert m["synthetic"] is True and m["dropped"] == 2 and m["rows"] == len(rows)
        lex = m["lexical_overlap"]
        assert lex["synth_who_n"] == 2 and lex["synth_who_latin_term"] == 0.0 and "qa_who_number_shared" in lex
        calls = fake.calls
        rows2, _ = await bs.build(chunks, plan, bs.LLMCache(tmp_path / "llm.jsonl"), concurrency=2)
        assert fake.calls == calls and [r["question"] for r in rows2] == [r["question"] for r in rows]  # all from cache
    finally:
        llm.set_fake(None)


async def test_audit_counts_misses_answered_elsewhere(env, tmp_path):
    from tulpar_ai import llm

    chunks = _corpus()
    llm.set_fake(lambda **kw: json.dumps({"answerable": "all-cause mortality" in kw["user"], "specific": True}))
    try:
        gold = [{"id": "s001", "question": "Спорт продлевает жизнь?"}, {"id": "s002", "question": "Сколько минут?"},
                {"id": "s003", "question": "Что-то другое?"}]
        rows = [{"id": "s001", "tags": ["who"], "rank": None, "top": ["who:400:12:0", "who:400:44:0"]},
                {"id": "s002", "tags": ["who"], "rank": 1, "top": ["who:400:12:0"]},
                {"id": "s003", "tags": ["exercises"], "rank": 7, "top": ["ex:1", "ex:2"]}]
        result = {"configs": {"prod": {"sets": {"synth": {"rows": rows}}}}}
        a = await bs.audit(result, gold, chunks, bs.LLMCache(tmp_path / "c.jsonl"), k=4)
        assert a["n"] == 3 and a["misses"] == 2 and a["misses_answered_by_retrieved"] == 1
        assert a["hit@4"] == round(1 / 3, 4) and a["hit@4_if_counted"] == round(2 / 3, 4)
        assert a["by_tag"]["who"]["misses_answered_by_retrieved"] == 1 and a["synthetic"] is True
        assert [r["answering_chunks"] for r in a["rows"]] == [["who:400:44:0"], []]
    finally:
        llm.set_fake(None)
