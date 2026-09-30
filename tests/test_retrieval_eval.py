"""evals/retrieval_eval.py: matchers, metrics, bootstrap, paired tests and the CLI — no network, no keys."""

import json
import math

import numpy as np
import pytest

from evals import retrieval_eval as re_


def _chunks():
    return [
        {"id": "ex:1", "source": "exercises", "title": "Жим штанги лёжа", "page": None,
         "text": "Упражнение: Жим штанги лёжа. Техника: локти под углом 45 градусов к корпусу."},
        {"id": "who:400:12:0", "source": "who2020", "title": "WHO", "page": 12,
         "text": "Adults should do at least 150–300 minutes of moderate-intensity aerobic physical activity."},
        {"id": "who:400:12:1", "source": "who2020", "title": "WHO", "page": 12,
         "text": "Adults should also do muscle-strengthening activities on 2 or more days a week."},
        {"id": "who:400:42:0", "source": "who2020", "title": "WHO", "page": 42,
         "text": "Adults should do at least 150–300 minutes of moderate-intensity aerobic physical activity."},
        {"id": "nut:Вода:0", "source": "nutrition", "title": "Правила питания Tulpar — Вода", "page": None,
         "text": "Норма воды — 32 мл на каждый килограмм веса."},
    ]


def test_score_row_by_hand():
    s = re_.score_row(["x", "a", "y", "b"], [frozenset({"a"}), frozenset({"b"})])
    assert s["rank"] == 2 and s["hit@1"] == 0.0 and s["hit@4"] == 1.0 and s["mrr"] == 0.5
    assert s["recall@10"] == 1.0
    ideal = 1 + 1 / math.log2(3)
    assert s["ndcg@10"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)


def test_score_row_groups_count_once_and_depth_cuts():
    # two chunks of one group: the second adds no gain, recall counts the group once
    s = re_.score_row(["a1", "a2"], [frozenset({"a1", "a2"}), frozenset({"b"})])
    assert s["recall@10"] == 0.5 and s["ndcg@10"] == pytest.approx(1 / (1 + 1 / math.log2(3)))
    # relevant only past the depth: nothing counts
    ranked = [f"n{i}" for i in range(20)] + ["a"]
    s = re_.score_row(ranked, [frozenset({"a"})], depth=20)
    assert s["rank"] is None and s["mrr"] == 0.0 and s["hit@4"] == 0.0
    # found at rank 12: MRR sees it, recall@10 and nDCG@10 do not
    s = re_.score_row([f"n{i}" for i in range(11)] + ["a"], [frozenset({"a"})])
    assert s["rank"] == 12 and s["mrr"] == pytest.approx(1 / 12) and s["recall@10"] == 0.0 and s["ndcg@10"] == 0.0


def test_resolve_matchers():
    uni = re_.Universe.of(_chunks())
    assert re_.resolve("ex:1", uni) == {"ex:1"}
    assert re_.resolve({"source": "exercises", "title": "Жим штанги лёжа"}, uni) == {"ex:1"}
    assert re_.resolve({"source": "who2020", "pages": [11, 12, 13]}, uni) == {"who:400:12:0", "who:400:12:1"}
    assert re_.resolve({"source": "nutrition"}, uni) == {"nut:Вода:0"}
    # id of another chunking: falls back to source + text coverage and finds both copies of the quote
    m = {"id": "who:800:12:0", "source": "who2020", "text": "at least 150–300 minutes of moderate-intensity aerobic"}
    assert re_.resolve(m, uni) == {"who:400:12:0", "who:400:42:0"}
    # the id exists but no longer covers the text (another overlap setting): not trusted
    m = {"id": "who:400:12:1", "source": "who2020", "page": 12, "text": "150–300 minutes of moderate-intensity aerobic"}
    assert re_.resolve(m, uni) == {"who:400:12:0"}
    assert re_.resolve({"id": "missing"}, uni) == frozenset()


def test_qa_rows_keep_run_py_rules(tmp_path):
    p = tmp_path / "qa.jsonl"
    rows = [
        {"id": "q1", "question": "Жим?", "answerable": True, "expected_sources": [{"source": "exercises", "title": "Жим штанги лёжа"}], "tags": ["technique"]},
        {"id": "q2", "question": "ВОЗ?", "answerable": True, "expected_sources": [{"source": "who2020", "page": 12}, {"source": "who2020", "page": 42}], "tags": ["who"]},
        {"id": "q3", "question": "Вода?", "answerable": True, "expected_sources": [{"source": "nutrition"}], "tags": []},
        {"id": "q4", "question": "Креатин?", "answerable": False, "expected_sources": [], "tags": ["unanswerable"]},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    out = re_.qa_rows(p)
    assert [r["id"] for r in out] == ["q1", "q2", "q3"]
    assert out[1]["relevant"] == [{"source": "who2020", "pages": [11, 12, 13]}, {"source": "who2020", "pages": [41, 42, 43]}]
    uni = re_.Universe.of(_chunks())
    assert re_.groups_for(out[1], uni) == [frozenset({"who:400:12:0", "who:400:12:1"}), frozenset({"who:400:42:0"})]


def test_bootstrap_is_seeded_and_brackets_the_mean():
    v = {"hit@4": np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=float)}
    a, b = re_.bootstrap(v, 1000, 7), re_.bootstrap(v, 1000, 7)
    assert a == b and a["hit@4"]["lo"] <= a["hit@4"]["mean"] == 0.7 <= a["hit@4"]["hi"]
    const = re_.bootstrap({"mrr": np.ones(5)}, 200, 1)["mrr"]
    assert const == {"mean": 1.0, "lo": 1.0, "hi": 1.0}


def test_mcnemar_exact():
    assert re_.mcnemar_exact(0, 0) == 1.0
    assert re_.mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert re_.mcnemar_exact(11, 1) == pytest.approx(2 * 13 / 4096)
    assert re_.mcnemar_exact(5, 5) == 1.0


def test_paired_compare():
    rows_a = [{"id": f"q{i}", **{m: 0.0 for m in re_.METRICS}} for i in range(20)]
    rows_b = [{"id": f"q{i}", **{m: 1.0 for m in re_.METRICS}} for i in range(20)]
    same = re_.paired_compare(rows_a, rows_a, 500, 1)
    assert same["metrics"]["hit@4"]["diff"] == 0 and same["metrics"]["hit@4"]["p_boot"] == 1.0
    better = re_.paired_compare(rows_a, rows_b + [{"id": "extra", **{m: 1.0 for m in re_.METRICS}}], 500, 1)
    assert better["n"] == 20 and better["metrics"]["mrr"]["lo"] == 1.0
    assert better["mcnemar_hit@4"] == {"only_a": 0, "only_b": 20, "p": pytest.approx(2 / 2 ** 20, rel=1e-2)}  # 3 significant digits
    assert better["metrics"]["hit@4"]["p_boot"] == 0.0 and "< 0.002" in re_.compare_table(better, "a", "b")


async def test_evaluate_sync_and_async_configs():
    uni = re_.Universe.of(_chunks())
    rows = [{"id": "r1", "question": "локти", "relevant": ["ex:1"], "tags": ["t"]},
            {"id": "r2", "question": "минуты", "relevant": [{"source": "who2020", "pages": [12]}], "tags": ["t", "who"]},
            {"id": "r3", "question": "нет", "relevant": ["gone"], "tags": []}]
    answers = {"локти": ["ex:1"], "минуты": ["nut:Вода:0", "who:400:12:1"]}
    warmed = []

    async def warm(qs):
        warmed.extend(qs)

    cfg = re_.RetrievalConfig("sync", lambda q, k: answers.get(q, [])[:k], warmup=warm)
    res = await re_.evaluate(cfg, rows, universe=uni, n_boot=100)
    assert res["unresolved"] == ["r3"] and res["n"] == 2 and warmed == ["локти", "минуты"]
    assert res["metrics"]["hit@1"]["mean"] == 0.5 and res["metrics"]["mrr"]["mean"] == 0.75
    assert res["by_tag"]["who"]["n"] == 1 and res["rows"][1]["rank"] == 2

    async def aretrieve(q, k):
        return answers.get(q, [])[:k]

    res2 = await re_.evaluate(re_.RetrievalConfig("async", aretrieve), rows, universe=uni, n_boot=100)
    assert res2["metrics"] == res["metrics"]
    # a config with its own chunking resolves matchers against its own chunks
    own = re_.RetrievalConfig("own", lambda q, k: ["c1"], chunks=[{"id": "c1", "source": "exercises", "title": "Жим штанги лёжа",
                                                                    "page": None, "text": "..."}])
    by_title = {"id": "r4", "question": "жим", "relevant": [{"source": "exercises", "title": "Жим штанги лёжа"}], "tags": []}
    res3 = await re_.evaluate(own, [rows[0], by_title], universe=uni, n_boot=10)
    assert res3["unresolved"] == ["r1"] and res3["n"] == 1 and res3["metrics"]["hit@1"]["mean"] == 1.0


async def test_evaluate_callable(tmp_path):
    p = tmp_path / "mini.jsonl"
    p.write_text(json.dumps({"id": "m1", "question": "вода", "relevant": [{"source": "nutrition"}], "tags": ["nutrition"],
                             "synthetic": True}, ensure_ascii=False), encoding="utf-8")
    out = await re_.evaluate_callable(lambda q, k: ["ex:1", "nut:Вода:0"], [str(p)], chunks=_chunks(), n_boot=20)
    assert out["mini"]["rows"][0]["rank"] == 2 and out["mini"]["metrics"]["mrr"]["mean"] == 0.5


async def test_unknown_config_and_registry():
    @re_.register("test_fixed", "always the same ids")
    async def _fixed(ctx, arg):
        return re_.RetrievalConfig("test_fixed", lambda q, k: ["ex:1"])

    try:
        cfg = await re_.make_config("test_fixed:anything", ctx=None)
        assert cfg.retrieve("q", 4) == ["ex:1"]
        with pytest.raises(SystemExit):
            await re_.make_config("no_such_config", ctx=None)
    finally:
        re_.CONFIGS.pop("test_fixed", None)


def test_cli_run_and_compare(env, tmp_path, monkeypatch):
    """Two configs on a small set through the CLI: the production local fallback (real index) and a plug-in."""
    gold = tmp_path / "set.jsonl"
    gold.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"id": "a", "question": "Как делать жим штанги лёжа, под каким углом локти?", "relevant": [
            {"source": "exercises", "title": "Жим штанги лёжа"}], "tags": ["technique"], "synthetic": True},
        {"id": "b", "question": "Сколько воды пить в день на килограмм веса?", "relevant": [{"source": "nutrition"}],
         "tags": ["nutrition"], "synthetic": True},
    ]), encoding="utf-8")
    plugin = tmp_path / "my_variant.py"
    plugin.write_text(
        "from evals.retrieval_eval import RetrievalConfig, register\n"
        "@register('nothing', 'finds nothing')\n"
        "async def make(ctx, arg):\n"
        "    return RetrievalConfig('nothing', lambda q, k: [])\n", encoding="utf-8")
    out = tmp_path / "res.json"
    try:
        assert re_.main(["run", "--config", "prod_local_fallback", "--config", "nothing", "--plugin", str(plugin),
                         "--sets", str(gold), "--out", str(out), "--n-boot", "50"]) == 0
    finally:
        re_.CONFIGS.pop("nothing", None)
    d = json.loads(out.read_text(encoding="utf-8"))
    local = d["configs"]["prod_local_fallback"]
    assert local["meta"]["rerank"] is True and local["meta"]["embedder"] == "local"
    s = local["sets"]["set"]
    assert s["n"] == 2 and s["metrics"]["hit@4"]["mean"] == 1.0 and d["sets"]["set"]["synthetic"] is True
    assert d["configs"]["nothing"]["sets"]["set"]["metrics"]["hit@4"]["mean"] == 0.0
    [cmp] = d["comparisons"]
    assert cmp["a"] == "prod_local_fallback" and cmp["mcnemar_hit@4"]["only_a"] == 2
    again = re_.cmd_compare(re_.parser().parse_args(["compare", str(out), "--n-boot", "50"]))
    assert again["comparisons"][0]["metrics"]["hit@4"]["diff"] == -1.0
    sliced = re_.cmd_compare(re_.parser().parse_args(["compare", str(out), "--tag", "nutrition", "--n-boot", "50"]))
    assert sliced["comparisons"][0]["n"] == 1 and sliced["comparisons"][0]["tag"] == "nutrition"
