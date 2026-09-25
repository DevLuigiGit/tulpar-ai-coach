"""Agreement math of evals/judge_agreement.py on hand-computed examples, and the human-check round trip."""

import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("evals_judge_agreement", ROOT / "evals" / "judge_agreement.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evals_judge_agreement"] = mod
    spec.loader.exec_module(mod)
    return mod


ja = _mod()


def test_cohen_kappa_textbook_example():
    # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 → po = 0.7, pe = 0.5, kappa = 0.4
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert ja.cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_extremes_and_degenerate():
    assert ja.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert ja.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)
    # everybody gives 5: agreement is 100% but kappa is undefined (chance agreement is total too)
    assert ja.cohen_kappa([5, 5, 5], [5, 5, 5]) is None
    # None means "judge failed": such pairs are dropped, not counted as disagreement
    assert ja.cohen_kappa([1, None, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_cohen_kappa_high_prevalence_paradox():
    # one rater passes everything: 90% agreement, yet kappa is 0 — no better than chance at this prevalence
    a = [True] * 10
    b = [True] * 9 + [False]
    assert ja.pair_stats([5] * 10, [5] * 9 + [3])["pass_agree_pct"] == 90.0
    assert ja.cohen_kappa(a, b, labels=[False, True]) == pytest.approx(0.0)


def test_quadratic_kappa_matches_closed_form():
    # quadratic weighted kappa == 2·cov / (var_a + var_b + (mean_a − mean_b)²), population moments
    rng = random.Random(7)
    a = [rng.randint(1, 5) for _ in range(60)]
    b = [min(5, max(1, x + rng.choice([-1, 0, 0, 1]))) for x in a]
    ma, mb = statistics.mean(a), statistics.mean(b)
    cov = statistics.mean((x - ma) * (y - mb) for x, y in zip(a, b))
    closed = 2 * cov / (statistics.pvariance(a) + statistics.pvariance(b) + (ma - mb) ** 2)
    assert ja.cohen_kappa(a, b, labels=ja.SCALE, weights="quadratic") == pytest.approx(closed)


def test_linear_weights_small_example():
    # a=[1,2,3], b=[1,3,3] on scale 1..3, weight |i−j|/2: observed disagreement (½)/3 = 1/6,
    # expected ⅓·⅔ + ⅓·½ + ⅓·⅓ = 1/2 → kappa = 1 − (1/6)/(1/2) = 2/3
    assert ja.cohen_kappa([1, 2, 3], [1, 3, 3], labels=[1, 2, 3], weights="linear") == pytest.approx(2 / 3)


def test_fleiss_kappa_hand_computed():
    items = [[1, 1, 1], [0, 0, 0], [1, 1, 0], [1, 0, 0]]  # P̄ = 2/3, Pe = 1/2 → 1/3
    assert ja.fleiss_kappa(items) == pytest.approx(1 / 3)
    assert ja.fleiss_kappa([[1, 1, 1], [0, 0, 0]]) == pytest.approx(1.0)
    assert ja.fleiss_kappa([[1, 1, 1], [1, 1, 1]]) is None
    assert ja.fleiss_kappa([[1], [0]]) is None  # a single rater


def test_spearman_with_ties():
    assert ja.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert ja.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # ranks x = [1, 2.5, 2.5, 4] vs [1, 2, 3, 4] → 4.5 / sqrt(4.5 · 5)
    assert ja.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(4.5 / (4.5 * 5) ** 0.5)
    assert ja.spearman([5, 5, 5, 5], [1, 2, 3, 4]) is None


def test_pair_stats():
    s = ja.pair_stats([5, 5, 4, 3, 2, None], [5, 4, 3, 3, 4, 5])
    assert s["n"] == 5
    assert s["exact_pct"] == 40.0
    assert s["pass_agree_pct"] == 60.0  # (5,5) (5,4) (3,3) agree on pass/fail; (4,3) and (2,4) do not
    assert s["mad"] == pytest.approx((0 + 1 + 1 + 0 + 2) / 5)


EMPTY_REPLY = "all providers failed: ollama:kimi-k2.6: JSONDecodeError: no JSON value found: line 1 column 1 (char 0)"


def test_retry_waits():
    assert ja._retry_wait("groq 429: Please try again in 7.5s", 0) == pytest.approx(10)
    assert ja._retry_wait("groq 429: Please try again in 1m2s", 0) == pytest.approx(63)
    assert ja._retry_wait("ollama 503: overloaded", 2) == pytest.approx(40)
    assert ja._retry_wait('ollama 410: {"error":"model was retired"}', 0) is None
    # an empty or non-JSON reply is intermittent: asked again soon, but only MALFORMED_RETRIES times
    assert ja._retry_wait(EMPTY_REPLY, 0) == pytest.approx(3)
    assert ja._retry_wait("expected a JSON object, got list", 1) == pytest.approx(6)
    assert ja._retry_wait(EMPTY_REPLY, ja.MALFORMED_RETRIES) is None
    # a column number in a decode error is not an HTTP 5xx
    assert ja._retry_wait("Expecting value: line 1 column 512 (char 511)", 0) is None


def test_only_scores_and_hard_refusals_are_final():
    assert ja.is_final({"score": 3})
    assert ja.is_final({"score": None, "error": 'ollama 410: {"error":"model was retired"}'})
    assert ja.is_final({"score": None, "error": "all providers failed: groq:x: LLMError: groq 400: bad request"})
    assert not ja.is_final({"score": None, "error": EMPTY_REPLY})
    assert not ja.is_final({"score": None, "error": "groq 429: Please try again in 2s"})
    assert not ja.is_final({"score": None, "error": "retries exhausted"})


async def test_empty_reply_is_retried_within_the_call(env, monkeypatch):
    """kimi sometimes answers with nothing and scores the same input on the next call."""
    from types import SimpleNamespace

    from tulpar_ai import llm

    replies = iter(["", json.dumps({"score": 4, "reason": "ok"})])
    llm.set_fake(lambda **_: next(replies))
    waits: list[float] = []

    async def no_sleep(sec):
        waits.append(sec)

    monkeypatch.setattr(ja, "asyncio", SimpleNamespace(sleep=no_sleep))
    try:
        res = await ja.judge_once("ollama:kimi-k2.6", "correctness", "u")
    finally:
        llm.set_fake(None)
    assert res["score"] == 4 and res["attempts"] == 2 and waits == [3.0]


async def test_judge_once_gives_up_on_malformed_after_budget(env, monkeypatch):
    from types import SimpleNamespace

    from tulpar_ai import llm

    calls: list[int] = []

    def always_empty(**_):
        calls.append(1)
        return ""

    async def no_sleep(sec):
        pass

    llm.set_fake(always_empty)
    monkeypatch.setattr(ja, "asyncio", SimpleNamespace(sleep=no_sleep))
    try:
        res = await ja.judge_once("ollama:kimi-k2.6", "faithfulness", "u")
    finally:
        llm.set_fake(None)
    assert res["score"] is None and "no JSON value" in res["error"]
    assert len(calls) == ja.MALFORMED_RETRIES + 1  # not the 5 retries a rate limit gets


async def test_failed_calls_are_cached_but_asked_again_next_run(tmp_path, monkeypatch):
    """An empty reply or a 429 is a failure in this run's stats, but a later run asks that row again."""
    replies = {"retired": [{"score": None, "error": 'ollama 410: {"error":"model was retired"}'}],
               "flaky": [{"score": None, "error": EMPTY_REPLY}, {"score": 4, "reason": "ok"}],
               "limited": [{"score": None, "error": "groq 429: Please try again in 2s"}, {"score": 5, "reason": "ok"}]}
    asked: list[str] = []

    async def fake_once(spec, kind, user, retries=5):
        name = spec.split(":", 1)[1]
        asked.append(name)
        return dict(replies[name].pop(0) if len(replies[name]) > 1 else replies[name][0])

    monkeypatch.setattr(ja, "judge_once", fake_once)
    cache = ja.Cache(tmp_path / "cache.json")
    assert "410" in await ja.probe("ollama:retired", "q1", "u", cache)
    assert asked == ["retired"]
    row = {"id": "q1", "sources": "s", "answer": "a", "answerable": False}
    specs = ["ollama:retired", "ollama:flaky", "groq:limited"]
    await ja.score_all(specs, [row], cache, None)
    first = {k.split("|")[0]: v for k, v in ja.Cache(tmp_path / "cache.json").data.items()}
    assert [first[s]["score"] for s in specs] == [None, None, None]  # all three recorded as failures

    asked.clear()
    await ja.score_all(specs, [row], cache, None)
    assert sorted(asked) == ["flaky", "limited"]  # the retired judge is final, the other two are asked again
    again = {k.split("|")[0]: v for k, v in cache.data.items()}
    assert again["ollama:flaky"]["score"] == 4 and again["ollama:flaky"]["runs"] == 2
    assert again["groq:limited"]["score"] == 5

    asked.clear()
    await ja.score_all(specs, [row], cache, None)
    assert asked == []  # scores are final


async def test_judge_user_matches_evals_runner(env, monkeypatch):
    """The agreement study must judge exactly what evals/run.py judges."""
    from tulpar_ai import llm

    seen: list[str] = []

    def fake(*, role, system, user, images, json_mode):
        seen.append(user)
        return json.dumps({"score": 4, "reason": "ok"})

    llm.set_fake(fake)
    try:
        kw = dict(sources="[1] Жим\nтекст", answer="Ответ [1].", question="Как жать?", reference="Эталон")
        for kind in ("faithfulness", "correctness"):
            assert await ja.ev.judge(kind, **kw) == 4
            assert seen[-1] == ja.judge_user(kind, **kw)
            res = await ja.judge_once("ollama:any", kind, ja.judge_user(kind, **kw))
            assert res["score"] == 4
    finally:
        llm.set_fake(None)


def _results(live: list[str]) -> dict:
    rows = []
    for i in range(14):
        faith = {n: 5 for n in live}
        if i < 3:
            faith[live[0]] = 2 + i  # 2, 3, 4: one judge flags rows 0 and 1
        faith[ja.STORED] = 5
        rows.append({"id": f"q{i:02d}", "tags": [["technique", "nutrition", "who"][i % 3]], "answer_chars": 100 + i,
                     "out_tokens": 50 + i, "scores": {"faithfulness": faith, "correctness": {}}})
    return {"summary": {"live_judges": live}, "rows": rows}


def test_human_sample_and_agreement_round_trip(tmp_path):
    live = ["judge-a", "judge-b"]
    res = _results(live)
    sample = ja.pick_human_sample(res["rows"], live)
    ids = [s["id"] for s in sample]
    assert len(ids) == 10 and len(set(ids)) == 10
    assert [s["stratum"] for s in sample[:2]] == ["contested", "contested"]
    assert {"tag:nutrition", "tag:technique", "tag:who"} <= {s["stratum"] for s in sample}

    inputs = [{"id": r["id"], "question": f"вопрос {r['id']}", "sources": "[1] Источник\nтекст, с запятой",
               "answer": "ответ\nв две строки"} for r in res["rows"]]
    path = tmp_path / "sample.csv"
    ja.write_human_csv(path, sample, inputs)
    text = path.read_text(encoding="utf-8-sig")
    assert "judge-a" not in text and "score" not in text  # no judge scores for the labeller
    assert ja.read_human_labels(path) == {}

    import csv

    with path.open(encoding="utf-8-sig", newline="") as fh:
        recs = list(csv.DictReader(fh))
    # the sample lists contested rows first; the file must not, or its order says which rows a judge failed
    assert [r["id"] for r in recs] == sorted(ids) != ids

    # a person fills the column: 5 everywhere except q00 → 2
    for rec in recs:
        rec[ja.HUMAN_COLUMN] = "2" if rec["id"] == "q00" else "5"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    labels = ja.read_human_labels(path)
    assert len(labels) == 10 and labels["q00"] == 2

    out = ja.human_agreement(labels, res)
    assert out["n_labelled"] == 10
    a = out["vs"]["judge-a"]  # judge-a: q00=2 (agrees), q01=3 (human 5), rest 5
    assert a["exact_pct"] == 80.0 and a["pass_agree_pct"] == 90.0
    assert out["vs"]["judge-b"]["pass_agree_pct"] == 90.0  # misses q00
    assert out["vs"]["median_of_live_judges"]["n"] == 10


def test_bad_human_label_is_rejected(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text(f"id,question,sources,answer,{ja.HUMAN_COLUMN},comment\nq1,в,и,о,6,\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        ja.read_human_labels(path)


def test_krippendorff_alpha_reference_example():
    """Krippendorff (2011), "Computing Krippendorff's alpha-reliability": 4 coders, 12 units, gaps included."""
    n = None
    coders = [[1, 2, 3, 3, 2, 1, 4, 1, 2, n, n, n],
              [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, n, 3],
              [n, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, n],
              [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, n]]
    units = [list(u) for u in zip(*coders)]
    assert ja.krippendorff_alpha(units) == pytest.approx(0.743, abs=5e-4)
    assert ja.krippendorff_alpha(units, "interval") == pytest.approx(0.849, abs=5e-4)
    assert ja.krippendorff_alpha([[5, 5, 5], [5, None, 5]]) is None  # one category only: undefined
    assert ja.krippendorff_alpha([[1, None, None]]) is None  # nothing pairable


def test_panel_agreement_does_not_hide_rows_a_judge_failed():
    """Judge c fails exactly where a and b disagree: complete-row stats look perfect, the report must not."""
    live = ["a", "b", "c"]
    units = [[f"q{i}", [5, 5, 5]] for i in range(8)] + [[f"q{i}", [2, 2, 2]] for i in range(8, 10)]
    units += [["q10", [2, 4, None]], ["q11", [4, 3, None]]]
    out = ja.panel_agreement(units, live)
    assert out["live_all_scored"] == 10
    assert out["all_live_pass_agree_pct"] == 100.0 and out["fleiss_kappa_pass"] == pytest.approx(1.0)
    assert [r["id"] for r in out["incomplete_rows"]] == ["q10", "q11"]
    assert out["incomplete_rows"][0]["scores"] == {"a": 2, "b": 4, "c": None}
    assert out["incomplete_pass_split"] == 2
    assert out["krippendorff_alpha_pass"] < 0.8  # the two split rows count here


def test_pass_spread_compares_judges_on_the_same_rows():
    """A judge that fails on the hard rows would look more lenient on its own rows."""
    live = ["a", "b"]
    units = [[f"q{i}", [5, 5]] for i in range(8)] + [["q8", [3, 3]], ["q9", [2, None]]]
    out = ja.panel_agreement(units, live)
    assert out["pass_pct_common"] == {"a": 88.9, "b": 88.9} and out["pass_pct_spread"] == 0.0
    # on own rows: a 8/10 = 80%, b 8/9 = 88.9% — a spread of 8.9 points that is only a row-set artefact
    per_judge = [ja.judge_stats([u[1][j] for u in units], [100] * 10, [50] * 10) for j in range(2)]
    assert [pj["pass_pct"] for pj in per_judge] == [80.0, 88.9]


def test_spearman_permutation_p():
    # n = 4, perfect order: exactly 2 of the 24 orderings reach |rho| = 1 → p = 1/12
    assert ja.spearman_perm_p([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1 / 12, abs=0.015)
    x = list(range(30))
    assert ja.spearman_perm_p(x, x) < 0.001
    assert ja.spearman_perm_p(x, x) == ja.spearman_perm_p(x, x)  # seeded: the same number on every run
    assert ja.spearman_perm_p([5] * 5, [1, 2, 3, 4, 5]) is None
    assert ja.spearman_perm_p([1, 2, 1, 2, 1, 2, 1, 2], [1, 1, 2, 2, 3, 3, 4, 4]) > 0.5


def test_summary_flags_length_effects():
    live = ["a"]
    rows = []
    for i in range(20):
        # a is harsher on longer answers, all under the 400-char cut
        rows.append({"id": f"q{i:02d}", "answer_chars": 100 + 10 * i, "out_tokens": 50 + i,
                     "scores": {"faithfulness": {"a": 5 if i < 14 else 3, ja.STORED: None},
                                "correctness": {"a": 5, ja.STORED: None}}})
    summary = ja.summarize(rows, live, None)
    flags = summary["length_effects_p_lt_0_05"]
    assert {(f["metric"], f["judge"], f["subset"]) for f in flags} == {("faithfulness", "a", "all"),
                                                                       ("faithfulness", "a", "uncut")}
    assert all(f["rho"] < 0 and f["p"] < 0.05 and f["n"] == 20 for f in flags)


def test_scoring_history_survives_reruns():
    legacy = {"measured_at": "2026-09-25 21:12", "usage_scoring_run": {"kimi-k2.6": {"calls": 83}}}
    assert ja.scoring_history(legacy) == [{"at": "2026-09-25 21:12", "usage": {"kimi-k2.6": {"calls": 83}}}]
    runs = [{"at": "a", "usage": {}}, {"at": "b", "usage": {}}]
    assert ja.scoring_history({"scoring_runs": runs}) == runs and ja.scoring_history({}) == []
    calls = [{"model": "m", "in": 10, "out": 2, "ms": 100}, {"model": "m", "in": 20, "out": 4, "ms": 300}]
    assert ja.usage_by_model(calls) == {"m": {"calls": 2, "tokens_in": 30, "tokens_out": 6, "p50_ms": 200}}
