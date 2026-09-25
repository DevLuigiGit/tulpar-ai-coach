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


def test_retry_only_on_transient_errors():
    assert ja._retry_wait("groq 429: Please try again in 7.5s", 0) == pytest.approx(10)
    assert ja._retry_wait("groq 429: Please try again in 1m2s", 0) == pytest.approx(63)
    assert ja._retry_wait("ollama 503: overloaded", 2) == pytest.approx(40)
    assert ja._retry_wait('ollama 410: {"error":"model was retired"}', 0) is None
    assert ja._retry_wait("expected a JSON object", 0) is None


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

    # a person fills the column: 5 everywhere except q00 → 2
    import csv

    with path.open(encoding="utf-8-sig", newline="") as fh:
        recs = list(csv.DictReader(fh))
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
