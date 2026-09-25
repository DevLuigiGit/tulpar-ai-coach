"""Threshold selection of evals/cache_eval.py on synthetic vectors (no embedder, no network).

Texts without digits or guard words, so only the vectors decide unless a test says otherwise.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("cache_eval", ROOT / "evals" / "cache_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cache_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_lowest_threshold_without_wrong_hits():
    ce = _mod()
    rows = [{"id": "p1", "kind": "paraphrase", "a": "A", "b": "AA"},
            {"id": "n1", "kind": "near_miss", "a": "A", "b": "X"},
            {"id": "p2", "kind": "paraphrase", "a": "B", "b": "BB"}]
    vecs = {k: ce._unit(v) for k, v in {"A": [1, 0, 0], "AA": [0.97, 0.243, 0], "X": [0.905, 0.4254, 0],
                                        "B": [0, 0, 1], "BB": [0, 0.5, 0.866]}.items()}
    scored = ce.nearest(rows, vecs)
    assert [x["top_is_own"] for x in scored] == [True, True, True]
    assert ce.at_threshold(scored, 0.85)["false_hit_ids"] == ["n1"]
    assert ce.choose(scored) == 0.91  # X scores 0.905: the first grid value above it
    best = ce.at_threshold(scored, 0.91)
    assert best["false_hits"] == 0 and best["paraphrase_hit_rate"] == 50.0


def test_paraphrase_on_someone_elses_question_is_a_wrong_hit():
    ce = _mod()
    rows = [{"id": "p1", "kind": "paraphrase", "a": "A", "b": "AA"},
            {"id": "p2", "kind": "paraphrase", "a": "B", "b": "BB"}]
    vecs = {k: ce._unit(v) for k, v in {"A": [1, 0], "B": [0, 1], "AA": [0.99, 0.141], "BB": [0.945, 0.327]}.items()}
    scored = ce.nearest(rows, vecs)
    assert scored[1]["top_a"] == "A" and not scored[1]["top_is_own"]
    assert ce.at_threshold(scored, 0.9)["paraphrase_wrong_target"] == 1
    assert ce.choose(scored) == 0.95


def test_guard_in_the_sweep_matches_production():
    """A flip scored above every threshold is a wrong hit without the guard and no hit with it."""
    ce = _mod()
    fem, male = "Сколько белка нужно женщине в день?", "Сколько белка нужно мужчине в день?"
    rows = [{"id": "n1", "kind": "near_miss", "batch": "flips-new", "a": fem, "b": male},
            {"id": "p1", "kind": "paraphrase", "a": male + " ", "b": "Мужчине сколько белка нужно в день?"}]
    vecs = {k: ce._unit(v) for k, v in {fem: [1, 0.01], male: [1, 0], male + " ": [1, 0.02],
                                        "Мужчине сколько белка нужно в день?": [1, 0.03]}.items()}
    raw, guarded = ce.nearest(rows, vecs, guard=False), ce.nearest(rows, vecs)
    assert raw[0]["top_score"] > 0.99 and ce.at_threshold(raw, 0.99)["false_hit_ids"] == ["n1"]
    assert guarded[0]["top_a"] == male + " " and not guarded[0]["top_is_own"]  # the stored male question, not the female one
    assert guarded[0]["pair_guard"] == ["sex"]
    assert guarded[1]["top_is_own"] and guarded[1]["top_score"] > 0.99
