"""Threshold selection of evals/cache_eval.py on synthetic vectors (no embedder, no network)."""

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
    rows = [{"id": "p1", "kind": "paraphrase", "a": "A", "b": "A2"},
            {"id": "n1", "kind": "near_miss", "a": "A", "b": "X"},
            {"id": "p2", "kind": "paraphrase", "a": "B", "b": "B2"}]
    vecs = {k: ce._unit(v) for k, v in {"A": [1, 0, 0], "A2": [0.97, 0.243, 0], "X": [0.905, 0.4254, 0],
                                        "B": [0, 0, 1], "B2": [0, 0.5, 0.866]}.items()}
    scored = ce.nearest(rows, vecs)
    assert [x["top_is_own"] for x in scored] == [True, True, True]
    assert ce.at_threshold(scored, 0.85)["false_hit_ids"] == ["n1"]
    assert ce.choose(scored) == 0.91  # X scores 0.905: the first grid value above it
    best = ce.at_threshold(scored, 0.91)
    assert best["false_hits"] == 0 and best["paraphrase_hit_rate"] == 50.0


def test_paraphrase_on_someone_elses_question_is_a_wrong_hit():
    ce = _mod()
    rows = [{"id": "p1", "kind": "paraphrase", "a": "A", "b": "A2"},
            {"id": "p2", "kind": "paraphrase", "a": "B", "b": "B2"}]
    vecs = {k: ce._unit(v) for k, v in {"A": [1, 0], "B": [0, 1], "A2": [0.99, 0.141], "B2": [0.945, 0.327]}.items()}
    scored = ce.nearest(rows, vecs)
    assert scored[1]["top_a"] == "A" and not scored[1]["top_is_own"]
    assert ce.at_threshold(scored, 0.9)["paraphrase_wrong_target"] == 1
    assert ce.choose(scored) == 0.95
