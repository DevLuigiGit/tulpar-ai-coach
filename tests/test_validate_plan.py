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
