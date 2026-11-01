"""Gate against the committed results of the SBML test suite: no regressions."""

import os
from pathlib import Path

import pytest

from sbml2jax.testsuite import results
from sbml2jax.testsuite.runner import run_suite

RESULTS = Path(__file__).parent.parent / "testsuite" / "results.json"
#: cases run by the gate; the full suite (~1 h) is run with `sbml2jax-testsuite run`
GATE_SIZE = 60


def gate_ids(committed: list[results.CaseResult]) -> list[str]:
    """Return an evenly spread selection of the committed passing cases."""
    passing = [r.id for r in committed if r.passed]
    step = max(len(passing) // GATE_SIZE, 1)
    return passing[::step][:GATE_SIZE]


@pytest.mark.testsuite
@pytest.mark.skipif(not os.environ.get("SBML2JAX_TESTSUITE"), reason="set SBML2JAX_TESTSUITE=1")
def test_no_regressions() -> None:
    old = results.load(RESULTS)
    ids = set(gate_ids(old))
    new = run_suite(sorted(ids))
    assert results.regressions([r for r in old if r.id in ids], new) == []
