"""Gate against the committed results of the SBML test suite: no regressions."""

import os
from pathlib import Path

import pytest

from sbml2jax.testsuite import results
from sbml2jax.testsuite.runner import run_suite

RESULTS = Path(__file__).parent.parent / "testsuite" / "results.json"


@pytest.mark.testsuite
@pytest.mark.skipif(not os.environ.get("SBML2JAX_TESTSUITE"), reason="set SBML2JAX_TESTSUITE=1")
def test_no_regressions() -> None:
    old = results.load(RESULTS)
    # only the cases of the committed results, `sbml2jax-testsuite run` adds more
    new = run_suite([r.id for r in old])
    assert results.regressions(old, new) == []
