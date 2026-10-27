"""The pipeline of a test case: roadrunner, sbml2jax and the JAX simulation.

Stages of a case, each `pass`, `fail` or `skip`:

1. `roadrunner`: the SBML model simulated with roadrunner, compared with the
   expected results (checks the reference simulator and the settings);
2. `sbml2jax`: conversion of the SBML model into a JAX module;
3. `jax`: the module simulated with diffrax, compared with the expected
   results (the verification of work package 1).

A stage after a failed conversion is skipped. Every case runs in a worker
process of its own with a time limit, so a crash or a hanging solve fails the
case and not the run.
"""

import multiprocessing
import signal
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sbml2jax.testsuite.cases import Case, ensure_suite, load_cases, skip_reason
from sbml2jax.testsuite.compare import Comparison, compare, requested_frame, strip_brackets
from sbml2jax.testsuite.results import CaseResult, StageResult

#: solver settings of roadrunner and diffrax, tighter than the case tolerances
RTOL = 1e-9
ATOL = 1e-12
MAX_STEPS = 100_000
#: seconds per case before it fails
TIMEOUT = 120


def _stage(comparison: Comparison) -> StageResult:
    """Return the stage result of a comparison."""
    return StageResult("pass" if comparison.passed else "fail", comparison.message)


def _error(err: BaseException) -> str:
    """Return the message of an exception, with its type."""
    return f"{type(err).__name__}: {err}".strip()


def roadrunner_selections(case: Case, species: set[str]) -> list[str]:
    """Return the roadrunner selections of the case variables (`[S]` for concentrations)."""
    return [
        f"[{v}]" if v in species and v in case.settings.concentration else v for v in case.settings.variables
    ]


def run_roadrunner(case: Case) -> pd.DataFrame:
    """Simulate the SBML model of `case` with roadrunner."""
    import libsbml
    import roadrunner

    assert case.sbml_path is not None
    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_ERROR)
    doc = libsbml.readSBMLFromFile(str(case.sbml_path))
    species = {s.getId() for s in doc.getModel().getListOfSpecies()}
    rr = roadrunner.RoadRunner(str(case.sbml_path))
    rr.integrator.setValue("relative_tolerance", RTOL)
    rr.integrator.setValue("absolute_tolerance", ATOL)
    rr.integrator.setValue("maximum_num_steps", MAX_STEPS)
    rr.timeCourseSelections = ["time", *roadrunner_selections(case, species)]
    s = case.settings
    result = rr.simulate(s.start, s.end, s.steps + 1)
    return strip_brackets(pd.DataFrame(np.asarray(result), columns=list(result.colnames)))


def run_jax(module: Any, case: Case) -> pd.DataFrame:
    """Simulate a generated module at the time points of `case`."""
    import jax.numpy as jnp

    from sbml2jax.model import JaxModel, SolverSettings

    s = case.settings
    ts = jnp.linspace(s.start, s.end, s.steps + 1)
    model = JaxModel.from_module(module)
    ys = model.simulate(ts, settings=SolverSettings(rtol=RTOL, atol=ATOL, max_steps=MAX_STEPS))
    df = pd.DataFrame(np.asarray(ys), columns=list(module.OUTPUTS))
    df.insert(0, "time", np.asarray(ts))
    return df


def run_case(case: Case) -> CaseResult:
    """Run the stages of one case."""
    from sbml2jax.convert import convert_sbml2jax, load_module

    stages: dict[str, StageResult] = {}
    reason = skip_reason(case)
    if reason is not None:
        skip = StageResult("skip", reason)
        return CaseResult(case.id, {"roadrunner": skip, "sbml2jax": skip, "jax": skip}, list(case.test_tags))
    assert case.sbml_path is not None and case.expected is not None

    try:
        stages["roadrunner"] = _stage(compare(run_roadrunner(case), case.expected, case.settings))
    except Exception as err:  # noqa: BLE001
        stages["roadrunner"] = StageResult("fail", _error(err))

    try:
        source = convert_sbml2jax(case.sbml_path)
        module = load_module(source=source, name=f"case_{case.id}")
        stages["sbml2jax"] = StageResult("pass", "")
    except Exception as err:  # noqa: BLE001
        stages["sbml2jax"] = StageResult("fail", _error(err))
        stages["jax"] = StageResult("skip", "conversion failed")
        return CaseResult(case.id, stages, list(case.test_tags))

    try:
        df = run_jax(module, case)
        quantities = {sid: (only, comp) for sid, (comp, only) in module.SPECIES.items()}
        frame = requested_frame(df, quantities, case.settings)
        stages["jax"] = _stage(compare(frame, case.expected, case.settings))
    except Exception as err:  # noqa: BLE001
        stages["jax"] = StageResult("fail", _error(err) or traceback.format_exc(limit=1))
    return CaseResult(case.id, stages, list(case.test_tags))


def _timeout(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise TimeoutError(f"case exceeded {TIMEOUT} s")


def _worker(case_dir: Path) -> CaseResult:
    """Load and run one case in a worker process, with the time limit."""
    from sbml2jax.testsuite.cases import load_case

    case = load_case(case_dir)
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(TIMEOUT)
    try:
        return run_case(case)
    except TimeoutError as err:
        fail = StageResult("fail", str(err))
        return CaseResult(case.id, {"roadrunner": fail, "sbml2jax": fail, "jax": fail}, list(case.test_tags))
    finally:
        signal.alarm(0)


def run_suite(
    ids: Iterable[str] | None = None,
    processes: int | None = None,
    progress: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    """Run the cases `ids` (all when None) of the SBML test suite.

    Args:
        ids: case ids such as `00001`.
        processes: worker processes, the number of CPUs by default.
        progress: called with each result as it arrives.

    Returns:
        The results sorted by case id.
    """
    root = ensure_suite()
    cases = load_cases(root, list(ids) if ids is not None else None)
    results: list[CaseResult] = []
    # spawn, not fork: JAX and roadrunner are not fork-safe; one worker
    # process per 200 cases keeps the memory of compiled models bounded
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes, maxtasksperchild=200) as pool:
        for result in pool.imap_unordered(_worker, [c.case_dir for c in cases]):
            results.append(result)
            if progress is not None:
                progress(result)
    return sorted(results, key=lambda r: r.id)
