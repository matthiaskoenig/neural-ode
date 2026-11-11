"""`sbml2jax-testsuite`: run the SBML test suite and write the report.

sbml2jax-testsuite run                 # all cases
sbml2jax-testsuite run 00001 00002     # selected cases
sbml2jax-testsuite run --first 200     # the first 200 cases
sbml2jax-testsuite report              # rerender the report from the results
"""

import argparse
from pathlib import Path

from sbml2jax.console import console
from sbml2jax.testsuite import report, results
from sbml2jax.testsuite.cases import ensure_suite
from sbml2jax.testsuite.runner import run_suite

RESULTS = Path("testsuite/results.json")
REPORT = Path("docs/testsuite.md")


def main(argv: list[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(
        prog="sbml2jax-testsuite", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run cases and write results and report")
    run.add_argument("ids", nargs="*", help="case ids, all by default")
    run.add_argument("--first", type=int, help="run the first N cases")
    run.add_argument("-j", "--processes", type=int, help="worker processes")
    sub.add_parser("report", help="rerender the report from the results")
    args = parser.parse_args(argv)

    if args.command == "report":
        REPORT.write_text(report.render(results.load(RESULTS)))
        return 0

    ids: list[str] | None = args.ids or None
    if args.first:
        ids = sorted(p.name for p in ensure_suite().iterdir() if p.name.isdigit())[: args.first]
    old = results.load(RESULTS) if RESULTS.exists() else []

    def progress(result: results.CaseResult) -> None:
        status = " ".join(f"{k}={v.status}" for k, v in result.stages.items())
        console.print(f"{result.id} {status}", highlight=False)

    new = run_suite(ids, args.processes, progress)
    if ids is not None and old:
        # a partial run updates the results of its cases only
        merged = {r.id: r for r in old} | {r.id: r for r in new}
        new_all = sorted(merged.values(), key=lambda r: r.id)
    else:
        new_all = new
    results.save(new_all, RESULTS)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report.render(new_all))

    passed = sum(r.passed for r in new)
    console.print(f"jax stage passed: {passed} of {len(new)}")
    if old:
        console.print(f"regressions: {results.regressions(old, new_all) or 'none'}")
        console.print(f"improvements: {len(results.improvements(old, new_all))}")
    return 0
