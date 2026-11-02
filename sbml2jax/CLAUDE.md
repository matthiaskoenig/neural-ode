# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project

`sbml2jax` generates differentiable JAX code for the ODE system of an SBML
model and simulates it with diffrax. It is work package 1 of the M.Sc. thesis
"Hybrid neural-network ODE models for PK/PD" (supervisor Matthias König): the
mechanistic baseline into which work package 2 puts neural networks. The
structure follows [sbml2cellml](https://github.com/matthiaskoenig/sbml2cellml).
Python >= 3.11 (3.11 for Colab; CI and `.python-version` use 3.13), packaged with hatchling (version in `src/sbml2jax/__init__.py`),
environment pinned by `uv.lock`. Runtime dependencies: `python-libsbml`, `jax`,
`diffrax`, `equinox`, `rich`; the `testsuite` extra adds `libroadrunner` (the
reference simulator), `pandas`, `requests`, `matplotlib`.

## Repository

`sbml2jax` is a module of the monorepo `matthiaskoenig/neural-ode`; this
directory is a self-contained package (own `pyproject.toml`, `uv.lock`, tests).
Run all commands below from `sbml2jax/`. CI: `.github/workflows/sbml2jax.yml`
at the repository root (`working-directory: sbml2jax`, runs only on changes
under `sbml2jax/`); pre-commit: `.pre-commit-config.yaml` at the root.

## Commands

```bash
uv sync --extra dev
uv run pre-commit install

uv run pytest                                  # unit tests + apixaban roadrunner check
uv run pytest tests/test_apixaban.py
tox r -e ruff                                  # lint and format check
SBML2JAX_TESTSUITE=1 tox r -e testsuite        # gate: no regressions against testsuite/results.json

uv run sbml2jax model.xml -o model.py          # write the JAX module of a model
uv run python examples/apixaban_example.py     # apixaban vs roadrunner, gradient check

uv run sbml2jax-testsuite run                  # full SBML test suite (~40 min on 2 CPUs), writes testsuite/results.json and docs/testsuite.md
uv run sbml2jax-testsuite run 00001 00950      # selected cases, merged into the results
uv run sbml2jax-testsuite report               # rerender docs/testsuite.md
```

## Architecture

The pipeline is `sbml.read_document` -> `analysis.analyse` -> `codegen.generate`
(`convert.convert_sbml2jax`), the generated module is loaded with
`convert.load_module` and wrapped by `model.JaxModel`.

- `sbml.py`: reads the SBML file with libsbml, flattens comp models
  (`flatten comp` conversion), expands function definitions
  (`expandFunctionDefinitions`). Afterwards the model is core SBML.
- `jaxmath.py`: `translate(ast, resolve, rate_of) -> Expr` walks a libsbml
  `ASTNode` into the source of a `jnp` expression and collects the identifiers
  it uses (`Expr.deps`). Formulas are never printed and reparsed. piecewise ->
  nested `jnp.where` (no otherwise: NaN), relations chained with
  `logical_and`, `log` with one argument is log10, `factorial` via `gammaln`,
  `rem` -> `fmod`, `quotient` -> `trunc`. delay, lambda, unexpanded function
  calls raise `UnsupportedError`.
- `analysis.py`: `analyse(model) -> ModelSystem`, the work libcellml's
  analyser does in sbml2cellml. Every identifier has one role during the
  simulation: state (`y`), constant (`c`) or computed (assignment rules,
  concentrations, reaction rates, derivatives `d_<state>`). A species which no
  rule determines has an amount `<id>_amount` (the id itself with
  `hasOnlySubstanceUnits`), which is the state when reactions change it and a
  constant otherwise; its concentration is `amount / compartment`, so a
  changing compartment keeps the amount as SBML requires. A rate rule species
  integrates its value as is. Reaction terms are `stoichiometry * rate`
  (reactants negative, boundary species excluded), times the conversion
  factor. Local parameters are `<reaction>_<id>`. The values at the start
  (`ModelSystem.initial`) are initial assignment > assignment rule > attribute;
  the attributes are the inputs `p`, so gradients reach every number of the
  model, also through initial assignments (they are not evaluated by libsbml
  as in sbml2cellml, which would cut the gradient). rateOf is resolved to the
  derivative of a state, `(dA/dt - S * dC/dt) / C` for a concentration, 0 for a
  constant. A model without ODEs gets a constant `dummy_state`. Not supported:
  events, algebraic rules, delays, fast reactions, rateOf of rule targets and
  in initial values (see `docs/conversion-issues.md`).
- `codegen.py`: writes the module: `INPUTS`, `INPUT_VALUES`, `STATES`,
  `CONSTANTS`, `OUTPUTS`, `SPECIES`, `initial(p, t)`, `rhs(t, y, c)`,
  `outputs(t, y, c)`. `order` emits only the definitions a function needs, in
  topological order, and raises `CodegenError` on cycles. The generated code is
  a fixed contract read by `model.py` and the test suite runner.
- `convert.py`: `convert_sbml2jax(sbml_path, module_path=None) -> source`,
  `load_module(path=..., source=...)`.
- `model.py`: `JaxModel(eqx.Module)` with the inputs `p` as only array field
  and the module static; `jax.grad` of a loss of `simulate` is the gradient
  w.r.t. all inputs. `simulate(ts, segments, settings)` solves with diffrax
  (`Kvaerno5`, `PIDController`); a `Segment` restarts the integration with
  changes of states or constants (dosing). `rhs` is the hook for hybrid
  models (work package 2). `ts` must be concrete (Python level segment logic).
- `variables.py`: `Names`, the identifiers of the generated module: SBML ids as
  they are, keywords and the reserved `t`, `y`, `c`, `p`, `jnp`, `jax`,
  `special` with a trailing underscore, derived names with numeric suffixes.
- `cli.py`, `console.py`, `log.py`: command line, rich console, opt-in rich
  logging. Library code logs with `logging.getLogger(__name__)`, never prints.
- `testsuite/`: `sbml2jax-testsuite run|report`. `cases.py` and `compare.py`
  are adapted from sbml2cellml (download of the suite 3.5.0 into
  `~/.cache/sbml2jax`, settings, `|value - expected| <= absolute + relative *
  |expected|` with the tolerances of the case). `runner.py` runs the stages
  `roadrunner`, `sbml2jax`, `jax` per case in spawned worker processes with a
  time limit (`TIMEOUT`), solver tolerances `RTOL`/`ATOL` 1e-9/1e-12;
  `results.py` (JSON, `regressions`, `improvements`), `report.py`
  (`docs/testsuite.md`, failures grouped by reason).

## Conventions

- `testsuite/results.json` (all 1823 cases) and `docs/testsuite.md` are
  generated and committed; `tests/test_testsuite_full.py` reruns 60 evenly
  spread passing cases (`gate_ids`) and fails on regressions. After converter
  changes rerun the full suite and commit the regenerated files.
  `CaseResult.category` is pass / fail / unsupported (conversion failed) / skip.
- Known conversion gaps are documented in `docs/conversion-issues.md`; fixing a
  gap updates the document and the results.
- `examples/models/` holds the apixaban models (Myshkina et al. 2026,
  https://github.com/matthiaskoenig/apixaban-model, commit d0e3277); generated
  modules go to the gitignored `examples/results/`.
- JAX runs in double precision (`jax_enable_x64` in `__init__.py`).
- Google style docstrings on modules, classes and functions (ruff `D`).
