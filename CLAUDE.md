# CLAUDE.md

Monorepo of the M.Sc. thesis "Hybrid neural-network ODE models for PK/PD"
(supervisor Matthias König). Every top-level directory is a module: a
self-contained Python package with its own `pyproject.toml`, `uv.lock`,
`tests/`, `CLAUDE.md` and a CI workflow `.github/workflows/<module>.yml` that
runs with `working-directory: <module>` and only on changes under it. Run the
commands of a module from its directory.

- `sbml2jax/`: SBML to JAX converter and the roadrunner / SBML test suite
  verification (work package 1), see `sbml2jax/CLAUDE.md`.

New modules (hybrid models, PEtab SciML export) go into new top-level
directories and depend on `sbml2jax` as a package, not by importing its files.
`.pre-commit-config.yaml` at the root serves all modules; ruff uses the
`.ruff.toml` of each module.
