# Installation

Every module of the repository is a self-contained Python package in its own directory, with its own `pyproject.toml`, `uv.lock`, tests and CI workflow. Today there is one module, [`sbml2jax`](sbml2jax/index.md). It needs Python 3.11 or newer (3.11 so that it runs in Google Colab; development and CI use 3.13).

## With pip

`sbml2jax` is not on PyPI, install it from GitHub:

```bash
pip install "sbml2jax @ git+https://github.com/matthiaskoenig/neural-ode.git#subdirectory=sbml2jax"
```

The `testsuite` extra adds what the verification needs: [libroadrunner](https://libroadrunner.org) as the reference simulator, pandas, requests and matplotlib. The [first simulation](index.md#a-first-simulation) plots with matplotlib, so install it with the extra:

```bash
pip install "sbml2jax[testsuite] @ git+https://github.com/matthiaskoenig/neural-ode.git#subdirectory=sbml2jax"
```

The same line works in a Google Colab notebook with a leading `!`.

| extra | content |
| --- | --- |
| (none) | `python-libsbml`, `jax`, `diffrax`, `equinox`, `rich` |
| `testsuite` | in addition `libroadrunner`, `pandas`, `requests`, `matplotlib` |
| `dev` | in addition `ruff`, `pre-commit`, `tox`, `pytest` |

!!! note "Double precision"
    `import sbml2jax` switches JAX to double precision (`jax_enable_x64`). PK/PD models span many orders of magnitude, and the comparison with roadrunner needs it.

## For development

Clone the repository and let [uv](https://docs.astral.sh/uv/) create the environment of the module from its lock file:

```bash
git clone https://github.com/matthiaskoenig/neural-ode.git
cd neural-ode/sbml2jax
uv sync --extra dev
uv run pre-commit install
uv run pytest
```

Run the commands of a module from its directory. See [Contributing](development.md) for the checks and the documentation.

## Check the installation

```python
import sbml2jax

print(sbml2jax.__version__)
```

```bash
sbml2jax --help
```
