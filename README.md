# neural-ode

Hybrid neural-network ODE models for PK/PD: neural networks replace poorly
known terms of mechanistic ODE models (universal differential equations),
implemented in [JAX](https://github.com/jax-ml/jax) with
[diffrax](https://github.com/patrick-kidger/diffrax) and
[equinox](https://github.com/patrick-kidger/equinox). The application is the
[apixaban PBPK/PD model](https://github.com/matthiaskoenig/apixaban-model)
(Myshkina et al. 2026); [libRoadRunner](https://github.com/sys-bio/roadrunner)
is the reference simulator of the mechanistic part, and the hybrid models are
exchanged in the [PEtab SciML](https://petab-sciml.readthedocs.io) format.

M.Sc. thesis of Tom Oberländer, supervised by Matthias König.

## Modules

Every module is a self-contained Python package in its own directory, with
its own `pyproject.toml`, lock file, tests and CI workflow.

| module | content | status |
| --- | --- | --- |
| [`sbml2jax`](sbml2jax/) | SBML to differentiable JAX code, diffrax simulation, verification against roadrunner and the SBML test suite | work package 1: apixaban model verified, 998 of 1823 test suite cases pass |

Planned: hybrid models and training (work package 2), evaluation (work
package 3), PEtab SciML export (work package 4).

## Installation

```bash
cd sbml2jax && uv sync --extra dev        # development of a module
pip install "sbml2jax[testsuite] @ git+https://github.com/matthiaskoenig/neural-ode.git#subdirectory=sbml2jax"
```

## License

MIT, see [LICENSE](LICENSE). Models in `*/examples/models/` keep the licenses
stated next to them.
