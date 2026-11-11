# API reference

The API of `sbml2jax`, rendered from the docstrings of the code. Most users need only [`sbml2jax.model`](model.md): `JaxModel`, `Segment` and `SolverSettings`.

| module | content |
| --- | --- |
| [`sbml2jax.model`](model.md) | `JaxModel`, the differentiable model and its simulation; `Segment`, `SolverSettings` |
| [`sbml2jax.convert`](convert.md) | `convert_sbml2jax`, `load_module` |
| [`sbml2jax.sbml`](sbml.md) | reading and preparing SBML with libsbml |
| [`sbml2jax.analysis`](analysis.md) | `analyse`, the ODE system of a model as `ModelSystem` |
| [`sbml2jax.jaxmath`](jaxmath.md) | `translate`, formulas to `jax.numpy` code |
| [`sbml2jax.codegen`](codegen.md) | `generate`, the source of the module |
| [`sbml2jax.variables`](variables.md) | `Names`, the identifiers of the generated code |
| [`sbml2jax.cli`](cli.md) | the command line `sbml2jax` |
| [`sbml2jax.testsuite`](testsuite.md) | the SBML test suite harness |

::: sbml2jax
