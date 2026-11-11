# Roadmap

The thesis builds a simulator for hybrid PK/PD models in which neural networks replace poorly known terms of a mechanistic ODE model. It starts from the apixaban PBPK/PD model ([Myshkina *et al.* 2026](references.md#models)), implements the hybrid simulator in JAX independently of existing hybrid-modelling tools, verifies it against libRoadRunner, and exports the final models in the PEtab SciML format so they can be exchanged.

Every work package is a module of the monorepo: a self-contained Python package in a top-level directory with its own `pyproject.toml`, lock file, tests and CI workflow. New modules depend on `sbml2jax` as a package.

| work package | content | module | status |
| --- | --- | --- | --- |
| 1 | SBML to differentiable JAX code, diffrax simulation, verification against roadrunner and the SBML test suite | [`sbml2jax`](sbml2jax/index.md) | done: apixaban model verified, 998 of 1823 test suite cases pass |
| 2 | hybrid models: three placements of a neural network in the apixaban model (covariates, effects, absorption), training and validation | `hybrid` (planned) | in progress |
| 3 | evaluation of the hybrid models against the mechanistic model and the data | | planned |
| 4 | export of the hybrid models in PEtab SciML | | planned |

## Work package 1: the mechanistic baseline

- [x] Conversion of core SBML, comp models and function definitions into JAX code
- [x] Simulation with diffrax, dosing by segments
- [x] Gradients with respect to every number of the model
- [x] Apixaban model: all outputs match roadrunner, gradients match finite differences
- [x] SBML test suite run with a regression gate in CI
- [ ] Events as diffrax events (root finding and restart), see [Conversion issues](sbml2jax/conversion-issues.md)

## Work package 2: hybrid models

A new module `hybrid/`, which imports `sbml2jax` as a package; the converter stays unchanged. Each variant is trained and validated on its own data, see [Hybrid models](hybrid/index.md).

- [ ] Neural networks on example data (no hybrid model): small networks trained on synthetic data to set up and test the training pipeline
- [ ] Prerequisites: simulation compiled with `jax.jit` and batched with `jax.vmap`; common training setup (experiments, parameter mask, NLL loss, Adam then L-BFGS) and a test that every hybrid model equals the mechanistic model at initialization
- [ ] Variant A, covariate model: a network maps covariates (renal and hepatic function, body weight, age) to scaling factors of the renal and hepatic clearance, evaluated once before the simulation
- [ ] Variant B, effect model: a network maps the plasma concentration to PT, INR, aPTT, mPT and anti-Xa, replacing the Emax and linear laws
- [ ] Variant C, absorption (UDE): a network for the dissolution rate of the tablet in the right-hand side of the ODE system, evaluated in every solver step
- [ ] Optional: purely data-driven baseline, a neural ODE on the observed quantities

## Work package 3: evaluation

- [ ] Comparison of the mechanistic model, the hybrid variants A to C and the data-driven baseline, also outside the training data (left-out patient groups and doses)
- [ ] Interpretation of the learned terms, e.g. the dissolution rate of variant C compared with the saturation law of the mechanistic model

## Work package 4: PEtab SciML export

- [ ] Export of the hybrid models in [PEtab SciML](https://petab-sciml.readthedocs.io): variant A as pre-initialization hybridization, B in the observables, C in the right-hand side of the ODE system
- [ ] Loading and reproducing the exported models in an independent tool

## Principles

- Open-source tools only, public code and models.
- Every result can be regenerated from the repository: committed lock files, generated reports, and examples which write the figures of this documentation.
