# sbml2jax

`sbml2jax` converts the ODE system of an [SBML](https://sbml.org) model into a
readable, differentiable [JAX](https://github.com/jax-ml/jax) module and
simulates it with [diffrax](https://github.com/patrick-kidger/diffrax). The
model is an [equinox](https://github.com/patrick-kidger/equinox) module, so
`jax.grad` gives gradients with respect to every number of the SBML model, and
neural networks can be added to its right-hand side (hybrid models,
universal differential equations).

It is work package 1 of the M.Sc. thesis *Hybrid neural-network ODE models for
PK/PD* (supervisor Matthias König), with the structure of
[sbml2cellml](https://github.com/matthiaskoenig/sbml2cellml).
[libRoadRunner](https://github.com/sys-bio/roadrunner) is the reference
simulator.

## Status

- **Apixaban PBPK/PD model** (Myshkina et al. 2026, flattened whole-body
  model, 51 states): all 257 outputs match roadrunner over a 24 h oral dose
  (relative 1e-5, absolute 1e-9), and the gradient of the plasma
  concentration agrees with finite differences
  (`tests/test_apixaban.py`, `examples/apixaban_example.py`).
- **SBML test suite**: cases 1-10 are committed as the baseline
  (`docs/testsuite.md`). In a first run over about 740 cases, 478 passed, and
  every failure was an event or algebraic rule case, see
  [conversion issues](docs/conversion-issues.md).

## Installation

```bash
uv sync --extra dev           # development, from uv.lock
pip install "sbml2jax[testsuite] @ git+https://github.com/tom-del792/smbl2jaxtest.git"   # e.g. in Colab
```

## Usage

```python
from pathlib import Path

import jax
import jax.numpy as jnp

from sbml2jax.model import JaxModel, Segment

model = JaxModel.from_sbml(Path("examples/models/apixaban_body_flat.xml"))
ts = jnp.linspace(0.0, 24 * 60.0, 241)                      # min
dose = [Segment(end=24 * 60.0, changes={"PODOSE_api": 10.0})]  # 10 mg oral
ys = model.simulate(ts, dose)                                # (time, OUTPUTS)
cve = ys[:, model.module.OUTPUTS.index("Cve_api")]

# gradient of the mean plasma concentration w.r.t. all inputs of the model
grad = jax.grad(lambda m: m.simulate(ts, dose)[:, model.module.OUTPUTS.index("Cve_api")].mean())(model)
```

The generated module can also be written and read as a file:

```bash
sbml2jax examples/models/apixaban_body_flat.xml -o apixaban.py
```

It contains `INPUTS`, `STATES`, `CONSTANTS`, `OUTPUTS` and the functions
`initial(p, t)`, `rhs(t, y, c)` and `outputs(t, y, c)` as plain `jax.numpy`
code.

## Verification

```bash
uv run pytest                                # unit tests and the apixaban check
uv run sbml2jax-testsuite run --first 10     # SBML test suite cases (all: ~40 min on 2 CPUs)
uv run python examples/apixaban_example.py
```

Every case of the [SBML test suite](https://github.com/sbmlteam/sbml-test-suite)
is simulated with roadrunner, converted, simulated with diffrax, and both
simulations are compared with the expected results. The results are committed
(`testsuite/results.json`), and `tests/test_testsuite_full.py` fails on
regressions (`SBML2JAX_TESTSUITE=1`).

## License

MIT. `src/sbml2jax/testsuite/cases.py` and `compare.py` are adapted from
sbml2cellml (MIT, Matthias König). The apixaban models in `examples/models/`
are from [apixaban-model](https://github.com/matthiaskoenig/apixaban-model)
(Myshkina et al. 2026, doi:10.5281/zenodo.18877904), licensed CC BY-SA 4.0.
