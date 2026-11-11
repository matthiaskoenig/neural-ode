# Simulation

[`JaxModel`](../api/model.md#sbml2jax.model.JaxModel) is the generated module as a differentiable object. It is an [equinox](https://github.com/patrick-kidger/equinox) module with two fields:

- `p`: the inputs, a float64 array with the values of `INPUTS`. It is the only array of the model, so `jax.grad` of a function of the model gives the gradient with respect to every input.
- `module`: the generated module, a static field.

```python
from pathlib import Path

from sbml2jax.model import JaxModel, Segment, SolverSettings

model = JaxModel.from_sbml(Path("examples/models/apixaban_body_flat.xml"))
```

## Simulating

`simulate(ts, segments=None, settings=None)` returns the outputs at the times `ts`, an array of shape `(len(ts), len(OUTPUTS))`. The simulation starts at `ts[0]` with the states and constants of `initial(p, ts[0])`.

```python
import jax.numpy as jnp

ts = jnp.linspace(0.0, 24 * 60.0, 241)
ys = model.simulate(ts)
cve = ys[:, model.module.OUTPUTS.index("Cve_api")]
```

## Segments

A timecourse is a sequence of [`Segment`](../api/model.md#sbml2jax.model.Segment)s. A segment runs from the end of the previous one (or from `ts[0]`) to its `end`; its `changes` set states or constants to new values at its start, and the integration restarts from there. This is how doses are given, as in roadrunner and sbmlsim timecourses, without SBML events:

```python
bid = [
    Segment(end=12 * 60.0, changes={"PODOSE_api": 5.0}),
    Segment(end=24 * 60.0, changes={"PODOSE_api": 5.0}),
]
ys = model.simulate(ts, bid)
```

- A change of a state sets it, for example the amount in the oral dosing compartment. A change of a constant sets a parameter from that time on. Any other name raises `KeyError`.
- Initial assignments are not evaluated again after a change, which matches roadrunner when a value is set during a timecourse.
- An output time equal to the end of a segment belongs to the next segment, so it shows the value after the change; the last segment includes its end.
- Without segments there is one segment from `ts[0]` to `ts[-1]` without changes.

## Inputs

`with_inputs` returns a copy with other input values; the model itself is immutable:

```python
heavy = model.with_inputs({"BW": 100.0})
```

`index(names, of)` gives the positions of names in `"INPUTS"`, `"STATES"` or `"CONSTANTS"`.

## Solver settings

[`SolverSettings`](../api/model.md#sbml2jax.model.SolverSettings) chooses the diffrax solver and its tolerances:

| field | default | content |
| --- | --- | --- |
| `rtol`, `atol` | 1e-8, 1e-10 | tolerances of the `PIDController` |
| `max_steps` | 100000 | maximal number of steps per segment |
| `dt0` | `None` | initial step, chosen by diffrax if `None` |
| `solver` | `"Kvaerno5"` | name of a diffrax solver class; Kvaerno5 is implicit, for stiff PBPK models |

```python
ys = model.simulate(ts, bid, SolverSettings(rtol=1e-10, atol=1e-12))
```

The verification against roadrunner uses `rtol=1e-10`, `atol=1e-12`.

## Gradients

```python
import jax

def loss(model: JaxModel) -> jax.Array:
    return jnp.mean(model.simulate(ts, bid)[:, model.module.OUTPUTS.index("Cve_api")])

grad = jax.grad(loss)(model)  # a JaxModel, grad.p holds d loss / d p
```

diffrax differentiates through the solve; the gradient also passes through the initial assignments of the model, which `initial` evaluates from `p`.

!!! warning "Concrete times"
    The assignment of output times to segments happens in Python, so `ts` and the segment ends must be concrete. `jax.jit` and `jax.vmap` around `simulate` are not supported yet; they are planned for work package 2.

## The right-hand side

`JaxModel.rhs(t, y, c)` returns `module.rhs(t, y, c)`, and `simulate` integrates `self.rhs`. A subclass which overrides it, with neural networks as additional fields, is a hybrid model; see [Hybrid models](../hybrid/index.md).
