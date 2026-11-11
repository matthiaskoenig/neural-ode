# Generated module

The result of the conversion is a Python module: plain code with `jax.numpy`, one assignment per quantity, named by its SBML id. It is a fixed contract between the converter, [`JaxModel`](simulation.md) and the test suite runner, and can be versioned next to the model.

## An example

A one-compartment model with first-order absorption from a gut depot, linear elimination and an Emax effect ([`absorption.xml`](../examples/absorption.xml)):

| SBML | content |
| --- | --- |
| compartment | `Vplasma` = 5 |
| species | `Agut` (amount, initial 10), `C` (concentration, initial 0) |
| reactions | `absorption_rate`: `Agut` → `C`, `ka * Agut`; `elimination`: `C` →, `CL * C` |
| assignment rule | `E = Emax * C / (C + EC50)` |

`sbml2jax absorption.xml -o absorption.py` writes:

```python
--8<-- "docs/examples/absorption.py"
```

The module is regenerated and compared with this file when the documentation is built, so the page always shows the output of the current converter.

## Contents

| name | content |
| --- | --- |
| `MODEL_ID` | the id of the SBML model |
| `INPUTS`, `INPUT_VALUES` | names and SBML values of the inputs `p`: every number of the model |
| `STATES` | names of the states `y`, in the order of the vector |
| `CONSTANTS` | names of the constants `c` |
| `OUTPUTS` | the SBML ids whose values `outputs` returns: compartments, parameters, reactions, species |
| `SPECIES` | compartment and `hasOnlySubstanceUnits` of every species |
| `initial(p, t=0.0)` | the states `y0` and the constants `c` from the inputs |
| `rhs(t, y, c)` | the derivatives of the states, the right-hand side for diffrax |
| `outputs(t, y, c)` | the values of `OUTPUTS` at one time point |

Some things to note in the example:

- The species `C` is in concentration, so its state is the amount `C_amount`, set to `C * Vplasma` in `initial`; `rhs` and `outputs` compute the concentration `C = C_amount / Vplasma`. `Agut` has only substance units, so it is its own state.
- The assignment rule `E` is not needed by any derivative, so `rhs` does not compute it; `outputs` does. Every function contains only the definitions it needs, each after the ones it uses (topological order, [`codegen.order`](../api/codegen.md)). A cyclic definition raises `CodegenError`.
- The initial concentration of `C` is an input, so a gradient with respect to it is possible; it is not a constant, because it only defines the start.

## Loading a module

`JaxModel.from_sbml(path, module_path=...)` writes the module to a file and imports it from there; without `module_path` the module is created in memory. An existing file is loaded with [`convert.load_module`](../api/convert.md):

```python
from pathlib import Path

from sbml2jax.convert import load_module
from sbml2jax.model import JaxModel

model = JaxModel.from_module(load_module(path=Path("absorption.py")))
```

The functions are pure and can be used without `JaxModel`, for example with diffrax directly:

```python
import diffrax
import jax.numpy as jnp

module = load_module(path=Path("absorption.py"))
y0, c = module.initial(jnp.asarray(module.INPUT_VALUES))
solution = diffrax.diffeqsolve(
    diffrax.ODETerm(module.rhs), diffrax.Kvaerno5(), t0=0.0, t1=100.0, dt0=None, y0=y0, args=c,
    stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
    saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, 100.0, 11)),
)
```
