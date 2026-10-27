"""The generated model as a differentiable object and its simulation.

`JaxModel` is an equinox module: the inputs `p` are its only array field, so
`jax.grad` of a loss of `model.simulate(...)` gives the gradient with respect
to every input. The functions of the generated module are static. Hybrid
models (work package 2) subclass `JaxModel` and add networks as fields.

`simulate` solves the ODE system with diffrax. Dosing and other changes are
`Segment`s: the integration restarts at the start of each segment with the
changes applied to states or constants, as roadrunner/sbmlsim timecourses do.
"""

import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from sbml2jax.convert import convert_sbml2jax, load_module


@dataclass(frozen=True)
class SolverSettings:
    """Settings of the diffrax solve."""

    rtol: float = 1e-8
    atol: float = 1e-10
    max_steps: int = 100_000
    dt0: float | None = None
    solver: str = "Kvaerno5"
    """a diffrax solver class name; Kvaerno5 is implicit (stiff PBPK models)"""


@dataclass(frozen=True)
class Segment:
    """Part of a timecourse, from the end of the previous segment to `end`."""

    end: float
    changes: Mapping[str, float] = field(default_factory=dict)
    """state or constant name -> value set at the start of the segment"""


class JaxModel(eqx.Module):
    """A generated model with its inputs as trainable array."""

    p: jax.Array
    module: types.ModuleType = eqx.field(static=True)

    @classmethod
    def from_sbml(cls, sbml_path: Path, module_path: Path | None = None) -> "JaxModel":
        """Convert `sbml_path` and return the model with the SBML input values."""
        source = convert_sbml2jax(sbml_path, module_path)
        name = f"sbml2jax_{sbml_path.stem.replace('-', '_').replace('.', '_')}"
        if module_path is not None:
            module = load_module(path=module_path, name=name)
        else:
            module = load_module(source=source, name=name)
        return cls.from_module(module)

    @classmethod
    def from_module(cls, module: types.ModuleType) -> "JaxModel":
        """Return the model of a generated module with its input values."""
        return cls(p=jnp.asarray(module.INPUT_VALUES, dtype=jnp.float64), module=module)

    def index(self, names: Sequence[str], of: str) -> jax.Array:
        """Return the indices of `names` in `INPUTS`, `STATES` or `CONSTANTS`."""
        vector: tuple[str, ...] = getattr(self.module, of)
        return jnp.asarray([vector.index(n) for n in names], dtype=int)

    def with_inputs(self, values: Mapping[str, float]) -> "JaxModel":
        """Return a copy with the inputs `values` changed."""
        p = self.p
        for name, value in values.items():
            p = p.at[self.module.INPUTS.index(name)].set(value)
        return eqx.tree_at(lambda m: m.p, self, p)

    def rhs(self, t: jax.Array, y: jax.Array, c: jax.Array) -> jax.Array:
        """Return the derivatives; subclasses add network terms here."""
        return self.module.rhs(t, y, c)

    def simulate(
        self,
        ts: jax.Array,
        segments: Sequence[Segment] | None = None,
        settings: SolverSettings | None = None,
    ) -> jax.Array:
        """Simulate and return the outputs at the times `ts`.

        Args:
            ts: increasing output times; the simulation starts at `ts[0]`.
            segments: changes during the timecourse; default one segment to
                `ts[-1]` without changes.
            settings: solver settings, `SolverSettings()` by default.

        Returns:
            Array of shape `(len(ts), len(OUTPUTS))`.
        """
        settings = settings or SolverSettings()
        ts = jnp.asarray(ts, dtype=jnp.float64)
        t0 = float(ts[0])
        y, c = self.module.initial(self.p, t0)
        segments = list(segments) if segments else [Segment(end=float(ts[-1]))]
        solver = getattr(diffrax, settings.solver)()
        controller = diffrax.PIDController(rtol=settings.rtol, atol=settings.atol)
        term = diffrax.ODETerm(lambda t, y, c: self.rhs(t, y, c))
        states, constants = self.module.STATES, self.module.CONSTANTS

        values = []
        start = t0
        for k, segment in enumerate(segments):
            for name, value in segment.changes.items():
                if name in states:
                    y = y.at[states.index(name)].set(value)
                elif name in constants:
                    c = c.at[constants.index(name)].set(value)
                else:
                    raise KeyError(f"'{name}' is neither a state nor a constant")
            last = k == len(segments) - 1
            # output times of this segment; the end of a segment belongs to the next
            mask = (ts >= start) & ((ts <= segment.end) if last else (ts < segment.end))
            seg_ts = ts[mask]
            save = jnp.concatenate([seg_ts, jnp.asarray([segment.end])])
            if segment.end > start:
                solution = diffrax.diffeqsolve(
                    term,
                    solver,
                    t0=start,
                    t1=segment.end,
                    dt0=settings.dt0,
                    y0=y,
                    args=c,
                    stepsize_controller=controller,
                    saveat=diffrax.SaveAt(ts=save),
                    max_steps=settings.max_steps,
                )
                ys = solution.ys
                assert ys is not None
            else:
                ys = jnp.broadcast_to(y, (save.shape[0], y.shape[0]))
            if seg_ts.shape[0]:
                values.append(jax.vmap(self.module.outputs, in_axes=(0, 0, None))(seg_ts, ys[:-1], c))
            y = ys[-1]
            start = segment.end
        return jnp.concatenate(values) if values else jnp.zeros((0, len(self.module.OUTPUTS)))
