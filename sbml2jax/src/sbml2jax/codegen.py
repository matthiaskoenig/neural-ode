"""Generation of the Python module of a model.

`generate` writes a `ModelSystem` as the source of a module with

- `INPUTS`, `INPUT_VALUES`: names and SBML values of the inputs `p`,
- `STATES`, `CONSTANTS`, `OUTPUTS`: names of `y`, `c` and the outputs,
- `SPECIES`: compartment and `hasOnlySubstanceUnits` of every species,
- `initial(p, t=0.0) -> (y0, c)`: the state and the constants at the start,
- `rhs(t, y, c) -> dy/dt`: the right-hand side for diffrax,
- `outputs(t, y, c)`: the SBML values of all outputs at one time point.

Every function evaluates only the definitions it needs, in an order in which
each definition comes after the ones it uses (topological order). The module
is plain Python with `jax.numpy`: readable, versioned with the model and
traceable with `jit`, `grad` and `vmap`.
"""

import math
from collections.abc import Iterable

from sbml2jax import __version__
from sbml2jax.analysis import ModelSystem
from sbml2jax.jaxmath import Expr


class CodegenError(ValueError):
    """The system cannot be written, e.g. a cyclic or missing definition."""


def order(needed: Iterable[str], definitions: dict[str, Expr], known: set[str]) -> list[str]:
    """Return the definitions `needed` requires, each after its dependencies.

    Args:
        needed: identifiers whose values are required.
        definitions: identifier -> expression.
        known: identifiers which have values already (arguments).

    Returns:
        The identifiers to define, in topological order.

    Raises:
        CodegenError: a dependency cycle (e.g. assignment rules which use each
            other) or an identifier without definition.
    """
    result: list[str] = []
    done: set[str] = set(known)
    active: list[str] = []

    def visit(name: str) -> None:
        if name in done:
            return
        if name in active:
            cycle = " -> ".join([*active[active.index(name) :], name])
            raise CodegenError(f"cyclic definitions: {cycle}")
        if name not in definitions:
            raise CodegenError(f"'{name}' has no definition")
        active.append(name)
        for dep in sorted(definitions[name].deps):
            visit(dep)
        active.pop()
        done.add(name)
        result.append(name)

    for name in needed:
        visit(name)
    return result


def _float(value: float) -> str:
    """Return the source of a float, `float("nan")` and `float("inf")` for special values."""
    if math.isfinite(value):
        return repr(float(value))
    return f'float("{value}")'


def _tuple(values: Iterable[str]) -> str:
    """Return the source of a tuple of strings."""
    items = [repr(v) for v in values]
    if len(items) == 1:
        return f"({items[0]},)"
    return "(\n" + "".join(f"    {item},\n" for item in items) + ")"


def _array(names: list[str], indent: str = "    ") -> str:
    """Return the source of `jnp.array` of the identifiers `names`."""
    return f"{indent}return jnp.array([" + ", ".join(names) + "])\n"


def _function(
    signature: str,
    doc: str,
    unpack: list[tuple[str, str]],
    lines: list[tuple[str, Expr]],
    result: str,
) -> str:
    """Return the source of a function."""
    out = [f"def {signature}:\n", f'    """{doc}"""\n']
    out += [f"    {name} = {vector}\n" for name, vector in unpack]
    out += [f"    {name} = {expr.code}\n" for name, expr in lines]
    out.append(result)
    return "".join(out)


def generate(system: ModelSystem, source: str = "") -> str:
    """Return the source of the Python module of `system`.

    Args:
        system: the analysed model.
        source: name of the SBML file, for the docstring.

    Returns:
        The module source.
    """
    inputs = list(system.inputs)
    output_ids = list(system.outputs)
    output_names = [system.outputs[sid] for sid in output_ids]

    # initial(p, t): inputs -> states and constants at the start
    known_initial = set(inputs) | {"t"}
    initial_lines = order([*system.states, *system.constants], system.initial, known_initial)
    initial_fn = _function(
        "initial(p, t=0.0)",
        "Return the states `y0` and the constants `c` at the start time `t`.",
        [(name, f"p[{k}]") for k, name in enumerate(inputs)],
        [(name, system.initial[name]) for name in initial_lines],
        "    return "
        + "jnp.array(["
        + ", ".join(system.states)
        + "]), jnp.array(["
        + ", ".join(system.constants)
        + "])\n",
    )

    # rhs(t, y, c) and outputs(t, y, c): states and constants -> computed
    known = set(system.states) | set(system.constants) | {"t"}
    unpack = [(name, f"y[{k}]") for k, name in enumerate(system.states)]
    unpack += [(name, f"c[{k}]") for k, name in enumerate(system.constants)]
    derivative_names = [system.derivatives[s] for s in system.states]

    rhs_lines = order(derivative_names, system.computed, known)
    rhs_fn = _function(
        "rhs(t, y, c)",
        "Return the derivatives of the states `y` at time `t`.",
        _used(unpack, rhs_lines, system, derivative_names),
        [(name, system.computed[name]) for name in rhs_lines],
        _array(derivative_names),
    )
    output_lines = order(output_names, system.computed, known)
    outputs_fn = _function(
        "outputs(t, y, c)",
        "Return the values of `OUTPUTS` at time `t`.",
        _used(unpack, output_lines, system, output_names),
        [(name, system.computed[name]) for name in output_lines],
        _array(output_names),
    )

    species = "".join(
        f"    {info.sid!r}: ({info.compartment!r}, {info.only_substance_units!r}),\n"
        for info in system.species.values()
    )
    values = "".join(f"    {_float(value)},\n" for value in system.inputs.values())
    header = (
        f'"""JAX model of `{system.model_id}`'
        + (f" generated from `{source}`" if source else "")
        + f" by sbml2jax {__version__}.\n\nDo not edit, regenerate the module from the SBML model.\n"
        + '"""\n\n'
        + "import jax.numpy as jnp\n"
        + "from jax.scipy import special  # noqa: F401\n\n"
        + f"MODEL_ID = {system.model_id!r}\n\n"
        + f"INPUTS = {_tuple(inputs) if inputs else '()'}\n\n"
        + f"INPUT_VALUES = (\n{values})\n\n"
        + f"STATES = {_tuple(system.states)}\n\n"
        + f"CONSTANTS = {_tuple(system.constants) if system.constants else '()'}\n\n"
        + f"OUTPUTS = {_tuple(output_ids) if output_ids else '()'}\n\n"
        + f"SPECIES = {{\n{species}}}\n\n\n"
    )
    module = header + initial_fn + "\n\n" + rhs_fn + "\n\n" + outputs_fn
    return module.replace("jnp.array([])", "jnp.zeros(0)")


def _used(
    unpack: list[tuple[str, str]], lines: list[str], system: ModelSystem, results: list[str]
) -> list[tuple[str, str]]:
    """Return the unpacked arguments which the function body uses."""
    used = set(results)
    for name in lines:
        used |= system.computed[name].deps
    return [(name, vector) for name, vector in unpack if name in used]
