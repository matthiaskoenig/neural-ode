"""Translation of libsbml formula trees into JAX source code.

`translate` walks a `libsbml.ASTNode` and returns an `Expr`: the source of a
`jax.numpy` expression and the identifiers it depends on. The formula is never
turned into text and parsed again, ids are resolved by the `resolve` callable
of the caller (which knows local parameters and the naming of the generated
module), `rateOf` by the `rate_of` callable.

Conventions of the generated code: `jnp` is `jax.numpy`, `special` is
`jax.scipy.special`, `t` is the time. Piecewise becomes nested `jnp.where`,
relations and logical operators the element-wise functions of `jnp`, so every
formula traces under `jit`, `grad` and `vmap`.
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import libsbml

#: value of the avogadro csymbol of SBML L3
AVOGADRO = 6.02214076e23


class UnsupportedError(ValueError):
    """A construct of the SBML model is not supported by the converter."""

    def __init__(self, feature: str, detail: str = "") -> None:
        """Create the error for `feature` with an optional detail."""
        self.feature = feature
        super().__init__(f"unsupported: {feature}" + (f" ({detail})" if detail else ""))


@dataclass(frozen=True)
class Expr:
    """Source code of an expression and the identifiers it uses."""

    code: str
    deps: frozenset[str] = field(default_factory=frozenset)


def number(value: float) -> Expr:
    """Return the source of a float literal (`jnp.inf`, `jnp.nan` for special values)."""
    if math.isnan(value):
        return Expr("jnp.nan")
    if math.isinf(value):
        return Expr("jnp.inf" if value > 0 else "(-jnp.inf)")
    code = repr(float(value))
    return Expr(f"({code})" if value < 0 else code)


_UNARY: dict[int, str] = {
    libsbml.AST_FUNCTION_ABS: "jnp.abs",
    libsbml.AST_FUNCTION_EXP: "jnp.exp",
    libsbml.AST_FUNCTION_LN: "jnp.log",
    libsbml.AST_FUNCTION_FLOOR: "jnp.floor",
    libsbml.AST_FUNCTION_CEILING: "jnp.ceil",
    libsbml.AST_FUNCTION_SIN: "jnp.sin",
    libsbml.AST_FUNCTION_COS: "jnp.cos",
    libsbml.AST_FUNCTION_TAN: "jnp.tan",
    libsbml.AST_FUNCTION_SINH: "jnp.sinh",
    libsbml.AST_FUNCTION_COSH: "jnp.cosh",
    libsbml.AST_FUNCTION_TANH: "jnp.tanh",
    libsbml.AST_FUNCTION_ARCSIN: "jnp.arcsin",
    libsbml.AST_FUNCTION_ARCCOS: "jnp.arccos",
    libsbml.AST_FUNCTION_ARCTAN: "jnp.arctan",
    libsbml.AST_FUNCTION_ARCSINH: "jnp.arcsinh",
    libsbml.AST_FUNCTION_ARCCOSH: "jnp.arccosh",
    libsbml.AST_FUNCTION_ARCTANH: "jnp.arctanh",
}

#: reciprocal functions: f(x) = 1 / g(x)
_RECIPROCAL: dict[int, str] = {
    libsbml.AST_FUNCTION_SEC: "jnp.cos",
    libsbml.AST_FUNCTION_CSC: "jnp.sin",
    libsbml.AST_FUNCTION_COT: "jnp.tan",
    libsbml.AST_FUNCTION_SECH: "jnp.cosh",
    libsbml.AST_FUNCTION_CSCH: "jnp.sinh",
    libsbml.AST_FUNCTION_COTH: "jnp.tanh",
}

#: inverse reciprocal functions: f(x) = g(1 / x)
_INVERSE_RECIPROCAL: dict[int, str] = {
    libsbml.AST_FUNCTION_ARCSEC: "jnp.arccos",
    libsbml.AST_FUNCTION_ARCCSC: "jnp.arcsin",
    libsbml.AST_FUNCTION_ARCCOT: "jnp.arctan",
    libsbml.AST_FUNCTION_ARCSECH: "jnp.arccosh",
    libsbml.AST_FUNCTION_ARCCSCH: "jnp.arcsinh",
    libsbml.AST_FUNCTION_ARCCOTH: "jnp.arctanh",
}

_RELATIONAL: dict[int, str] = {
    libsbml.AST_RELATIONAL_EQ: "==",
    libsbml.AST_RELATIONAL_NEQ: "!=",
    libsbml.AST_RELATIONAL_LT: "<",
    libsbml.AST_RELATIONAL_LEQ: "<=",
    libsbml.AST_RELATIONAL_GT: ">",
    libsbml.AST_RELATIONAL_GEQ: ">=",
}

_LOGICAL: dict[int, str] = {
    libsbml.AST_LOGICAL_AND: "jnp.logical_and",
    libsbml.AST_LOGICAL_OR: "jnp.logical_or",
    libsbml.AST_LOGICAL_XOR: "jnp.logical_xor",
}


def _fold(function: str, args: list[Expr], empty: Expr) -> Expr:
    """Fold `args` with the binary `function` from the left."""
    if not args:
        return empty
    result = args[0]
    for arg in args[1:]:
        result = Expr(f"{function}({result.code}, {arg.code})", result.deps | arg.deps)
    return result


def _join(args: list[Expr], operator: str, empty: str) -> Expr:
    """Join `args` with an infix `operator`."""
    if not args:
        return Expr(empty)
    deps = frozenset().union(*(a.deps for a in args))
    if len(args) == 1:
        return args[0]
    return Expr("(" + f" {operator} ".join(a.code for a in args) + ")", deps)


def translate(
    node: libsbml.ASTNode,
    resolve: Callable[[str], str],
    rate_of: Callable[[str], Expr] | None = None,
) -> Expr:
    """Translate the formula `node` into a JAX expression.

    Args:
        node: the libsbml formula.
        resolve: maps an SBML id (or local parameter id) to the identifier of
            the generated code.
        rate_of: returns the expression of `rateOf(id)`; None if rateOf is not
            available in the context (initial values).

    Returns:
        The expression and the identifiers it uses.

    Raises:
        UnsupportedError: delay, lambda, unexpanded function calls, or rateOf
            in a context without derivatives.
    """

    def tr(n: libsbml.ASTNode) -> Expr:
        return translate(n, resolve, rate_of)

    t = node.getType()
    children = [node.getChild(k) for k in range(node.getNumChildren())]

    # numbers and constants
    if t == libsbml.AST_INTEGER:
        return number(float(node.getInteger()))
    if t in (libsbml.AST_REAL, libsbml.AST_REAL_E, libsbml.AST_RATIONAL):
        return number(node.getValue())
    if t == libsbml.AST_CONSTANT_PI:
        return Expr("jnp.pi")
    if t == libsbml.AST_CONSTANT_E:
        return Expr("jnp.e")
    if t == libsbml.AST_CONSTANT_TRUE:
        return Expr("True")
    if t == libsbml.AST_CONSTANT_FALSE:
        return Expr("False")
    if t == libsbml.AST_NAME_AVOGADRO:
        return number(AVOGADRO)
    if t == libsbml.AST_NAME_TIME:
        return Expr("t", frozenset({"t"}))
    if t == libsbml.AST_NAME:
        name = resolve(node.getName())
        return Expr(name, frozenset({name}))

    # arithmetic
    if t == libsbml.AST_PLUS:
        return _join([tr(c) for c in children], "+", "0.0")
    if t == libsbml.AST_TIMES:
        return _join([tr(c) for c in children], "*", "1.0")
    if t == libsbml.AST_MINUS:
        args = [tr(c) for c in children]
        if len(args) == 1:
            return Expr(f"(-{args[0].code})", args[0].deps)
        return Expr(f"({args[0].code} - {args[1].code})", args[0].deps | args[1].deps)
    if t == libsbml.AST_DIVIDE:
        a, b = (tr(c) for c in children)
        return Expr(f"({a.code} / {b.code})", a.deps | b.deps)
    if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
        a, b = (tr(c) for c in children)
        return Expr(f"jnp.power({a.code}, {b.code})", a.deps | b.deps)
    if t == libsbml.AST_FUNCTION_ROOT:
        if len(children) == 1:
            a = tr(children[0])
            return Expr(f"jnp.sqrt({a.code})", a.deps)
        n, a = (tr(c) for c in children)
        return Expr(f"jnp.power({a.code}, 1.0 / {n.code})", a.deps | n.deps)
    if t == libsbml.AST_FUNCTION_LOG:
        if len(children) == 1:
            a = tr(children[0])
            return Expr(f"jnp.log10({a.code})", a.deps)
        b, a = (tr(c) for c in children)
        return Expr(f"(jnp.log({a.code}) / jnp.log({b.code}))", a.deps | b.deps)
    if t == libsbml.AST_FUNCTION_FACTORIAL:
        a = tr(children[0])
        return Expr(f"jnp.exp(special.gammaln({a.code} + 1.0))", a.deps)
    if t in _UNARY:
        a = tr(children[0])
        return Expr(f"{_UNARY[t]}({a.code})", a.deps)
    if t in _RECIPROCAL:
        a = tr(children[0])
        return Expr(f"(1.0 / {_RECIPROCAL[t]}({a.code}))", a.deps)
    if t in _INVERSE_RECIPROCAL:
        a = tr(children[0])
        return Expr(f"{_INVERSE_RECIPROCAL[t]}(1.0 / {a.code})", a.deps)
    if t == libsbml.AST_FUNCTION_MIN:
        return _fold("jnp.minimum", [tr(c) for c in children], Expr("jnp.nan"))
    if t == libsbml.AST_FUNCTION_MAX:
        return _fold("jnp.maximum", [tr(c) for c in children], Expr("jnp.nan"))
    if t == libsbml.AST_FUNCTION_REM:
        a, b = (tr(c) for c in children)
        return Expr(f"jnp.fmod({a.code}, {b.code})", a.deps | b.deps)
    if t == libsbml.AST_FUNCTION_QUOTIENT:
        a, b = (tr(c) for c in children)
        return Expr(f"jnp.trunc({a.code} / {b.code})", a.deps | b.deps)

    # relations, chained for more than two arguments (a < b < c)
    if t in _RELATIONAL:
        args = [tr(c) for c in children]
        pairs = [
            Expr(f"({a.code} {_RELATIONAL[t]} {b.code})", a.deps | b.deps)
            for a, b in itertools.pairwise(args)
        ]
        return _fold("jnp.logical_and", pairs, Expr("True"))

    # logic
    if t in _LOGICAL:
        empty = Expr("False") if t != libsbml.AST_LOGICAL_AND else Expr("True")
        return _fold(_LOGICAL[t], [tr(c) for c in children], empty)
    if t == libsbml.AST_LOGICAL_NOT:
        a = tr(children[0])
        return Expr(f"jnp.logical_not({a.code})", a.deps)
    if t == libsbml.AST_LOGICAL_IMPLIES:
        a, b = (tr(c) for c in children)
        return Expr(f"jnp.logical_or(jnp.logical_not({a.code}), {b.code})", a.deps | b.deps)

    # piecewise(value1, condition1, value2, condition2, ..., otherwise)
    if t == libsbml.AST_FUNCTION_PIECEWISE:
        args = [tr(c) for c in children]
        result = args[-1] if len(args) % 2 == 1 else Expr("jnp.nan")
        pieces = args[: len(args) - len(args) % 2]
        for k in range(len(pieces) - 2, -1, -2):
            value, condition = pieces[k], pieces[k + 1]
            result = Expr(
                f"jnp.where({condition.code}, {value.code}, {result.code})",
                value.deps | condition.deps | result.deps,
            )
        return result

    if t == libsbml.AST_FUNCTION_RATE_OF:
        if rate_of is None:
            raise UnsupportedError("rateOf", "in an initial value")
        return rate_of(children[0].getName())
    if t == libsbml.AST_FUNCTION_DELAY:
        raise UnsupportedError("delay")
    if t == libsbml.AST_LAMBDA:
        raise UnsupportedError("lambda")
    if t == libsbml.AST_FUNCTION:
        raise UnsupportedError("function call", node.getName())
    raise UnsupportedError("formula node", f"{node.getName() or t}")
