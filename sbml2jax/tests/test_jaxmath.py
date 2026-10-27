import math

import jax.numpy as jnp
import libsbml
import pytest
from jax.scipy import special

from sbml2jax.jaxmath import UnsupportedError, translate

VALUES = {"x": 2.0, "y": 3.0, "z": -0.5}


def evaluate(formula: str, t: float = 1.5) -> float:
    ast = libsbml.parseL3Formula(formula)
    assert ast is not None, libsbml.getLastParseL3Error()
    expr = translate(ast, lambda sid: sid)
    return float(eval(expr.code, {"jnp": jnp, "special": special, "t": t, **VALUES}))


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("x + y * 2", 8.0),
        ("-x", -2.0),
        ("x - y - 1", -2.0),
        ("x / y", 2 / 3),
        ("x^y", 8.0),
        ("pow(x, y)", 8.0),
        ("sqrt(4)", 2.0),
        ("root(3, 8)", 2.0),
        ("log(100)", 2.0),
        ("log(2, 8)", 3.0),
        ("ln(exp(x))", 2.0),
        ("abs(z)", 0.5),
        ("floor(z)", -1.0),
        ("ceil(z)", 0.0),
        ("factorial(4)", 24.0),
        ("sec(z)", 1 / math.cos(-0.5)),
        ("arccot(x)", math.atan(0.5)),
        ("min(x, y, z)", -0.5),
        ("max(x, y, z)", 3.0),
        ("rem(7, 3)", 1.0),
        ("quotient(-7, 2)", -3.0),
        ("piecewise(1, x > y, 2, x < y, 3)", 2.0),
        ("piecewise(1, x > y, 3)", 3.0),
        ("piecewise(1, x > y, 2, true)", 2.0),
        ("piecewise(10, (x < y) && (z < 0), 20)", 10.0),
        ("piecewise(10, xor(true, true), 20)", 20.0),
        ("piecewise(10, z < x < y, 20)", 10.0),
        ("piecewise(10, implies(false, false), 20)", 10.0),
        ("time * 2", 3.0),
        ("pi", math.pi),
        ("exponentiale", math.e),
        ("avogadro", 6.02214076e23),
        ("INF", math.inf),
    ],
)
def test_translate(formula: str, expected: float) -> None:
    assert evaluate(formula) == pytest.approx(expected)


def test_dependencies() -> None:
    expr = translate(libsbml.parseL3Formula("k1 * S1 / (Km + S1) + time"), lambda sid: sid)
    assert expr.deps == {"k1", "S1", "Km", "t"}


def test_resolve_local() -> None:
    expr = translate(libsbml.parseL3Formula("k * S"), lambda sid: {"k": "R1_k"}.get(sid, sid))
    assert expr.deps == {"R1_k", "S"}


def test_delay_unsupported() -> None:
    with pytest.raises(UnsupportedError, match="delay"):
        translate(libsbml.parseL3Formula("delay(x, 1)"), lambda sid: sid)


def test_rate_of_without_context() -> None:
    with pytest.raises(UnsupportedError, match="rateOf"):
        translate(libsbml.parseL3Formula("rateOf(x)"), lambda sid: sid)
