import pytest

from sbml2jax.codegen import CodegenError, order
from sbml2jax.jaxmath import Expr


def e(*deps: str) -> Expr:
    return Expr("+".join(deps) or "0.0", frozenset(deps))


def test_order_topological() -> None:
    definitions = {"a": e("b", "c"), "b": e("c"), "c": e("x"), "unused": e("x")}
    assert order(["a"], definitions, {"x"}) == ["c", "b", "a"]


def test_order_cycle() -> None:
    with pytest.raises(CodegenError, match="cyclic"):
        order(["a"], {"a": e("b"), "b": e("a")}, set())


def test_order_missing() -> None:
    with pytest.raises(CodegenError, match="no definition"):
        order(["a"], {"a": e("b")}, set())
