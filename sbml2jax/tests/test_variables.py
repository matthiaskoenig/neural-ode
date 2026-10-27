from sbml2jax.variables import Names


def test_ids_kept() -> None:
    names = Names()
    assert names.get("S1") == "S1"
    assert names.get("S1") == "S1"


def test_reserved_and_keywords() -> None:
    names = Names()
    assert names.get("t") == "t_"
    assert names.get("lambda") == "lambda_"
    assert names.get("jnp") == "jnp_"


def test_derived_names_unique() -> None:
    names = Names()
    names.get("S1_amount")
    assert names.get("S1#amount", "S1_amount") == "S1_amount_1"
