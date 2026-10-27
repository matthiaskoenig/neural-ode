"""Python identifiers for the quantities of a model.

SBML ids are valid Python identifiers, so the generated code uses them as they
are. Only ids which are Python keywords or names of the generated code (the
arguments `t`, `y`, `c`, `p` and the modules `jnp`, `special`) get a trailing
underscore. Derived quantities (the amount of a concentration species, the
derivative of a state) get names built from the id, with a numeric suffix if
the name is taken.
"""

import keyword

#: names used by the generated code itself
RESERVED: frozenset[str] = frozenset({"t", "y", "c", "p", "jnp", "jax", "special"})


class Names:
    """Registry of the identifiers of one generated module."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._by_key: dict[str, str] = {}
        self._taken: set[str] = set(RESERVED)

    def _free(self, base: str) -> str:
        """Return `base`, or `base` with a suffix, which is not taken yet."""
        if keyword.iskeyword(base) or keyword.issoftkeyword(base):
            base = f"{base}_"
        name = base
        n = 1
        while name in self._taken:
            if base in RESERVED and n == 1:
                name = f"{base}_"
            else:
                name = f"{base}_{n}"
            n += 1
        self._taken.add(name)
        return name

    def get(self, key: str, base: str | None = None) -> str:
        """Return the identifier of `key`, created from `base` (default `key`)."""
        if key not in self._by_key:
            self._by_key[key] = self._free(base if base is not None else key)
        return self._by_key[key]

    def __contains__(self, key: str) -> bool:
        """Return whether `key` has an identifier."""
        return key in self._by_key
