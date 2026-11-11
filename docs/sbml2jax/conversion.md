# Conversion

The conversion turns an SBML model into the three functions of an ODE system: the values at the start, the right-hand side and the outputs. This page describes how the SBML constructs map to them; [`analysis`](../api/analysis.md) and [`jaxmath`](../api/jaxmath.md) implement it.

## Reading

[`sbml.read_document`](../api/sbml.md) reads the file with libsbml and checks it for errors. Hierarchical models of the comp package are flattened (the apixaban whole-body model consists of submodels for liver, kidney, intestine and coagulation; their ids get the prefix of the submodel, for example `LI__API2M_k`). Calls of function definitions are expanded. Afterwards the converter only sees core SBML.

## Roles of identifiers

Every identifier of the model has exactly one role during a simulation:

| role | what | in the generated code |
| --- | --- | --- |
| state | changed by reactions or a rate rule | an entry of `y`, integrated by the solver |
| constant | does not change during the simulation | an entry of `c`, fixed at the start |
| computed | assignment rules, concentrations, reaction rates, derivatives | a local assignment in a function |

The values at the start are definitions of their own: an initial assignment, else the assignment rule, else the value of the attribute. The attribute values are the **inputs** `p`.

## Species

SBML keeps the amount of a species when the size of its compartment changes. A species which no rule determines therefore has an amount `<id>_amount` (the id itself if the species has `hasOnlySubstanceUnits`), which is a state when reactions change it and a constant otherwise. Its concentration is computed:

$$
[S] = \frac{A_S}{V}
$$

The change of the amount is the sum of the reaction terms, with the stoichiometry negative for reactants, times the conversion factor of the species, if any; boundary species are excluded:

$$
\frac{\mathrm{d} A_S}{\mathrm{d} t} = f_S \sum_{r} n_{S,r} \, v_r
$$

A species with a rate rule integrates its value as it is.

## Formulas

[`jaxmath.translate`](../api/jaxmath.md) walks the libsbml formula tree (`ASTNode`) and writes the source of a `jax.numpy` expression, together with the identifiers it uses. Formulas are never printed and parsed again. Some translations:

| MathML | JAX |
| --- | --- |
| `piecewise` | nested `jnp.where` (without `otherwise`: NaN) |
| relations with more than two arguments | chained with `jnp.logical_and` |
| `log` with one argument | `jnp.log10` |
| `factorial` | `jnp.exp(special.gammaln(x + 1))` |
| `rem`, `quotient` | `jnp.fmod`, `jnp.trunc` of the division |
| `avogadro` | 6.02214179e23, the value of the SBML Level 3 specification |
| `time` | the argument `t` |

Local parameters of a kinetic law become `<reaction>_<id>`. Python keywords and the reserved names `t`, `y`, `c`, `p`, `jnp`, `jax` and `special` get a trailing underscore ([`variables`](../api/variables.md)).

## rateOf

`rateOf(x)` is resolved to the derivative of a state. For a species in concentration with a changing compartment it is

$$
\frac{\mathrm{d} [S]}{\mathrm{d} t} = \frac{\dot A_S - [S] \, \dot V}{V}
$$

and 0 for a constant.

## Not supported

Events, algebraic rules, delays, fast reactions, rateOf of rule targets and in initial values, and rate rules of species references raise `UnsupportedError`. Dosing does not need events: doses are [segments](simulation.md#segments). See [Conversion issues](conversion-issues.md).
