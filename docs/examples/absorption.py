"""JAX model of `absorption` generated from `absorption.xml` by sbml2jax 0.1.0.

Do not edit, regenerate the module from the SBML model.
"""

import jax.numpy as jnp
from jax.scipy import special  # noqa: F401

MODEL_ID = 'absorption'

INPUTS = (
    'Vplasma',
    'ka',
    'CL',
    'Emax',
    'EC50',
    'Agut',
    'C',
)

INPUT_VALUES = (
    5.0,
    0.02,
    0.1,
    1.0,
    0.5,
    10.0,
    0.0,
)

STATES = (
    'Agut',
    'C_amount',
)

CONSTANTS = (
    'Vplasma',
    'ka',
    'CL',
    'Emax',
    'EC50',
)

OUTPUTS = (
    'Vplasma',
    'ka',
    'CL',
    'Emax',
    'EC50',
    'E',
    'absorption_rate',
    'elimination',
    'Agut',
    'C',
)

SPECIES = {
    'Agut': ('Vplasma', True),
    'C': ('Vplasma', False),
}


def initial(p, t=0.0):
    """Return the states `y0` and the constants `c` at the start time `t`."""
    Vplasma = p[0]
    ka = p[1]
    CL = p[2]
    Emax = p[3]
    EC50 = p[4]
    Agut = p[5]
    C = p[6]
    C_amount = (C * Vplasma)
    return jnp.array([Agut, C_amount]), jnp.array([Vplasma, ka, CL, Emax, EC50])


def rhs(t, y, c):
    """Return the derivatives of the states `y` at time `t`."""
    Agut = y[0]
    C_amount = y[1]
    Vplasma = c[0]
    ka = c[1]
    CL = c[2]
    absorption_rate = (ka * Agut)
    d_Agut = (-absorption_rate)
    C = (C_amount / Vplasma)
    elimination = (CL * C)
    d_C_amount = (absorption_rate - elimination)
    return jnp.array([d_Agut, d_C_amount])


def outputs(t, y, c):
    """Return the values of `OUTPUTS` at time `t`."""
    Agut = y[0]
    C_amount = y[1]
    Vplasma = c[0]
    ka = c[1]
    CL = c[2]
    Emax = c[3]
    EC50 = c[4]
    C = (C_amount / Vplasma)
    E = ((Emax * C) / (C + EC50))
    absorption_rate = (ka * Agut)
    elimination = (CL * C)
    return jnp.array([Vplasma, ka, CL, Emax, EC50, E, absorption_rate, elimination, Agut, C])
