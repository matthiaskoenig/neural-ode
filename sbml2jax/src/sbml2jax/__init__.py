"""sbml2jax: differentiable JAX code for the ODE system of SBML models."""

import jax

# PK/PD models span many orders of magnitude, the comparison with roadrunner
# needs double precision
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
