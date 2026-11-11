"""A first simulation: apixaban 5 mg twice daily, plasma concentration and INR.

The example of the documentation (`docs/index.md` includes the part between the
`--8<--` markers). Run it from `sbml2jax/`; it writes the figure of the
documentation:

    uv run python examples/first_simulation.py ../docs/images/first-simulation.png
"""

import sys

import matplotlib

matplotlib.use("Agg")

# --8<-- [start:example]
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from sbml2jax.model import JaxModel, Segment

model = JaxModel.from_sbml(Path("examples/models/apixaban_body_flat.xml"))
outputs = model.module.OUTPUTS

# 5 mg apixaban orally every 12 hours; the model time is in minutes
ts = jnp.linspace(0.0, 48 * 60.0, 481)
doses = [Segment(end=60.0 * h, changes={"PODOSE_api": 5.0}) for h in (12, 24, 36, 48)]
ys = model.simulate(ts, doses)  # (time, outputs)

cve = ys[:, outputs.index("Cve_api")] * 1e6  # venous plasma, mmol/l -> nmol/l
inr = ys[:, outputs.index("INR")]


# the gradient of the mean INR with respect to every input of the model
def mean_inr(model: JaxModel) -> jax.Array:
    """Return the mean INR of the timecourse."""
    return model.simulate(ts, doses)[:, outputs.index("INR")].mean()


grad = jax.grad(mean_inr)(model).p
for name in ("Emax_INR", "EC50_api_INR", "fu_api"):
    k = model.module.INPUTS.index(name)
    print(f"d mean(INR) / d {name:<12} {float(grad[k]):+.3e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
ax1.plot(ts / 60, cve, color="tab:blue")
ax1.set(xlabel="time [h]", ylabel="apixaban [nmol/l]", title="venous plasma")
ax2.plot(ts / 60, inr, color="tab:red")
ax2.set(xlabel="time [h]", ylabel="INR [-]", title="international normalized ratio")
# --8<-- [end:example]

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/results/first-simulation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.set_xticks(range(0, 49, 12))
    fig.savefig(out, dpi=150)
    print(f"figure written: {out}")
