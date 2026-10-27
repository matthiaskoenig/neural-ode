"""Work package 1 milestone: the apixaban model matches roadrunner, gradients are correct."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import roadrunner

from sbml2jax.model import JaxModel, Segment, SolverSettings

DOSE = 10.0
TS = np.linspace(0.0, 24 * 60.0, 97)
SETTINGS = SolverSettings(rtol=1e-10, atol=1e-12)
SEGMENTS = [Segment(end=float(TS[-1]), changes={"PODOSE_api": DOSE})]


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> JaxModel:
    sbml = Path(__file__).parent.parent / "examples" / "models" / "apixaban_body_flat.xml"
    return JaxModel.from_sbml(sbml, tmp_path_factory.mktemp("apixaban") / "apixaban.py")


def test_matches_roadrunner(model: JaxModel) -> None:
    m = model.module
    sbml = Path(__file__).parent.parent / "examples" / "models" / "apixaban_body_flat.xml"
    rr = roadrunner.RoadRunner(str(sbml))
    rr.integrator.setValue("relative_tolerance", 1e-10)
    rr.integrator.setValue("absolute_tolerance", 1e-12)
    rr.timeCourseSelections = [f"[{o}]" if o in m.SPECIES and not m.SPECIES[o][1] else o for o in m.OUTPUTS]
    rr.setValue("PODOSE_api", DOSE)
    expected = np.asarray(rr.simulate(times=TS.tolist()))
    ys = np.asarray(model.simulate(jnp.asarray(TS), SEGMENTS, SETTINGS))
    excess = np.abs(ys - expected) - (1e-9 + 1e-5 * np.abs(expected))
    failed = [m.OUTPUTS[k] for k in range(len(m.OUTPUTS)) if excess[:, k].max() > 0]
    assert not failed


def test_gradient_matches_finite_differences(model: JaxModel) -> None:
    target = model.module.OUTPUTS.index("Cve_api")

    def loss(m: JaxModel) -> jax.Array:
        return jnp.mean(m.simulate(jnp.asarray(TS), SEGMENTS, SETTINGS)[:, target])

    name = "LI__API2M_k"
    k = model.module.INPUTS.index(name)
    value = float(model.p[k])
    h = 1e-6 * max(abs(value), 1.0)
    fd = (loss(model.with_inputs({name: value + h})) - loss(model.with_inputs({name: value - h}))) / (2 * h)
    assert float(jax.grad(loss)(model).p[k]) == pytest.approx(float(fd), rel=1e-5)
