# Apixaban model

The milestone of work package 1: the apixaban PBPK/PD model ([Myshkina *et al.* 2026](../references.md#models)) simulated with `sbml2jax` and diffrax matches libRoadRunner, and its gradients are correct.

## The model

`examples/models/apixaban_body_flat.xml` is the flattened whole-body model: arterial and venous blood, lung, liver, kidney, intestine and the rest of the body, with the absorption of the tablet, the hepatic metabolism to the metabolites M1, M2 and M7, renal and fecal excretion, and the effect of the plasma concentration on the coagulation markers INR, PT, aPTT, mPT and the anti-Xa activity. The comp model and its submodels are in the same directory, see `examples/models/README.md`.

| | |
| --- | --- |
| states | 51 |
| constants | 72 |
| inputs | 123 |
| outputs | 257 (species, parameters, compartments, reaction rates) |

## Comparison with roadrunner

A 10 mg oral dose is simulated over 24 hours with roadrunner and with diffrax (both with relative tolerance 1e-10 and absolute tolerance 1e-12). Every output is compared at 241 time points with

$$
|y_\text{jax} - y_\text{rr}| \le 10^{-9} + 10^{-5} \, |y_\text{rr}|
$$

The gradient of the mean plasma concentration with respect to the rate constant of the hepatic metabolism `LI__API2M_k` is compared with central finite differences.

```bash
cd sbml2jax
uv run python examples/apixaban_example.py
```

```text
51 states, 72 constants, 123 inputs, 257 outputs
jax simulation (incl. compilation): 9.2 s
outputs within tolerance: 257 of 257
  PODOSE_api                     max relative difference 5.26e-03
  GU__APITAB                     max relative difference 4.89e-06
  LI__M7EX                       max relative difference 8.03e-06
  GU__m2_colon                   max relative difference 5.68e-06
  GU__APIUNDIS                   max relative difference 1.92e-06
d mean(Cve_api) / d LI__API2M_k: autodiff -5.595828e-04, finite differences -5.595828e-04
```

All 257 outputs are within the tolerance. The largest relative differences are in quantities which decay towards zero, such as the remaining dose in the tablet `PODOSE_api`, where the absolute difference is below the absolute tolerance. The derivative by automatic differentiation agrees with finite differences to all printed digits.

The same checks run in CI as `tests/test_apixaban.py`, on 97 time points.

??? example "The script `examples/apixaban_example.py`"

    ```python
    --8<-- "sbml2jax/examples/apixaban_example.py"
    ```
