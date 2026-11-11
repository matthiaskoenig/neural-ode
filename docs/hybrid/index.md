# Hybrid models

!!! info "Work package 2, in progress"
    This page describes the plan. The hybrid models are developed in a new module `hybrid/`, which imports `sbml2jax` as a package; the converter itself stays unchanged.

A hybrid model keeps the known mechanistic structure of an ODE model and lets a neural network take over a term which is poorly known. Where the network sits decides what it can learn: from patient characteristics before the simulation, inside the ODE system during the simulation, or in the observation after it. A universal differential equation (UDE, [Rackauckas *et al.* 2020](../references.md#hybrid-models)) is the case where the network is part of the right-hand side:

$$
\frac{\mathrm{d} y}{\mathrm{d} t} = f(t, y, c; \, \mathrm{NN}_\theta)
$$

The networks are trained together with selected parameters of the mechanistic model by gradients through the ODE solver. The research question is which placement of a network improves the apixaban model most.

## The apixaban model

The basis is the flattened whole-body model `apixaban_body_flat.xml` ([Myshkina *et al.* 2026](../references.md#models)) which `sbml2jax` converts and verifies against roadrunner in work package 1, see [Apixaban model](../sbml2jax/apixaban.md):

- 51 states: blood and organ plasma for apixaban and the metabolites M1 and M7, the tissue of lung and rest of the body, the hepatocytes, the intestinal lumen and colon, the excreted amounts, and three dosing states.
- Elimination: renal (filtration of the unbound fraction), hepatic (CYP3A4 and CYP3A5 to M2 and M7, sulfation of M2 to M1, biliary excretion) and intestinal (secretion by P-gp and BCRP).
- Coagulation: PT, INR, aPTT, mPT and the anti-Xa activity are functions of the plasma concentration `Cve_api` (Emax or linear), not ODEs.

## Three placements of a network

| variant | network input | network output | evaluated | kind | PEtab SciML |
| --- | --- | --- | --- | --- | --- |
| A, covariate model | covariates: renal and hepatic function, body weight, age | scaling of the renal and hepatic clearance | once before the simulation | hybrid in a parameter | pre-initialization hybridization |
| B, effect model | plasma concentration `Cve_api` | PT, INR, aPTT, mPT, anti-Xa | after the simulation | hybrid in the observation | simulation hybridization (observable) |
| C, absorption | states: undissolved dose, time since the dose | dissolution rate of the tablet | in every solver step | UDE, network in the ODE | simulation hybridization (right-hand side) |

All three are scientific machine learning; strictly only C is a universal differential equation, because only there the network takes states as input and acts during the integration. A and B are a learned covariate model and a learned observation model.

Every network is initialized so that its output leaves the mechanistic law unchanged (last layer zero), so each hybrid model starts as exactly the published model.

### Variant A: covariate model

Replaces the fixed scaling of the clearance with patient characteristics: the renal clearance is linear in the creatinine clearance, cirrhosis acts only through `f_cirrhosis`, and age has no effect. A network $\mathrm{NN}_\theta$ of the covariates $x$ scales the renal excretion and the hepatic metabolism of apixaban:

$$
v_\text{renal} = v_\text{renal}^\text{mech} \cdot g_\text{renal}(x), \qquad
v_\text{hep} = v_\text{hep}^\text{mech} \cdot g_\text{hep}(x), \qquad
g(x) = \exp\big(\mathrm{NN}_\theta(x)\big)
$$

with $x$ = (log renal function, cirrhosis, log body weight relative to the reference, standardized age). Five of the 51 equations are affected; their form stays the same, only two factors depend on the patient and are constant in time. The network is very small (4 → 8 → 8 → 2) and strongly regularized. Data: studies in renal and hepatic impairment and on body weight and age; with few groups, it is evaluated on left-out groups and compared with a simple parametric covariate model.

### Variant B: effect model

Replaces the Emax laws of PT, INR, aPTT and mPT and the linear law of the anti-Xa activity:

$$
X(C) = X_\text{ref} \cdot \big(1 + E_\theta^X(C)\big), \qquad
\text{anti-Xa} = C \cdot \exp\big(h_\theta(C)\big), \qquad C = C_\text{ve,api}
$$

No ODE is affected, only the observation equations. One network with shared layers (1 → 16 → 16 → 6) gives all effects, with a penalty against decreasing effects. Derived quantities (ratios and changes to the baseline) are computed from the network outputs. As in the original fit of the pharmacodynamics, it is trained first with the pharmacokinetics fixed, on the simulated concentrations, without the ODE solver.

### Variant C: absorption (UDE)

Replaces the saturated dissolution of the tablet, which explains the less than dose-proportional exposure at high doses:

$$
v_\text{tab}^\text{mech} = \frac{K_\text{a,dis}}{60} \cdot \frac{D_\text{tab}/M_r}{1 + D_\text{tab}/D_{50}}, \qquad
v_\text{tab} = k^\text{mech}(D_\text{tab}) \cdot \exp\big(\mathrm{NN}_\theta(D_\text{tab}/D_{50}, \, t_\text{dose})\big) \cdot \frac{D_\text{tab}}{M_r}
$$

The network changes only the flux between the undissolved dose `PODOSE_api` and the intestinal lumen, so the mass balance is kept; the flux is non-negative and zero without a dose by construction. The undissolved dose changes during the simulation, so the network is evaluated in every solver step and in the Jacobian of the implicit solver; it is very small (2 → 8 → 8 → 1). Data: dose escalation studies (2.5 to 50 mg) and tablet versus solution. A synthetic test checks that the network recovers a different saturation law from simulated data; in work package 3 the learned rate is compared with the saturation law.

## Building on sbml2jax

[`JaxModel`](../sbml2jax/simulation.md) is an equinox module, so networks can be added as fields and are trained together with the inputs `p`. Each variant uses one point of it:

| variant | where the network acts |
| --- | --- |
| A | changes the inputs `p` before `initial(p)` |
| B | replaces output columns after `simulate` |
| C | overrides `rhs(t, y, c)`: the network output of the states replaces a constant in `c` (the dissolution rate constant `GU__Ka_dis_api`), then the mechanistic `rhs` is called |

The approach of C is the principle of PEtab SciML, in which a network output replaces a model parameter. Common to all variants: a data structure for the experiments, a mask of the trained parameters, a negative log-likelihood loss, Adam followed by L-BFGS, and a test that every hybrid model equals the mechanistic model at initialization.

## Export

The three variants cover the placements which [PEtab SciML](https://petab-sciml.readthedocs.io) ([Persson *et al.* 2026](../references.md#hybrid-models)) provides: pre-initialization hybridization (A), networks in the observables (B) and networks in the right-hand side (C). In work package 4 the hybrid models are exported in this format and loaded and reproduced in an independent tool. See the [Roadmap](../roadmap.md).
