# References

The publications behind the models, standards and tools of the project. Cite the project itself as described in [Home](index.md#how-to-cite).

## Models

**Apixaban PBPK/PD model.** The whole-body model of apixaban and its effect on coagulation, the application of the thesis. The SBML files in `sbml2jax/examples/models/` are from [apixaban-model](https://github.com/matthiaskoenig/apixaban-model) (commit d0e3277), licensed CC BY-SA 4.0.

> Myshkina M, *et al.*
> **Apixaban PBPK/PD model.**
> 2026.
> [doi:10.5281/zenodo.18877904](https://doi.org/10.5281/zenodo.18877904)

## Hybrid models

**Universal differential equations.** Mechanistic differential equations with neural networks for unknown terms, trained by gradients through the solver.

> Rackauckas C, Ma Y, Martensen J, Warner C, Zubov K, Supekar R, Skinner D, Ramadhan A, Edelman A.
> **Universal differential equations for scientific machine learning.**
> *arXiv.* 2020;2001.04385.
> [arXiv:2001.04385](https://arxiv.org/abs/2001.04385)

**PEtab SciML.** The exchange format of hybrid problems of a mechanistic model and neural networks, in which the hybrid models are exported.

> Persson S, Snelling B, Philipps M, Weindl D, Cvijovic M, Hasenauer J, Pathirana D, Fröhlich F.
> **PEtab SciML: an exchange format for specifying and training dynamic scientific machine learning models.**
> *arXiv.* 2026;2608.20184.
> [arXiv:2608.20184](https://arxiv.org/abs/2608.20184)

## Standards

**SBML Level 3.** The format of the models.

> Keating SM, Waltemath D, König M, Zhang F, Dräger A, Chaouiya C, Bergmann FT, Finney A, Gillespie CS, Helikar T, Hoops S, Malik-Sheriff RS, Moodie SL, Moraru II, Myers CJ, Naldi A, Olivier BG, Sahle S, Schaff JC, Smith LP, Swat MJ, Thieffry D, Watanabe L, Wilkinson DJ, Blinov ML, Begley K, Faeder JR, Gómez HF, Hamm TM, Inagaki Y, Liebermeister W, Lister AL, Lucio D, Mjolsness E, Proctor CJ, Raman K, Rodriguez N, Shaffer CA, Shapiro BE, Stelling J, Swainston N, Tanimura N, Wagner J, Meier-Schellersheim M, Sauro HM, Palsson B, Bolouri H, Kitano H, Funahashi A, Hermjakob H, Doyle JC, Hucka M; SBML Level 3 Community members.
> **SBML Level 3: an extensible format for the exchange and reuse of biological models.**
> *Molecular Systems Biology.* 2020;16(8):e9110.
> [doi:10.15252/msb.20199110](https://doi.org/10.15252/msb.20199110)

**PEtab.** The specification of parameter estimation problems which PEtab SciML extends.

> Schmiester L, Schälte Y, Bergmann FT, Camba T, Dudkin E, Egert J, Fröhlich F, Fuhrmann L, Hauber AL, Kemmer S, Lakrisenko P, Loos C, Merkt S, Müller W, Pathirana D, Raimúndez E, Refisch L, Rosenblatt M, Stapor PL, Städter P, Wang D, Wieland FG, Banga JR, Timmer J, Villaverde AF, Sahle S, Kreutz C, Hasenauer J, Weindl D.
> **PEtab - Interoperable specification of parameter estimation problems in systems biology.**
> *PLoS Computational Biology.* 2021;17(1):e1008646.
> [doi:10.1371/journal.pcbi.1008646](https://doi.org/10.1371/journal.pcbi.1008646)

**SBML test suite.** The semantic test cases against which `sbml2jax` is verified, version 3.5.0, see [SBML test suite](sbml2jax/testsuite.md). [github.com/sbmlteam/sbml-test-suite](https://github.com/sbmlteam/sbml-test-suite)

## Simulation

**libRoadRunner.** The reference simulator of the mechanistic models.

> Welsh C, Xu J, Smith L, König M, Choi K, Sauro HM.
> **libRoadRunner 2.0: a high performance SBML simulation and analysis library.**
> *Bioinformatics.* 2023;39(1):btac770.
> [doi:10.1093/bioinformatics/btac770](https://doi.org/10.1093/bioinformatics/btac770)

**libSBML.** Reading, validating and flattening of the SBML models.

> Bornstein BJ, Keating SM, Jouraku A, Hucka M.
> **LibSBML: an API library for SBML.**
> *Bioinformatics.* 2008;24(6):880-881.
> [doi:10.1093/bioinformatics/btn051](https://doi.org/10.1093/bioinformatics/btn051)

## JAX

**JAX.** Arrays, automatic differentiation and compilation. [github.com/jax-ml/jax](https://github.com/jax-ml/jax)

> Bradbury J, Frostig R, Hawkins P, Johnson MJ, Leary C, Maclaurin D, Necula G, Paszke A, VanderPlas J, Wanderman-Milne S, Zhang Q.
> **JAX: composable transformations of Python+NumPy programs.**
> 2018. [github.com/jax-ml/jax](https://github.com/jax-ml/jax)

**diffrax.** The ODE solvers, differentiable through the solve.

> Kidger P.
> **On neural differential equations.**
> PhD thesis, University of Oxford. 2021.
> [arXiv:2202.02435](https://arxiv.org/abs/2202.02435)

**equinox.** The models as differentiable pytrees, and the neural networks.

> Kidger P, Garcia C.
> **Equinox: neural networks in JAX via callable PyTrees and filtered transformations.**
> *Differentiable Programming workshop at Neural Information Processing Systems.* 2021.
> [arXiv:2111.00254](https://arxiv.org/abs/2111.00254)
