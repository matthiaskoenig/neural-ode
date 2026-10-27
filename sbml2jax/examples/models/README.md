# Models

The apixaban PBPK/PD models (`apixaban_*.xml`) are copied unchanged from
https://github.com/matthiaskoenig/apixaban-model (directory `models/`, commit
d0e3277). Myshkina M. et al. (2026), doi:10.5281/zenodo.18877904, licensed
CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).

`apixaban_body_flat.xml` is the flattened whole-body model that sbml2jax
verifies against roadrunner; the other files are the comp model and its
submodels (liver, kidney, coagulation).
