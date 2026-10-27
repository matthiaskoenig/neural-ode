# Conversion issues

What the current version does not convert, or converts with a loss. Each item
raises `UnsupportedError` with the feature name, so it shows up as a failure
reason in [the test suite report](testsuite.md).

In a first run over about 740 cases of the SBML test suite (3.5.0), 478
passed. All of the failures were cases with events (163) or algebraic rules
(62).

## Not supported

- **Events** (with and without delay). The plan is to support them as
  diffrax events (root finding on the trigger, then a restart with the
  assignments). Dosing does not need them: doses are `Segment` changes, see
  `model.py`.
- **Algebraic rules.** They would need a DAE or nonlinear solve. They are
  rare in PK/PD models.
- **The delay csymbol.**
- **Fast reactions** (level 2).
- **rateOf** of an assignment rule target, and rateOf in initial values
  (initial assignments).
- **Rate rules of species references** (stoichiometry that changes in time).

## Behaviour to be aware of

- **Units are not converted.** Numbers are used as they are, as in
  roadrunner.
- **Species without an initial value** and parameters without a value are
  NaN, with a warning. The apixaban model has two unused parameters of this
  kind (`LI__Vmem`, `GU__Vapical`).
- **Constraints** are ignored, with a warning.
- **Changes during a simulation** (`Segment.changes`) set states or constants
  directly. Initial assignments are not evaluated again, which matches
  roadrunner when a value is set during a timecourse.
- **`JaxModel.simulate` needs concrete output times**, since the segment logic
  runs in Python. Gradients with respect to the inputs work, and `vmap` over
  inputs is planned for work package 2.
