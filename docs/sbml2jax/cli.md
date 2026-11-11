# Command line

`sbml2jax` installs two commands.

## sbml2jax

Converts an SBML model into a JAX module, see [Generated module](generated-module.md):

```bash
sbml2jax model.xml                 # writes model.py next to the input
sbml2jax model.xml -o out/model.py
sbml2jax model.xml -v              # log the conversion
```

```text
usage: sbml2jax [-h] [-o OUTPUT] [-v] sbml

Convert SBML into a JAX module.

positional arguments:
  sbml                 SBML file

options:
  -h, --help           show this help message and exit
  -o, --output OUTPUT  module file, default <sbml stem>.py
  -v, --verbose        log the conversion
```

The exit code is 1 if the file cannot be read (`SBMLError`), uses a construct which is not supported (`UnsupportedError`, see [Conversion issues](conversion-issues.md)) or has cyclic definitions (`CodegenError`).

## sbml2jax-testsuite

Runs the [SBML test suite](testsuite.md) (needs the `testsuite` extra). The suite 3.5.0 is downloaded into `~/.cache/sbml2jax` on the first run.

```bash
sbml2jax-testsuite run                 # all cases, about 40 min on 2 CPUs
sbml2jax-testsuite run 00001 00002     # selected cases, merged into the results
sbml2jax-testsuite run --first 200     # the first 200 cases
sbml2jax-testsuite run -j 8            # with 8 worker processes
sbml2jax-testsuite report              # rerender the report from the results
```

`run` writes the results to `testsuite/results.json` and the report to `docs/testsuite.md` in the `sbml2jax` directory. Both are committed; the documentation includes the report.
