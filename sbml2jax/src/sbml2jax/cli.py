"""`sbml2jax`: convert an SBML model into a JAX module.

sbml2jax model.xml                # writes model.py next to the input
sbml2jax model.xml -o out/model.py
sbml2jax model.xml -v             # log the conversion
"""

import argparse
import logging
import sys
from pathlib import Path

from sbml2jax.codegen import CodegenError
from sbml2jax.convert import convert_sbml2jax
from sbml2jax.jaxmath import UnsupportedError
from sbml2jax.log import enable_rich_logging
from sbml2jax.sbml import SBMLError


def main(argv: list[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(prog="sbml2jax", description="Convert SBML into a JAX module.")
    parser.add_argument("sbml", type=Path, help="SBML file")
    parser.add_argument("-o", "--output", type=Path, help="module file, default <sbml stem>.py")
    parser.add_argument("-v", "--verbose", action="store_true", help="log the conversion")
    args = parser.parse_args(argv)
    enable_rich_logging(logging.INFO if args.verbose else logging.WARNING)
    output = args.output or args.sbml.with_suffix(".py")
    try:
        convert_sbml2jax(args.sbml, output)
    except (SBMLError, UnsupportedError, CodegenError) as err:
        print(f"sbml2jax: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
