"""Conversion of an SBML file into a JAX model module.

`convert_sbml2jax` reads the SBML file (`sbml.read_document`), analyses its ODE
system (`analysis.analyse`), writes the Python module (`codegen.generate`) and
returns its source. `load_module` imports a generated module from a file or
from source.
"""

import importlib.util
import logging
import sys
import types
from pathlib import Path

from sbml2jax.analysis import analyse
from sbml2jax.codegen import generate
from sbml2jax.sbml import read_document

logger = logging.getLogger(__name__)


def convert_sbml2jax(sbml_path: Path, module_path: Path | None = None) -> str:
    """Convert the SBML model `sbml_path` into the source of a JAX module.

    Args:
        sbml_path: the SBML file.
        module_path: if set, the module is written to this file.

    Returns:
        The module source.

    Raises:
        SBMLError: the file cannot be read.
        UnsupportedError: the model uses a construct which is not supported.
        CodegenError: the definitions are cyclic or incomplete.
    """
    doc = read_document(sbml_path)
    system = analyse(doc.getModel())
    source = generate(system, source=sbml_path.name)
    if module_path is not None:
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text(source)
        logger.info("JAX module written: %s", module_path)
    return source


def load_module(
    path: Path | None = None, source: str | None = None, name: str = "sbml2jax_model"
) -> types.ModuleType:
    """Import a generated module from the file `path` or from `source`."""
    if path is not None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    if source is None:
        raise ValueError("either path or source is required")
    module = types.ModuleType(name)
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)  # noqa: S102
    return module
