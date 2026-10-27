"""Reading and preparing SBML models with libsbml.

`read_model` reads an SBML file, flattens hierarchical models (comp package)
and expands the calls of function definitions, so the converter only sees
core SBML formulas without user functions.
"""

import logging
from pathlib import Path

import libsbml

logger = logging.getLogger(__name__)


class SBMLError(ValueError):
    """The SBML file cannot be read or prepared."""


def _convert(doc: libsbml.SBMLDocument, option: str, description: str) -> None:
    """Run the libsbml conversion `option` on `doc`, raise on failure."""
    props = libsbml.ConversionProperties()
    props.addOption(option, True, description)
    status = doc.convert(props)
    if status != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise SBMLError(f"libsbml conversion '{option}' failed: {_messages(doc)}")


def _messages(doc: libsbml.SBMLDocument, errors_only: bool = True) -> str:
    """Return the messages of the error log of `doc`."""
    log = doc.getErrorLog()
    messages = []
    for k in range(log.getNumErrors()):
        error = log.getError(k)
        if errors_only and error.getSeverity() < libsbml.LIBSBML_SEV_ERROR:
            continue
        messages.append(error.getMessage().strip())
    return "; ".join(messages)


def read_document(sbml_path: Path) -> libsbml.SBMLDocument:
    """Read `sbml_path`, flatten comp models and expand function definitions.

    Args:
        sbml_path: path of the SBML file.

    Returns:
        The prepared document; its model contains core SBML only.

    Raises:
        SBMLError: the file does not exist, has read errors or no model.
    """
    if not sbml_path.exists():
        raise SBMLError(f"SBML file does not exist: {sbml_path}")
    doc = libsbml.readSBMLFromFile(str(sbml_path))
    if doc.getNumErrors(libsbml.LIBSBML_SEV_FATAL) or doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR):
        raise SBMLError(f"SBML read errors in {sbml_path}: {_messages(doc)}")
    if doc.getModel() is None:
        raise SBMLError(f"SBML file has no model: {sbml_path}")

    if doc.getPlugin("comp") is not None and doc.isPackageEnabled("comp"):
        logger.info("flattening comp model %s", sbml_path.name)
        _convert(doc, "flatten comp", "flatten hierarchical model")

    if doc.getModel().getNumFunctionDefinitions():
        _convert(doc, "expandFunctionDefinitions", "expand function definitions")
    return doc
