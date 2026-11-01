"""Analysis of an SBML model into the parts of its ODE system.

`analyse` turns a prepared libsbml model (see `sbml.read_document`) into a
`ModelSystem`, which `codegen` writes as a Python module. In sbml2cellml the
libcellml analyser does this work; here it is done explicitly.

Quantities and their identifiers:

- every compartment, parameter, species, reaction and species reference with
  an id is a symbol with its SBML value (a concentration species its
  concentration, a reaction its rate);
- a species which no rule determines has an amount (`<id>_amount`, the id
  itself when `hasOnlySubstanceUnits`): SBML keeps the amount when the size of
  the compartment changes, so the amount is the state of the ODE system and
  the concentration `amount / compartment`;
- a local parameter of a kinetic law is `<reaction>_<id>`.

Every identifier is in one of three roles at simulation time:

- state: a rate rule or the reactions change it (`ModelSystem.states`);
- constant: it does not change, its value is fixed at the start
  (`ModelSystem.constants`, the `c` vector of the generated code);
- computed: assignment rules, concentrations from amounts, reaction rates,
  derivatives (`ModelSystem.computed`).

The values at the start (`ModelSystem.initial`) are definitions of their own:
an initial assignment, else the assignment rule, else the value of the
attribute. The attribute values are the inputs (`ModelSystem.inputs`, the `p`
vector), so gradients reach every number of the model, also through the
initial assignments.

Not supported (raise `UnsupportedError`): events, algebraic rules, delays,
fast reactions, rateOf in initial assignments and of rule targets, rate rules
of species references.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import libsbml

from sbml2jax.jaxmath import Expr, UnsupportedError, number, translate
from sbml2jax.variables import Names

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeciesInfo:
    """Species of the model, for the conversion of amounts and concentrations."""

    sid: str
    compartment: str
    only_substance_units: bool
    amount: str | None
    """identifier of the amount, None if a rule determines the species"""


@dataclass
class ModelSystem:
    """The ODE system of a model, in identifiers of the generated code."""

    model_id: str
    inputs: dict[str, float] = field(default_factory=dict)
    """input identifier -> value of the SBML attribute"""
    initial: dict[str, Expr] = field(default_factory=dict)
    """identifier -> definition at the start (inputs are their own definition)"""
    states: list[str] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    computed: dict[str, Expr] = field(default_factory=dict)
    """identifier -> definition during the simulation"""
    derivatives: dict[str, str] = field(default_factory=dict)
    """state -> identifier of its derivative (in `computed`)"""
    outputs: dict[str, str] = field(default_factory=dict)
    """SBML id -> identifier of its value"""
    species: dict[str, SpeciesInfo] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _check_supported(model: libsbml.Model) -> None:
    """Raise `UnsupportedError` for the constructs the converter cannot convert."""
    if model.getNumEvents():
        raise UnsupportedError("events", f"{model.getNumEvents()} events")
    for rule in model.getListOfRules():
        if rule.isAlgebraic() and rule.isSetMath():
            raise UnsupportedError("algebraic rules")
    for reaction in model.getListOfReactions():
        if reaction.isSetFast() and reaction.getFast():
            raise UnsupportedError("fast reactions", reaction.getId())
    for k in range(model.getNumConstraints()):
        logger.warning("constraint ignored: %s", model.getConstraint(k).getMetaId())


class _Builder:
    """Builds the `ModelSystem` of one model."""

    def __init__(self, model: libsbml.Model) -> None:
        self.model = model
        self.names = Names()
        self.system = ModelSystem(model_id=model.getId() or "model")
        self.assignment_rules: dict[str, libsbml.ASTNode] = {}
        self.rate_rules: dict[str, libsbml.ASTNode] = {}
        self.initial_assignments: dict[str, libsbml.ASTNode] = {}
        for rule in model.getListOfRules():
            if not rule.isSetMath():
                # a rule without math has no effect (SBML L3V2, section 4.11)
                logger.warning("rule of '%s' has no math, it is ignored", rule.getVariable())
                continue
            if rule.isAssignment():
                self.assignment_rules[rule.getVariable()] = rule.getMath()
            elif rule.isRate():
                self.rate_rules[rule.getVariable()] = rule.getMath()
        for ia in model.getListOfInitialAssignments():
            if ia.isSetMath():
                self.initial_assignments[ia.getSymbol()] = ia.getMath()
        # species reference ids -> species reference
        self.species_references: dict[str, libsbml.SpeciesReference] = {}
        for reaction in model.getListOfReactions():
            for ref in list(reaction.getListOfReactants()) + list(reaction.getListOfProducts()):
                if ref.isSetId():
                    self.species_references[ref.getId()] = ref
        # states whose derivative depends on reactions
        self.changed_by_reactions = self._changed_by_reactions()

    # names -----------------------------------------------------------------

    def name(self, sid: str) -> str:
        """Return the identifier of the SBML symbol `sid`."""
        return self.names.get(sid)

    def amount_name(self, species: libsbml.Species) -> str:
        """Return the identifier of the amount of `species`."""
        sid = species.getId()
        if species.getHasOnlySubstanceUnits():
            return self.name(sid)
        return self.names.get(f"{sid}#amount", f"{sid}_amount")

    def resolver(self, local: dict[str, str] | None = None) -> Callable[[str], str]:
        """Return a resolver of ids, with the local parameters `local` first."""
        local = local or {}

        def resolve(sid: str) -> str:
            if sid in local:
                return local[sid]
            return self.name(sid)

        return resolve

    # helpers ---------------------------------------------------------------

    def _changed_by_reactions(self) -> set[str]:
        """Return the ids of the species which reactions change."""
        changed = set()
        for reaction in self.model.getListOfReactions():
            if not reaction.isSetKineticLaw():
                continue
            for ref in list(reaction.getListOfReactants()) + list(reaction.getListOfProducts()):
                species = self.model.getSpecies(ref.getSpecies())
                if species is not None and not species.getBoundaryCondition():
                    changed.add(species.getId())
        return changed

    def input(self, key: str, value: float, base: str | None = None) -> str:
        """Register an input with the attribute `value` and return its identifier."""
        name = self.names.get(key, base)
        self.system.inputs[name] = value
        return name

    def set_initial(self, name: str, expr: Expr) -> None:
        """Set the definition of `name` at the start."""
        if expr.code != name:
            self.system.initial[name] = expr

    # quantities ------------------------------------------------------------

    def add_value_symbol(self, element: libsbml.SBase, value: float | None) -> None:
        """Add a compartment, parameter or species reference with a value attribute."""
        sid = element.getId()
        name = self.name(sid)
        self.system.outputs[sid] = name

        if sid in self.initial_assignments:
            self.set_initial(name, translate(self.initial_assignments[sid], self.resolver()))
        elif sid in self.assignment_rules:
            self.set_initial(name, translate(self.assignment_rules[sid], self.resolver()))
        else:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                self.system.warnings.append(f"'{sid}' has no value, NaN is used")
                value = math.nan
            self.input(sid, value)

        if sid in self.assignment_rules:
            self.system.computed[name] = translate(self.assignment_rules[sid], self.resolver(), self.rate_of)
        elif sid in self.rate_rules:
            if isinstance(element, libsbml.SpeciesReference):
                raise UnsupportedError("rate rule of a species reference", sid)
            self.add_state(name, translate(self.rate_rules[sid], self.resolver(), self.rate_of))
        else:
            self.system.constants.append(name)

    def add_state(self, name: str, derivative: Expr) -> None:
        """Add the state `name` with the formula of its derivative."""
        d = self.names.get(f"{name}#rate", f"d_{name}")
        self.system.states.append(name)
        self.system.computed[d] = derivative
        self.system.derivatives[name] = d

    def add_species(self, species: libsbml.Species) -> None:
        """Add a species: its value, its amount and their definitions."""
        sid = species.getId()
        name = self.name(sid)
        compartment = self.name(species.getCompartment())
        only_substance = species.getHasOnlySubstanceUnits()
        self.system.outputs[sid] = name

        # value at the start, in the units of the species value
        if sid in self.initial_assignments:
            self.set_initial(name, translate(self.initial_assignments[sid], self.resolver()))
        elif sid in self.assignment_rules:
            self.set_initial(name, translate(self.assignment_rules[sid], self.resolver()))
        elif species.isSetInitialAmount():
            value = species.getInitialAmount()
            if only_substance:
                self.input(sid, value)
            else:
                amount = self.input(f"{sid}#initial_amount", value, f"{sid}_initial_amount")
                self.set_initial(name, Expr(f"({amount} / {compartment})", frozenset({amount, compartment})))
        elif species.isSetInitialConcentration():
            value = species.getInitialConcentration()
            if not only_substance:
                self.input(sid, value)
            else:
                conc = self.input(f"{sid}#initial_concentration", value, f"{sid}_initial_concentration")
                self.set_initial(name, Expr(f"({conc} * {compartment})", frozenset({conc, compartment})))
        else:
            self.system.warnings.append(f"species '{sid}' has no initial value, NaN is used")
            self.input(sid, math.nan)

        if sid in self.assignment_rules:
            self.system.computed[name] = translate(self.assignment_rules[sid], self.resolver(), self.rate_of)
            self.system.species[sid] = SpeciesInfo(sid, species.getCompartment(), only_substance, None)
            return
        if sid in self.rate_rules:
            # the rate rule changes the species value (concentration or amount)
            self.add_state(name, translate(self.rate_rules[sid], self.resolver(), self.rate_of))
            self.system.species[sid] = SpeciesInfo(sid, species.getCompartment(), only_substance, None)
            return

        amount = self.amount_name(species)
        if not only_substance:
            self.set_initial(amount, Expr(f"({name} * {compartment})", frozenset({name, compartment})))
            self.system.computed[name] = Expr(f"({amount} / {compartment})", frozenset({amount, compartment}))
        if sid in self.changed_by_reactions:
            self.add_state(amount, self.reaction_terms(species))
        else:
            self.system.constants.append(amount)
        self.system.species[sid] = SpeciesInfo(sid, species.getCompartment(), only_substance, amount)

    def stoichiometry(self, reaction: libsbml.Reaction, ref: libsbml.SpeciesReference) -> Expr:
        """Return the stoichiometry of the species reference `ref`."""
        if ref.isSetId():
            name = self.name(ref.getId())
            return Expr(name, frozenset({name}))
        if ref.isSetStoichiometryMath():  # level 2
            math_ = ref.getStoichiometryMath().getMath()
            name = self.names.get(
                f"{reaction.getId()}#{ref.getSpecies()}#stoichiometry",
                f"{reaction.getId()}_{ref.getSpecies()}_stoichiometry",
            )
            self.system.computed[name] = translate(math_, self.resolver(), self.rate_of)
            return Expr(name, frozenset({name}))
        if ref.isSetStoichiometry():
            return number(ref.getStoichiometry())
        if self.model.getLevel() < 3:
            return number(1.0)
        self.system.warnings.append(
            f"stoichiometry of '{ref.getSpecies()}' in '{reaction.getId()}' is not set, 1 is used"
        )
        return number(1.0)

    def reaction_terms(self, species: libsbml.Species) -> Expr:
        """Return the derivative of the amount of `species` from the reactions."""
        sid = species.getId()
        terms: list[str] = []
        deps: set[str] = set()
        for reaction in self.model.getListOfReactions():
            if not reaction.isSetKineticLaw():
                continue
            rate = self.name(reaction.getId())
            for refs, sign in ((reaction.getListOfReactants(), "-"), (reaction.getListOfProducts(), "+")):
                for ref in refs:
                    if ref.getSpecies() != sid:
                        continue
                    stoichiometry = self.stoichiometry(reaction, ref)
                    deps |= stoichiometry.deps | {rate}
                    factor = "" if stoichiometry.code == "1.0" else f"{stoichiometry.code} * "
                    terms.append(f"{sign} {factor}{rate}")
        if not terms:
            return Expr("0.0")
        code = " ".join(terms)
        code = code[2:] if code.startswith("+ ") else f"-{code[2:]}" if code.startswith("- ") else code
        factor = species.getConversionFactor() if species.isSetConversionFactor() else ""
        if not factor and self.model.isSetConversionFactor():
            factor = self.model.getConversionFactor()
        if factor:
            f = self.name(factor)
            deps.add(f)
            return Expr(f"{f} * ({code})", frozenset(deps))
        return Expr(f"({code})", frozenset(deps))

    def add_reaction(self, reaction: libsbml.Reaction) -> None:
        """Add the rate of `reaction`, with its local parameters as inputs."""
        rid = reaction.getId()
        if not reaction.isSetKineticLaw() or not reaction.getKineticLaw().isSetMath():
            self.system.warnings.append(f"reaction '{rid}' has no kinetic law, it is ignored")
            return
        law = reaction.getKineticLaw()
        local: dict[str, str] = {}
        parameters = list(law.getListOfLocalParameters()) + list(law.getListOfParameters())
        for parameter in parameters:
            pid = parameter.getId()
            name = self.input(f"{rid}#{pid}", parameter.getValue(), f"{rid}_{pid}")
            self.system.constants.append(name)
            local[pid] = name
        name = self.name(rid)
        self.system.outputs[rid] = name
        self.system.computed[name] = translate(law.getMath(), self.resolver(local), self.rate_of)
        self.set_initial(name, translate(law.getMath(), self.resolver(local)))

    # rateOf ----------------------------------------------------------------

    def rate_of(self, sid: str) -> Expr:
        """Return the expression of `rateOf(sid)` during the simulation."""
        name = self.name(sid)
        if sid in self.assignment_rules:
            raise UnsupportedError("rateOf of an assignment rule target", sid)
        if name in self.system.derivatives:
            d = self.system.derivatives[name]
            return Expr(d, frozenset({d}))
        species = self.model.getSpecies(sid)
        if species is not None and sid not in self.rate_rules:
            amount = self.amount_name(species)
            # the state of the amount may not be added yet: name its derivative
            d_amount = self.names.get(f"{amount}#rate", f"d_{amount}")
            if sid not in self.changed_by_reactions:
                d_amount_expr = "0.0"
                deps: frozenset[str] = frozenset()
            else:
                d_amount_expr = d_amount
                deps = frozenset({d_amount})
            if species.getHasOnlySubstanceUnits():
                return Expr(d_amount_expr, deps)
            compartment = species.getCompartment()
            c = self.name(compartment)
            r = self.names.get(f"{sid}#rateOf", f"rate_of_{sid}")
            rc = self.rate_of(compartment)
            # d(A/C)/dt = (dA/dt - S * dC/dt) / C
            self.system.computed[r] = Expr(
                f"(({d_amount_expr} - {name} * {rc.code}) / {c})",
                deps | rc.deps | {name, c},
            )
            return Expr(r, frozenset({r}))
        if sid in self.rate_rules:
            # derivative of a rate rule state added later
            d = self.names.get(f"{name}#rate", f"d_{name}")
            return Expr(d, frozenset({d}))
        return Expr("0.0")

    # model -----------------------------------------------------------------

    def build(self) -> ModelSystem:
        """Analyse the model."""
        _check_supported(self.model)
        model = self.model
        # reserve the SBML ids first, derived names get suffixes on clashes
        for elements in (
            model.getListOfCompartments(),
            model.getListOfParameters(),
            model.getListOfSpecies(),
            model.getListOfReactions(),
        ):
            for element in elements:
                self.name(element.getId())
        for sid in self.species_references:
            self.name(sid)

        for compartment in model.getListOfCompartments():
            value = compartment.getSize() if compartment.isSetSize() else None
            self.add_value_symbol(compartment, value)
        for parameter in model.getListOfParameters():
            value = parameter.getValue() if parameter.isSetValue() else None
            self.add_value_symbol(parameter, value)
        for sid, ref in self.species_references.items():
            value = ref.getStoichiometry() if ref.isSetStoichiometry() else None
            if value is None and sid not in self.initial_assignments and sid not in self.assignment_rules:
                value = 1.0
            self.add_value_symbol(ref, value)
        for reaction in model.getListOfReactions():
            self.add_reaction(reaction)
        for species in model.getListOfSpecies():
            self.add_species(species)

        if not self.system.states:
            # diffrax integrates a state vector: a model without ODEs gets a
            # constant dummy state, its outputs are functions of time
            dummy = self.names.get("#dummy", "dummy_state")
            self.system.initial[dummy] = number(0.0)
            self.add_state(dummy, number(0.0))

        for warning in self.system.warnings:
            logger.warning(warning)
        return self.system


def analyse(model: libsbml.Model) -> ModelSystem:
    """Analyse the ODE system of the prepared SBML `model`.

    Args:
        model: a libsbml model without function definitions and comp elements.

    Returns:
        The states, constants, computed quantities and initial values.

    Raises:
        UnsupportedError: the model uses a construct which is not supported.
    """
    return _Builder(model).build()
