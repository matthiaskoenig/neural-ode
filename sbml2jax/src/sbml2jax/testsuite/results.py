"""Results of a test suite run, saved as JSON and compared between runs."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

STAGES = ("roadrunner", "sbml2jax", "jax")
STATUSES = ("pass", "fail", "skip")
CATEGORIES = ("pass", "fail", "unsupported", "skip")


@dataclass(frozen=True)
class StageResult:
    """Status of one stage and the message of a failure or skip."""

    status: str
    message: str = ""


@dataclass
class CaseResult:
    """Stages of one case."""

    id: str
    stages: dict[str, StageResult]
    tags: list[str] = field(default_factory=list)

    @property
    def category(self) -> str:
        """Return `pass`, `fail` (simulation wrong), `unsupported` (conversion failed) or `skip`."""
        if self.stages.get("sbml2jax", StageResult("skip")).status == "fail":
            return "unsupported"
        return self.stages.get("jax", StageResult("skip")).status

    @property
    def passed(self) -> bool:
        """Whether the JAX simulation matches the expected results."""
        return self.stages.get("jax", StageResult("skip")).status == "pass"


def save(results: list[CaseResult], path: Path) -> None:
    """Write `results` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {r.id: {"stages": {k: asdict(v) for k, v in r.stages.items()}, "tags": r.tags} for r in results}
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def load(path: Path) -> list[CaseResult]:
    """Read results written by `save`."""
    data = json.loads(path.read_text())
    return [
        CaseResult(cid, {k: StageResult(**v) for k, v in item["stages"].items()}, item.get("tags", []))
        for cid, item in sorted(data.items())
    ]


def regressions(old: list[CaseResult], new: list[CaseResult]) -> list[str]:
    """Return the ids of the cases which passed in `old` and do not in `new`."""
    passed = {r.id for r in old if r.passed}
    return sorted(r.id for r in new if r.id in passed and not r.passed)


def improvements(old: list[CaseResult], new: list[CaseResult]) -> list[str]:
    """Return the ids of the cases which pass in `new` and did not in `old`."""
    passed = {r.id for r in old if r.passed}
    return sorted(r.id for r in new if r.passed and r.id not in passed)
