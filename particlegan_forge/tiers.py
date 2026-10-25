"""First-class Tier API and workspace engine for particlegan_forge.

``particlegan_forge`` is the small, wheel-shipped facade of the repository's
``experiments/forge`` engine (which is *not* packaged).  It keeps its own
on-disk namespace -- ``configs/particlegan_forge/`` and
``reports/particlegan_forge/`` (operational logs under
``runs/particlegan_forge/``) -- and refuses to load files written with the
``experiments/forge`` schema, so it can never overwrite ``reports/forge`` or
``configs/forge`` outputs.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import inspect
import json
import math
import numbers
from pathlib import Path
import signal
import threading
import time
import traceback
from typing import Any, Callable

from .contracts import (
    HOLD_REASONS,
    IMPORTANCES,
    REMOVAL_REASONS,
    atomic_json,
    atomic_text,
    file_hash,
    file_lock,
    has_nonfinite,
    identifier,
    json_safe,
    read_json,
    stable_hash,
    utc_now,
)
from .rng import RNG_VERSION, NamedStreams, deterministic_scope

NAMESPACE = "particlegan_forge"
"""Schema/namespace marker carried by every particlegan_forge declaration."""

RECEIPT_SCHEMA_VERSION = 2
_ENGINE_SOURCES = ("tiers.py", "rng.py", "contracts.py")
_TIER_STATUS_PRECEDENCE = ("FAIL", "INVALID", "BLOCKED", "INCOMPLETE", "NOT_RUN")


def resolve_device(device: str | None = None) -> str:
    """Resolve the receipt cohort device identically for run, board and compile.

    ``None`` means ``"cpu"`` (the CLI default for every command); ``"auto"``
    selects ``"cuda"`` only when CUDA is available.  ``"cuda:N"`` maps to the
    ``"cuda"`` cohort.  The result never depends on which receipts exist.
    """
    if device is None:
        return "cpu"
    if not isinstance(device, str):
        raise ValueError(f"device must be a string, got {device!r}")
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    kind = device.split(":", 1)[0]
    if kind not in {"cpu", "cuda"} or (kind == "cpu" and device != "cpu"):
        raise ValueError(f"unsupported device {device!r}; use 'cpu', 'cuda', 'cuda:N' or 'auto'")
    return kind


def engine_source_digest() -> str:
    """Digest of the grading/RNG/contract sources receipts are bound to."""
    here = Path(__file__).resolve().parent
    return stable_hash({name: file_hash(here / name) for name in _ENGINE_SOURCES})


def adapter_source_digest(fn: Callable) -> str | None:
    """Digest of the module source defining an adapter (``None`` if unavailable)."""
    try:
        source_file = inspect.getsourcefile(fn)
        if source_file and Path(source_file).is_file():
            return file_hash(source_file)
        return stable_hash(inspect.getsource(fn))
    except (OSError, TypeError):
        return None


def _require_namespace(data: Any, kind: str) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("namespace") != NAMESPACE:
        raise ValueError(
            f"refusing to load {kind}: not a {NAMESPACE} declaration "
            "(missing namespace marker; experiments/forge workspace files are not supported)"
        )
    return data


class AttemptTimeout(BaseException):
    """Raised inside an adapter when the task's ``timeout_seconds`` elapses.

    Derives from BaseException so adapters catching ``Exception`` cannot
    swallow the deadline.
    """


@contextmanager
def _deadline(seconds: float):
    """Arm SIGALRM for ``seconds`` when possible; yield whether it was armed."""
    armed = (
        hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
        and signal.getitimer(signal.ITIMER_REAL)[0] == 0
    )
    if not armed:
        yield False
        return

    def _expire(signum, frame):
        raise AttemptTimeout(f"attempt exceeded timeout_seconds={seconds}")

    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

OPS = {
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
}


@dataclass(frozen=True)
class MetricBound:
    """Numerical pass/fail bound on a task metric."""
    metric: str
    op: str
    bound: float

    def __post_init__(self):
        identifier(self.metric, "metric")
        if self.op not in OPS:
            raise ValueError(f"unsupported metric operator: {self.op!r}")
        if isinstance(self.bound, bool) or not isinstance(self.bound, (int, float)) or not math.isfinite(self.bound):
            raise ValueError("metric bound must be finite")

    def check(self, value: float) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
        return bool(OPS[self.op](float(value), float(self.bound)))

    def to_list(self) -> list:
        return [self.metric, self.op, float(self.bound)]

    @classmethod
    def from_list(cls, entry: list | tuple) -> MetricBound:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError("threshold entry must be [metric, op, bound]")
        return cls(metric=entry[0], op=entry[1], bound=float(entry[2]))


@dataclass
class TaskSpec:
    """Task declaration owning problem, architecture, initial prior, batch_size, budget, and gates."""
    id: str
    description: str
    adapter: str
    steps: int
    batch_size: int = 1
    num_particles: int = 32
    z_dim: int = 8
    prior_kind: str = "mog"
    prior_sigma: float = 0.025
    prior_standardize: bool = False
    thresholds: list[MetricBound] = field(default_factory=list)
    observations: int = 24
    minimum_stable_checks: int = 5
    timeout_seconds: float = 300
    host_definition: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.id, "task id")
        identifier(self.adapter, "adapter")
        if self.steps < 1 or self.batch_size < 1 or self.num_particles < 1 or self.z_dim < 1:
            raise ValueError("task resource counts must be positive integers")
        if self.prior_kind not in {"mog", "particle_cloud"}:
            raise ValueError("prior_kind must be 'mog' or 'particle_cloud'")
        if self.prior_kind == "mog" and self.prior_sigma <= 0:
            raise ValueError("mog prior requires positive sigma")
        if not self.thresholds:
            raise ValueError(f"task {self.id!r} must declare at least one numerical pass/fail threshold")
        if not all(isinstance(b, MetricBound) for b in self.thresholds):
            raise ValueError("task thresholds must be MetricBound instances")
        if self.observations < 1 or self.minimum_stable_checks < 1:
            raise ValueError("observations and minimum_stable_checks must be positive")
        if (isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be a positive finite number")

    def observation_steps(self) -> list[int]:
        if self.steps <= self.observations:
            return list(range(1, self.steps + 1))
        return sorted({max(1, round(i * self.steps / self.observations)) for i in range(1, self.observations + 1)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "namespace": NAMESPACE,
            "id": self.id,
            "description": self.description,
            "adapter": self.adapter,
            "execution": {
                "initializer": "deterministic_orthogonal",
                "steps": self.steps,
                "prior_contract": "recipe_owned_v1",
                "prior": {
                    "kind": self.prior_kind,
                    "sigma": self.prior_sigma,
                    "standardize": self.prior_standardize,
                },
                "resources": {
                    "batch_size": self.batch_size,
                    "num_particles": self.num_particles,
                    "z_dim": self.z_dim,
                },
                "host_definition": dict(self.host_definition),
            },
            "evaluation": {
                "sampling_contract_version": 1,
                "sampling_law": "enumerated_prior_clean",
                "eval_output_noise": "clean",
                "kind": "sustained_window",
                "observations": self.observations,
                "minimum_stable_checks": self.minimum_stable_checks,
                "scoring_weights": "live",
                "thresholds": [b.to_list() for b in self.thresholds],
            },
            "resources": {
                "gpus": 1,
                "cpu_threads": 1,
                "timeout_seconds": self.timeout_seconds,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        _require_namespace(data, "task")
        exec_block = data["execution"]
        res = exec_block["resources"]
        prior = exec_block["prior"]
        eval_block = data["evaluation"]
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            adapter=data["adapter"],
            steps=int(exec_block["steps"]),
            batch_size=int(res["batch_size"]),
            num_particles=int(res["num_particles"]),
            z_dim=int(res["z_dim"]),
            prior_kind=prior["kind"],
            prior_sigma=float(prior["sigma"]),
            prior_standardize=bool(prior.get("standardize", False)),
            thresholds=[MetricBound.from_list(x) for x in eval_block.get("thresholds", [])],
            observations=int(eval_block.get("observations", 24)),
            minimum_stable_checks=int(eval_block.get("minimum_stable_checks", 5)),
            timeout_seconds=_load_timeout(data.get("resources", {}).get("timeout_seconds", 300)),
            host_definition=dict(exec_block.get("host_definition", {})),
        )


@dataclass
class TierAssignment:
    """Assignment of a task to a qualification tier with role and hold/removal notes."""
    task: str
    qualification_tier: int = 1
    importance: str = "required"
    order: int = 0
    hold: dict[str, Any] | None = None
    removal: dict[str, Any] | None = None

    def __post_init__(self):
        identifier(self.task, "task")
        if self.qualification_tier < 1:
            raise ValueError("qualification_tier must be >= 1")
        if self.importance not in IMPORTANCES:
            raise ValueError(f"invalid importance: {self.importance!r}")
        if self.importance == "held":
            if not isinstance(self.hold, dict) or self.hold.get("reason") not in HOLD_REASONS:
                raise ValueError("held task requires hold metadata with valid reason")
            for req in ("summary", "explanation", "reentry"):
                if not isinstance(self.hold.get(req), str) or len(self.hold[req].strip()) < 10:
                    raise ValueError(f"held task requires nonempty {req}")
        if self.removal is not None:
            if self.importance != "diagnostic" or self.removal.get("reason") not in REMOVAL_REASONS:
                raise ValueError("removed task must have importance='diagnostic' and valid removal reason")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "task": self.task,
            "qualification_tier": self.qualification_tier,
            "importance": self.importance,
            "order": self.order,
        }
        if self.hold is not None:
            out["hold"] = dict(self.hold)
        if self.removal is not None:
            out["removal"] = dict(self.removal)
        return out


@dataclass
class TierSpec:
    """Contiguous tier grouping in a TierView."""
    tier: int
    required: list[str] = field(default_factory=list)
    held: dict[str, dict[str, Any]] = field(default_factory=dict)
    diagnostic: list[str] = field(default_factory=list)
    ranking: list[str] = field(default_factory=list)


@dataclass
class TierView:
    """Versioned multi-tier view governing qualification and leaderboard totals."""
    id: str
    goal: str
    revision: int = 1
    assignments: list[TierAssignment] = field(default_factory=list)
    evidence_scope: str = "ordinary"
    policy_change_reason: str = "Initial tier view declaration."
    previous_revision: dict[str, Any] | None = None

    def __post_init__(self):
        identifier(self.id, "view id")
        identifier(self.goal, "goal")
        if self.revision < 1:
            raise ValueError("revision must be >= 1")
        seen = set()
        for a in self.assignments:
            if a.task in seen:
                raise ValueError(f"duplicate task in view {self.id}: {a.task}")
            seen.add(a.task)

    @classmethod
    def from_tiers(
        cls,
        *,
        id: str,
        goal: str,
        tiers: list[TierSpec],
        revision: int = 1,
        evidence_scope: str = "ordinary",
        policy_change_reason: str = "Initial tier view declaration.",
    ) -> TierView:
        tier_nums = sorted(t.tier for t in tiers)
        if tier_nums != list(range(1, len(tiers) + 1)):
            raise ValueError(f"tiers must be contiguous 1..N, got {tier_nums}")
        assignments: list[TierAssignment] = []
        order = 0
        for t in sorted(tiers, key=lambda x: x.tier):
            for task_id in t.required:
                assignments.append(TierAssignment(task=task_id, qualification_tier=t.tier, importance="required", order=order))
                order += 1
            for task_id, hold_meta in t.held.items():
                assignments.append(
                    TierAssignment(task=task_id, qualification_tier=t.tier, importance="held", order=order, hold=hold_meta)
                )
                order += 1
            for task_id in t.ranking:
                assignments.append(TierAssignment(task=task_id, qualification_tier=t.tier, importance="ranking", order=order))
                order += 1
            for task_id in t.diagnostic:
                assignments.append(TierAssignment(task=task_id, qualification_tier=t.tier, importance="diagnostic", order=order))
                order += 1
        return cls(
            id=id,
            goal=goal,
            revision=revision,
            assignments=assignments,
            evidence_scope=evidence_scope,
            policy_change_reason=policy_change_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": 1,
            "namespace": NAMESPACE,
            "id": self.id,
            "goal": self.goal,
            "revision": self.revision,
            "evidence_scope": self.evidence_scope,
            "policy_change_reason": self.policy_change_reason,
            "assignments": [a.to_dict() for a in sorted(self.assignments, key=lambda x: (x.qualification_tier, x.order))],
        }
        if self.previous_revision is not None:
            out["previous_revision"] = dict(self.previous_revision)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TierView:
        _require_namespace(data, "view")
        return cls(
            id=data["id"],
            goal=data["goal"],
            revision=int(data["revision"]),
            evidence_scope=data.get("evidence_scope", "ordinary"),
            policy_change_reason=data.get("policy_change_reason", ""),
            previous_revision=data.get("previous_revision"),
            assignments=[
                TierAssignment(
                    task=a["task"],
                    qualification_tier=int(a["qualification_tier"]),
                    importance=a["importance"],
                    order=int(a.get("order", idx)),
                    hold=a.get("hold"),
                    removal=a.get("removal"),
                )
                for idx, a in enumerate(data.get("assignments", []))
            ],
        )


@dataclass
class CandidateSpec:
    """Candidate recipe owning technique, losses, optimizer, prior update/reg, and memory policy."""
    id: str
    technique: str
    changed_factors: list[str]
    mechanism_class: str = "structural"
    parent: str | None = None
    recipe_preset: str = "bcap"
    recipe_overrides: dict[str, Any] = field(default_factory=dict)
    memory_policy: dict[str, Any] = field(default_factory=dict)

    FORBIDDEN_TASK_FIELDS = frozenset({"batch_size", "num_particles", "z_dim", "total_steps", "steps", "prior"})

    def __post_init__(self):
        identifier(self.id, "candidate id")
        identifier(self.technique, "technique")
        if self.mechanism_class not in {"structural", "floor_constant", "sampling_only_patch"}:
            raise ValueError(f"invalid mechanism_class: {self.mechanism_class!r}")
        overlap = set(self.recipe_overrides) & self.FORBIDDEN_TASK_FIELDS
        if overlap:
            raise ValueError(f"candidate recipe_overrides cannot override task-owned fields: {sorted(overlap)}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": 3,
            "namespace": NAMESPACE,
            "id": self.id,
            "technique": self.technique,
            "changed_factors": list(self.changed_factors),
            "mechanism_class": self.mechanism_class,
            "recipe_preset": self.recipe_preset,
            "recipe_overrides": dict(self.recipe_overrides),
            "memory_policy": dict(self.memory_policy),
            "claim_contract": {"sampling_law": "task_declared"},
        }
        if self.parent:
            out["parent"] = self.parent
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateSpec:
        _require_namespace(data, "candidate")
        return cls(
            id=data["id"],
            technique=data.get("technique", data["id"]),
            changed_factors=list(data.get("changed_factors", ["baseline"])),
            mechanism_class=data.get("mechanism_class", "structural"),
            parent=data.get("parent"),
            recipe_preset=data.get("recipe_preset", "bcap"),
            recipe_overrides=dict(data.get("recipe_overrides", {})),
            memory_policy=dict(data.get("memory_policy", {})),
        )


def _as_count(value: Any) -> int | None:
    """Return ``value`` as a non-negative int, or ``None`` when it is not a count.

    Accepts any ``numbers.Integral`` (e.g. ``numpy.int64``) and integral-valued
    finite reals (``4.0``); rejects ``bool``, fractional, negative and
    non-numeric values.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        count = int(value)
    elif isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value).is_integer():
        count = int(float(value))
    else:
        return None
    return count if count >= 0 else None


def grade_sustained_observations(
    task: TaskSpec,
    observations: list[dict[str, Any]],
    *,
    all_finite: bool | None = None,
    unintended_rng_deviations: int | None = None,
    completed_steps: int | None = None,
) -> dict[str, Any]:
    """Grade observations against a TaskSpec's sustained terminal window contract.

    ``all_finite``, ``unintended_rng_deviations`` and ``completed_steps`` are
    evidence the adapter must report explicitly; a missing (``None``) or
    ill-typed value grades INVALID and is never defaulted to a passing value.
    """
    missing = []
    if not isinstance(all_finite, bool):
        missing.append("all_finite")
    unintended_rng_deviations = _as_count(unintended_rng_deviations)
    if unintended_rng_deviations is None:
        missing.append("unintended_rng_deviations")
    completed_steps = _as_count(completed_steps)
    if completed_steps is None:
        missing.append("completed_steps")
    if not isinstance(observations, list) or not all(isinstance(row, dict) for row in observations):
        missing.append("observations")
    if missing:
        return {
            "status": "INVALID",
            "reason": "adapter did not explicitly report valid " + ", ".join(missing),
            "terminal_passing": 0,
            "failed_bounds": [f"reported {name}" for name in missing],
        }
    if unintended_rng_deviations > 0:
        return {
            "status": "INVALID",
            "reason": f"evaluation mutated {unintended_rng_deviations} non-eval RNG stream(s)",
            "terminal_passing": 0,
            "failed_bounds": ["unintended_rng_deviations == 0"],
        }
    if not all_finite:
        return {
            "status": "FAIL",
            "reason": "nonfinite parameters, gradients, or metrics observed",
            "terminal_passing": 0,
            "failed_bounds": ["all_finite == 1"],
        }
    if completed_steps < task.steps:
        return {
            "status": "INCOMPLETE",
            "reason": f"completed {completed_steps}/{task.steps} steps",
            "terminal_passing": 0,
            "failed_bounds": ["completed_steps == steps"],
        }
    if len(observations) < task.minimum_stable_checks:
        return {
            "status": "INCOMPLETE",
            "reason": f"only {len(observations)} observations recorded (need >= {task.minimum_stable_checks})",
            "terminal_passing": 0,
            "failed_bounds": ["observations >= minimum_stable_checks"],
        }

    per_step = []
    for row in observations:
        metrics = row.get("metrics", row)
        failed = [f"{b.metric} {b.op} {b.bound}" for b in task.thresholds if not b.check(metrics.get(b.metric, float("nan")))]
        per_step.append({"step": row.get("step", 0), "passed": len(failed) == 0, "failed_bounds": failed})

    window = per_step[-task.minimum_stable_checks :]
    terminal_passing = sum(1 for item in window if item["passed"])
    all_window_pass = terminal_passing == task.minimum_stable_checks
    failed_bounds = sorted({fb for item in window for fb in item["failed_bounds"]})
    return {
        "status": "PASS" if all_window_pass else "FAIL",
        "terminal_passing": terminal_passing,
        "required_terminal_checks": task.minimum_stable_checks,
        "failed_bounds": failed_bounds,
        "live": observations[-1].get("metrics", observations[-1]) if observations else {},
    }


class ForgeRegistry:
    """Plugin registry mapping adapter names to executable runners."""

    def __init__(self):
        self._adapters: dict[str, Callable] = {}

    def adapter(self, name: str):
        identifier(name, "adapter")

        def register(fn: Callable):
            self._adapters[name] = fn
            return fn

        return register

    def get_adapter(self, name: str) -> Callable:
        if name not in self._adapters:
            raise KeyError(f"unregistered task adapter: {name!r}")
        return self._adapters[name]


def _accepts_kwarg(fn: Callable, name: str) -> bool:
    """True only when ``fn`` names ``name`` as a keyword-capable parameter.

    A bare ``**kwargs`` does not count: adapters that forward ``**kwargs`` to a
    fixed-signature function must not receive optional extras they never asked for.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    param = params.get(name)
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _load_timeout(value: Any) -> float:
    """Load ``timeout_seconds`` without truncation (ints stay ints, 0.5 stays 0.5)."""
    if isinstance(value, str):
        value = float(value)
    return value


def _tier_status(required_statuses: list[str]) -> str:
    """Counted-tier status: PASS only when every required task passes.

    Otherwise the most severe required status wins (FAIL > INVALID > BLOCKED >
    INCOMPLETE > NOT_RUN), matching ``experiments/forge/views.py``: NOT_RUN is
    kept at tier level and only becomes INCOMPLETE in the overall status.
    """
    if all(status == "PASS" for status in required_statuses):
        return "PASS"
    return next((s for s in _TIER_STATUS_PRECEDENCE if s in required_statuses), "INCOMPLETE")


class ForgeWorkspace:
    """Manages tasks, views, candidates, tiered execution, retiering, and leaderboards.

    All declarations live under ``configs/particlegan_forge/`` and all
    generated reports/receipts under ``reports/particlegan_forge/``; the
    ``experiments/forge`` namespaces (``configs/forge``, ``reports/forge``)
    are never read or written.
    """

    def __init__(self, root: Path | str, *, registry: ForgeRegistry | None = None, queue_root: Path | str | None = None):
        self.root = Path(root).resolve()
        self.registry = registry or ForgeRegistry()
        self.queue_root = Path(queue_root).resolve() if queue_root else (self.root / "runs" / NAMESPACE)

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs" / NAMESPACE

    @property
    def tasks_dir(self) -> Path:
        return self.configs_dir / "tasks"

    @property
    def views_dir(self) -> Path:
        return self.configs_dir / "views"

    @property
    def view_history_dir(self) -> Path:
        return self.configs_dir / "view-history"

    @property
    def ideas_dir(self) -> Path:
        return self.configs_dir / "ideas"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports" / NAMESPACE

    def register_task(self, task: TaskSpec) -> Path:
        path = self.tasks_dir / f"{task.id}.json"
        atomic_json(path, task.to_dict())
        return path

    def register_view(self, view: TierView) -> Path:
        path = self.views_dir / f"{view.id}.json"
        atomic_json(path, view.to_dict())
        return path

    def register_candidate(self, candidate: CandidateSpec) -> Path:
        path = self.ideas_dir / f"{candidate.id}.json"
        atomic_json(path, candidate.to_dict())
        return path

    @staticmethod
    def _load_dir(directory: Path, loader: Callable) -> dict:
        if not directory.is_dir():
            return {}
        out = {}
        for path in sorted(directory.glob("*.json")):
            try:
                out[path.stem] = loader(read_json(path))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: {exc}") from exc
        return out

    def load_tasks(self) -> dict[str, TaskSpec]:
        return self._load_dir(self.tasks_dir, TaskSpec.from_dict)

    def load_views(self) -> dict[str, TierView]:
        return self._load_dir(self.views_dir, TierView.from_dict)

    def load_candidates(self) -> dict[str, CandidateSpec]:
        return self._load_dir(self.ideas_dir, CandidateSpec.from_dict)

    def validate(self) -> dict[str, Any]:
        """Validate this namespace's tasks, views and candidates (never experiments/forge)."""
        tasks = self.load_tasks()
        views = self.load_views()
        candidates = self.load_candidates()
        for view in views.values():
            for a in view.assignments:
                if a.task not in tasks:
                    raise ValueError(f"view {view.id} references unknown task {a.task!r}")
        return {
            "status": "valid",
            "namespace": NAMESPACE,
            "tasks": len(tasks),
            "views": sorted(views.keys()),
            "candidates": len(candidates),
            "training_launched": False,
        }

    def retier(
        self,
        view_id: str,
        *,
        task_id: str,
        new_tier: int,
        importance: str | None = None,
        hold: dict[str, Any] | None = None,
        reason: str,
    ) -> TierView:
        """Archive the current view revision byte-for-byte and write the updated revision."""
        view_path = self.views_dir / f"{view_id}.json"
        if not view_path.is_file():
            raise FileNotFoundError(f"view {view_id!r} not found at {view_path}")
        raw_bytes = view_path.read_bytes()
        current_dict = read_json(view_path)
        current_view = TierView.from_dict(current_dict)

        if not isinstance(new_tier, int) or new_tier < 1:
            raise ValueError(f"qualification_tier must be an integer >= 1, got {new_tier!r}")

        updated_assignments = []
        found = False
        for a in current_view.assignments:
            if a.task == task_id:
                found = True
                updated_assignments.append(
                    TierAssignment(
                        task=a.task,
                        qualification_tier=new_tier,
                        importance=importance or a.importance,
                        order=a.order,
                        hold=hold if (importance or a.importance) == "held" else None,
                        removal=a.removal,
                    )
                )
            else:
                updated_assignments.append(a)
        if not found:
            raise KeyError(f"task {task_id!r} not in view {view_id!r}")

        archive_path = self.view_history_dir / f"{view_id}-v{current_view.revision}.json"
        archive_rel = archive_path.relative_to(self.root).as_posix()
        if archive_path.exists() and archive_path.read_bytes() != raw_bytes:
            raise ValueError(f"conflicting view archive already exists at {archive_rel}")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(raw_bytes)

        next_view = TierView(
            id=current_view.id,
            goal=current_view.goal,
            revision=current_view.revision + 1,
            evidence_scope=current_view.evidence_scope,
            policy_change_reason=reason,
            previous_revision={
                "revision": current_view.revision,
                "archive": archive_rel,
                "policy_fingerprint": stable_hash(current_dict),
            },
            assignments=updated_assignments,
        )
        self.register_view(next_view)
        return next_view


    def _receipt_path(self, candidate_id: str, task_id: str, device: str | None = None) -> Path:
        return self.reports_dir / "receipts" / resolve_device(device) / candidate_id / f"{task_id}.json"

    def _binding(self, task: TaskSpec, candidate: CandidateSpec, device: str, seed: int,
                 adapter_source: str | None) -> dict[str, Any]:
        """Receipt cohort binding: what a receipt must match to count as current evidence."""
        return {
            "namespace": NAMESPACE,
            "task_fingerprint": stable_hash(task.to_dict()),
            "candidate_fingerprint": stable_hash(candidate.to_dict()),
            "rng_version": RNG_VERSION,
            "engine_source_sha256": engine_source_digest(),
            "adapter_source_sha256": adapter_source,
            "device": device,
            "seed": seed,
        }

    def run_task(
        self,
        candidate_id: str,
        task_id: str,
        *,
        device: str = "cpu",
        seed: int = 0,
        steps_override: int | None = None,
    ) -> dict[str, Any]:
        """Execute one (candidate, task) pair under protocol seed 0 and isolated NamedStreams.

        Adapters are called with ``task, candidate, streams, attempt_dir,
        workspace_root, device, steps_override, log_event`` and, when their
        signature names it explicitly (``**kwargs`` alone does not opt in),
        ``evaluate(step, fn)``, which runs ``fn`` under
        ``streams.isolated_evaluation()``, records ``{"step", "metrics"}`` and
        audits non-eval stream mutation.  The returned dict must report
        ``all_finite``, ``unintended_rng_deviations`` and ``completed_steps``
        explicitly (missing -> INVALID) plus ``observations`` (unless recorded
        through ``evaluate``).  ``timeout_seconds`` is enforced with SIGALRM in
        the main thread and checked against wall time everywhere (timeout ->
        INCOMPLETE).  An adapter exception is recorded as an INVALID receipt,
        replacing any earlier receipt for the same cohort path.
        """
        tasks = self.load_tasks()
        candidates = self.load_candidates()
        task = tasks[task_id]
        candidate = candidates[candidate_id]
        adapter_fn = self.registry.get_adapter(task.adapter)
        cohort_device = resolve_device(device)
        exec_device = cohort_device if device in (None, "auto") else device
        binding = self._binding(task, candidate, cohort_device, seed, adapter_source_digest(adapter_fn))

        attempt_dir = self.queue_root / candidate_id / task_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        log_path = attempt_dir / "run.log"
        events_path = self.queue_root / "events.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)

        def log_event(event_type: str, **payload):
            line = json.dumps(
                json_safe({"ts": utc_now(), "event": event_type, "candidate": candidate_id, "task": task_id, **payload}),
                sort_keys=True,
            )
            with file_lock(self.queue_root / ".events.lock"):
                for path in (events_path, log_path):
                    with path.open("a", encoding="utf-8") as f:
                        f.write(line + "\n")

        streams: NamedStreams | None = None
        recorded: list[dict[str, Any]] = []
        measured_deviations = 0

        def evaluate(step: int, fn: Callable[[], dict]) -> dict:
            nonlocal measured_deviations
            if streams is None:
                raise RuntimeError("evaluate() called outside an attempt")
            with streams.isolated_evaluation():
                metrics = fn()
            measured_deviations += int(streams.last_eval_audit["unintended_rng_deviations"])
            recorded.append({"step": int(step), "metrics": json_safe(dict(metrics))})
            return metrics

        adapter_kwargs = dict(
            task=task,
            candidate=candidate,
            attempt_dir=attempt_dir,
            workspace_root=self.root,
            device=exec_device,
            steps_override=steps_override,
            log_event=log_event,
        )
        if _accepts_kwarg(adapter_fn, "evaluate"):
            adapter_kwargs["evaluate"] = evaluate

        t0 = time.perf_counter()
        log_event("attempt_started", device=exec_device, steps=steps_override or task.steps)
        result: Any = None
        error: str | None = None
        timed_out = False
        try:
            # The deadline is armed inside the deterministic scope and disarmed
            # before the scope's ``finally`` restores process-wide torch flags,
            # so SIGALRM can never interrupt that restoration.
            with deterministic_scope(seed=seed):
                streams = NamedStreams(seed=seed, device=exec_device)
                with _deadline(task.timeout_seconds):
                    result = adapter_fn(streams=streams, **adapter_kwargs)
        except AttemptTimeout as exc:
            timed_out, error = True, str(exc)
        except Exception as exc:  # recorded as INVALID evidence, never silently dropped
            error = f"{type(exc).__name__}: {exc}"
            log_event("attempt_failed", error=error, traceback=traceback.format_exc())
        wall_seconds = round(time.perf_counter() - t0, 4)
        if error is None and wall_seconds > task.timeout_seconds:
            timed_out = True
            error = f"attempt exceeded timeout_seconds={task.timeout_seconds} (wall {wall_seconds}s)"

        reported = result if isinstance(result, dict) else {}
        raw_observations = reported.get("observations", recorded)
        all_finite = reported.get("all_finite")
        if isinstance(all_finite, bool) and has_nonfinite(raw_observations):
            all_finite = False
        unintended_rng_deviations = reported.get("unintended_rng_deviations")
        reported_deviations = _as_count(unintended_rng_deviations)
        if measured_deviations and reported_deviations is not None:
            unintended_rng_deviations = max(reported_deviations, measured_deviations)
        completed_steps = reported.get("completed_steps")

        if timed_out:
            grade = {"status": "INCOMPLETE", "reason": error, "terminal_passing": 0,
                     "failed_bounds": ["wall_seconds <= timeout_seconds"]}
        elif error is not None:
            grade = {"status": "INVALID", "reason": f"adapter raised {error}", "terminal_passing": 0,
                     "failed_bounds": ["adapter completed without raising"]}
        elif not isinstance(result, dict):
            grade = {"status": "INVALID", "reason": "adapter must return a dict of evidence", "terminal_passing": 0,
                     "failed_bounds": ["adapter returned evidence"]}
        else:
            grade = grade_sustained_observations(
                task,
                raw_observations,
                all_finite=all_finite,
                unintended_rng_deviations=unintended_rng_deviations,
                completed_steps=completed_steps,
            )
        now_ts = utc_now()
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "technique": candidate.technique,
            "task_id": task_id,
            "device": cohort_device,
            "execution_device": exec_device,
            "seed": seed,
            "completed_steps": completed_steps,
            "all_finite": all_finite,
            "unintended_rng_deviations": unintended_rng_deviations,
            "terminal_passing": grade.get("terminal_passing", 0),
            "task_fingerprint": binding["task_fingerprint"],
            "candidate_fingerprint": binding["candidate_fingerprint"],
            "binding": binding,
            "cohort_key": stable_hash(binding),
            "rng_manifest": streams.manifest() if streams is not None else None,
            "verdict": grade["status"],
            "grade": grade,
            "live": grade.get("live", {}),
            "observations": raw_observations,
            "artifacts": reported.get("artifacts", {}),
            "error": error,
            "cost": {
                "wall_seconds": wall_seconds,
                "completed_steps": completed_steps,
                "timeout_seconds": task.timeout_seconds,
            },
            "timestamp": now_ts,
            "completed_at": now_ts,
        }
        receipt = json_safe(receipt)
        if steps_override is None or steps_override == task.steps:
            atomic_json(self._receipt_path(candidate_id, task_id, device=cohort_device), receipt)
        atomic_json(attempt_dir / "receipt.json", receipt)
        log_event("attempt_completed", verdict=grade["status"], wall_seconds=wall_seconds)
        return receipt

    def _task_outcome(self, rpath: Path, expected: dict[str, Any], verify_adapter: bool) -> dict[str, Any]:
        if not rpath.is_file():
            return {"status": "NOT_RUN", "reason": "no receipt", "live": {}, "goal_gif": None, "receipt": None}
        rec = read_json(rpath)
        rec = rec if isinstance(rec, dict) else {}
        bound = rec.get("binding") if isinstance(rec.get("binding"), dict) else {}
        keys = [k for k in expected if verify_adapter or k != "adapter_source_sha256"]
        stale = sorted(k for k in keys if bound.get(k) != expected[k])
        if rec.get("cohort_key") != stable_hash(bound):
            stale.append("cohort_key")
        if stale:
            return {"status": "NOT_RUN", "reason": "stale receipt: " + ", ".join(stale) + " differ",
                    "live": {}, "goal_gif": None, "receipt": str(rpath)}
        grade = rec.get("grade") if isinstance(rec.get("grade"), dict) else {}
        return {
            "status": rec.get("verdict", "INVALID"),
            "reason": grade.get("reason"),
            "failed_bounds": grade.get("failed_bounds", []),
            "terminal_passing": grade.get("terminal_passing", 0),
            "required_terminal_checks": grade.get("required_terminal_checks"),
            "live": grade.get("live", {}),
            "goal_gif": (rec.get("artifacts") or {}).get("goal_gif"),
            "wall_seconds": (rec.get("cost") or {}).get("wall_seconds"),
            "receipt": str(rpath),
        }

    def qualify(self, candidate_id: str, view_id: str, *, device: str | None = None, seed: int = 0) -> dict[str, Any]:
        """Compute per-tier and whole-view qualification status from recorded receipts.

        A receipt counts only when its cohort binding (task and candidate
        fingerprints, RNG version, engine and adapter source digests, device
        and seed) matches the current declarations; otherwise the task is
        NOT_RUN.  The adapter digest is verified only when the adapter is
        registered; ``adapter_source_verified`` is False otherwise and the
        candidate is then not ``eligible``.  Tier semantics follow
        ``experiments/forge/views.py``: a tier without required tasks is HELD,
        neither passing nor blocking higher tiers; HELD is never PASS.
        """
        device = resolve_device(device)
        views = self.load_views()
        tasks = self.load_tasks()
        candidates = self.load_candidates()
        if view_id not in views:
            raise KeyError(f"view {view_id!r} not found in workspace")
        if candidate_id not in candidates:
            raise KeyError(f"candidate {candidate_id!r} not found in workspace")
        view = views[view_id]
        candidate = candidates[candidate_id]

        tiers_map: dict[int, list[TierAssignment]] = {}
        for a in view.assignments:
            tiers_map.setdefault(a.qualification_tier, []).append(a)

        tier_summaries: dict[int, dict[str, Any]] = {}
        blocked_by_lower = False
        source_verified = True
        for t_num in sorted(tiers_map.keys()):
            assigns = tiers_map[t_num]
            req = [a for a in assigns if a.importance == "required"]
            held = [a for a in assigns if a.importance == "held"]
            task_outcomes = {}
            for a in assigns:
                task = tasks[a.task]
                try:
                    adapter_digest = adapter_source_digest(self.registry.get_adapter(task.adapter))
                    verify_adapter = True
                except KeyError:
                    adapter_digest, verify_adapter = None, False
                expected = self._binding(task, candidate, device, seed, adapter_digest)
                outcome = self._task_outcome(self._receipt_path(candidate_id, a.task, device), expected, verify_adapter)
                outcome["importance"] = a.importance
                outcome["adapter_source_verified"] = verify_adapter
                if a.importance == "required" and not verify_adapter:
                    source_verified = False
                task_outcomes[a.task] = outcome
            req_statuses = [task_outcomes[a.task]["status"] for a in req]
            req_passed = sum(status == "PASS" for status in req_statuses)

            # Each tier reports its own status (as experiments/forge/views.py
            # does); a tier above the first open counted tier is flagged via
            # ``gated_by_lower_tier`` instead of having its status overwritten.
            gated = blocked_by_lower
            if not req:
                t_status = "HELD"
            else:
                t_status = _tier_status(req_statuses)
                if t_status != "PASS":
                    blocked_by_lower = True

            tier_summaries[t_num] = {
                "tier": t_num,
                "status": t_status,
                "gated_by_lower_tier": gated,
                "required_passed": req_passed,
                "required_total": len(req),
                "held_total": len(held),
                "diagnostic_total": sum(a.importance == "diagnostic" for a in assigns),
                "tasks": task_outcomes,
            }

        counted = [ts for _, ts in sorted(tier_summaries.items()) if ts["status"] != "HELD"]
        if view.evidence_scope == "research_diagnostic":
            overall = "DIAGNOSTIC"
        elif not counted:
            overall = "HELD"
        else:
            first_open = next((ts for ts in counted if ts["status"] != "PASS"), None)
            if first_open is not None:
                overall = "INCOMPLETE" if first_open["status"] == "NOT_RUN" else first_open["status"]
            elif tier_summaries[max(tier_summaries)]["status"] == "HELD":
                overall = "HELD"
            else:
                overall = "PASS"
        qualified_tier = 0
        for ts in counted:
            if ts["status"] != "PASS":
                break
            qualified_tier = ts["tier"]

        return {
            "candidate_id": candidate_id,
            "view_id": view_id,
            "device": device,
            "seed": seed,
            "status": overall,
            "qualified_tier": qualified_tier,
            "adapter_source_verified": source_verified,
            "eligible": overall == "PASS" and source_verified,
            "held_tiers": [ts["tier"] for ts in tier_summaries.values() if ts["status"] == "HELD"],
            "tiers": tier_summaries,
        }

    def run_tier(
        self,
        candidate_id: str,
        view_id: str,
        *,
        through_tier: int = 1,
        device: str = "cpu",
        seed: int = 0,
    ) -> dict[str, Any]:
        """Run all runnable tasks in each tier up to through_tier, stopping higher tiers on required failure."""
        views = self.load_views()
        if view_id not in views:
            raise KeyError(f"view {view_id!r} not found in workspace")
        view = views[view_id]
        tiers_map: dict[int, list[TierAssignment]] = {}
        for a in view.assignments:
            if a.qualification_tier <= through_tier:
                tiers_map.setdefault(a.qualification_tier, []).append(a)

        for t_num in sorted(tiers_map.keys()):
            for a in sorted(tiers_map[t_num], key=lambda x: x.order):
                self.run_task(candidate_id, a.task, device=device, seed=seed)
            q = self.qualify(candidate_id, view_id, device=device, seed=seed)
            if q["tiers"][t_num]["status"] not in {"PASS", "HELD"} and view.evidence_scope != "research_diagnostic":
                break
        return self.qualify(candidate_id, view_id, device=device, seed=seed)

    def compile_reports(self, *, view_id: str | None = None, device: str | None = None, seed: int = 0) -> dict[str, str]:
        """Generate the current leaderboard, EXPERIMENTS_BY_TIER.md and EXPERIMENT_MEMORY.md.

        Everything is derived from declarations and receipts; no narrative is
        hard-coded.  Outputs go to ``reports/particlegan_forge/`` only.
        """
        device = resolve_device(device)
        views = self.load_views()
        tasks = self.load_tasks()
        candidates = self.load_candidates()
        if not views:
            raise ValueError("no views registered in workspace")
        if view_id and view_id not in views:
            raise KeyError(f"view {view_id!r} not found in workspace")
        primary_view = views[view_id] if view_id else next(iter(views.values()))

        def fmt_live(live: dict) -> str:
            return ", ".join(
                f"{k}={float(v):.4f}" for k, v in sorted((live or {}).items())
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            )

        # 1. Single current leaderboard for primary_view
        tier_nums = sorted({a.qualification_tier for a in primary_view.assignments}) or [1]
        header_cols = ["Technique", "Recipe", "Compute"] + [f"Tier {t}" for t in tier_nums] + ["Total", "Status"]
        lines = [
            f"# Leaderboard — `{primary_view.goal}` (`{primary_view.id}` v{primary_view.revision})",
            "",
            f"Generated by `{NAMESPACE}` from receipts under `reports/{NAMESPACE}/receipts/{device}` (seed {seed}).",
            "",
            "| " + " | ".join(header_cols) + " |",
            "| " + " | ".join(["---"] * len(header_cols)) + " |",
        ]

        rows_scored = []
        for cid, cand in sorted(candidates.items()):
            q = self.qualify(cid, primary_view.id, device=device, seed=seed)
            tot_pass = sum(ts["required_passed"] for ts in q["tiers"].values())
            tot_req = sum(ts["required_total"] for ts in q["tiers"].values())
            rows_scored.append((q["qualified_tier"], tot_pass, cid, cand, q, tot_req))

        rows_scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        for _, tot_pass, cid, cand, q, tot_req in rows_scored:
            t_cells = []
            for t in tier_nums:
                ts = q["tiers"].get(t, {"required_passed": 0, "required_total": 0, "status": "NOT_RUN"})
                if ts["status"] == "HELD":
                    t_cells.append("`HELD`")
                    continue
                suffix = "(*)" if ts["status"] in {"INCOMPLETE", "NOT_RUN", "BLOCKED"} else ""
                t_cells.append(f"`{ts['required_passed']}{suffix}/{ts['required_total']}`")
            status = f"`{q['status']}`" + ("" if q["adapter_source_verified"] else " (adapter source unverified)")
            row = [f"**{cand.technique}**", f"`{cid}`", f"`{device}`", *t_cells, f"**`{tot_pass}/{tot_req}`**", status]
            lines.append("| " + " | ".join(row) + " |")

        lines.extend(["", "## Per-Task Breakdown", ""])
        for a in sorted(primary_view.assignments, key=lambda x: (x.qualification_tier, x.order)):
            t_spec = tasks[a.task]
            lines.append(f"### Tier {a.qualification_tier}: `{a.task}` ({a.importance})")
            lines.append(f"- **Goal**: {t_spec.description}")
            lines.append("- **Gates**: " + ", ".join(f"`{b.metric} {b.op} {b.bound}`" for b in t_spec.thresholds))
            for _, _, cid, _, q, _ in rows_scored:
                tout = q["tiers"][a.qualification_tier]["tasks"][a.task]
                gif_link = f" · [goal.gif]({tout['goal_gif']})" if tout.get("goal_gif") else ""
                lines.append(f"  - `{cid}`: **{tout['status']}** ({fmt_live(tout['live'])}){gif_link}")
            lines.append("")

        leaderboard_path = self.reports_dir / "leaderboard.md"
        atomic_text(leaderboard_path, "\n".join(lines) + "\n")

        # 2. EXPERIMENTS_BY_TIER.md: declared task contracts, one row per assignment.
        ebt_lines = [
            "# Experiments by Tier",
            "",
            "| View | Tier | Task | Role | Batch | Particles | Steps | Prior | Gates |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for vid, v in sorted(views.items()):
            for a in sorted(v.assignments, key=lambda x: (x.qualification_tier, x.order)):
                t = tasks[a.task]
                gates_s = ", ".join(f"`{b.metric}{b.op}{b.bound}`" for b in t.thresholds)
                ebt_lines.append(
                    f"| `{vid}` | {a.qualification_tier} | `{t.id}` | `{a.importance}` | {t.batch_size} | "
                    f"{t.num_particles} | {t.steps} | `{t.prior_kind}` (σ={t.prior_sigma}) | {gates_s} |"
                )
        ebt_lines.extend([
            "",
            "## Declared Evaluation Contracts",
            "",
            "| Task | Initializer | Sampling law | Observation steps | Sustained window | Timeout (s) |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for tid, t in sorted(tasks.items()):
            spec = t.to_dict()
            obs = t.observation_steps()
            window = obs[-t.minimum_stable_checks:]
            ebt_lines.append(
                f"| `{tid}` | `{spec['execution']['initializer']}` | `{spec['evaluation']['sampling_law']}` | "
                f"{len(obs)} ({obs[0]}..{obs[-1]}) | last {t.minimum_stable_checks}: {window} | {t.timeout_seconds} |"
            )
        ebt_path = self.reports_dir / "EXPERIMENTS_BY_TIER.md"
        atomic_text(ebt_path, "\n".join(ebt_lines) + "\n")

        # 3. EXPERIMENT_MEMORY.md: per-candidate receipt digest, no prose.
        mem_lines = [
            "# Experiment Memory",
            "",
            f"- **Primary view**: [`{primary_view.id}`](leaderboard.md) (revision {primary_view.revision})",
            "- **Tasks by tier**: [`EXPERIMENTS_BY_TIER.md`](EXPERIMENTS_BY_TIER.md)",
            f"- **Cohort**: device `{device}`, seed {seed}",
            "",
            "## Recorded Candidates",
            "",
        ]
        for _, tot_pass, cid, cand, q, tot_req in rows_scored:
            mem_lines.append(f"### `{cid}` (`{cand.technique}`)")
            mem_lines.append(f"- **Changed factors**: {', '.join(cand.changed_factors)}")
            if cand.parent:
                mem_lines.append(f"- **Parent**: `{cand.parent}`")
            mem_lines.append(
                f"- **Status**: `{q['status']}` (`{tot_pass}/{tot_req}` required passes, "
                f"qualified tier {q['qualified_tier']})"
            )
            for t_num, ts in sorted(q["tiers"].items()):
                for tid, tout in ts["tasks"].items():
                    detail = [f"terminal {tout.get('terminal_passing', 0)}/{tout.get('required_terminal_checks') or '-'}"]
                    if tout.get("failed_bounds"):
                        detail.append("failed " + "; ".join(f"`{fb}`" for fb in tout["failed_bounds"]))
                    if tout.get("reason") and tout["status"] != "PASS":
                        detail.append(f"reason: {tout['reason']}")
                    live = fmt_live(tout["live"])
                    if live:
                        detail.append(live)
                    mem_lines.append(
                        f"  - Tier {t_num} `{tid}` ({tout['importance']}): **{tout['status']}** — " + " · ".join(detail)
                    )
            mem_lines.append("")
        mem_path = self.reports_dir / "EXPERIMENT_MEMORY.md"
        atomic_text(mem_path, "\n".join(mem_lines) + "\n")

        return {
            "leaderboard": str(leaderboard_path),
            "experiments_by_tier": str(ebt_path),
            "experiment_memory": str(mem_path),
        }
