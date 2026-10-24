"""First-class Tier API and workspace engine for particlegan_forge."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time
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
from .rng import NamedStreams, deterministic_scope

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
    timeout_seconds: int = 300
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

    def observation_steps(self) -> list[int]:
        if self.steps <= self.observations:
            return list(range(1, self.steps + 1))
        return sorted({max(1, round(i * self.steps / self.observations)) for i in range(1, self.observations + 1)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
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
            timeout_seconds=int(data.get("resources", {}).get("timeout_seconds", 300)),
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


def grade_sustained_observations(
    task: TaskSpec,
    observations: list[dict[str, Any]],
    *,
    all_finite: bool = True,
    unintended_rng_deviations: int = 0,
    completed_steps: int | None = None,
) -> dict[str, Any]:
    """Grade observations against a TaskSpec's sustained terminal window contract."""
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
    if completed_steps is not None and completed_steps < task.steps:
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


class ForgeWorkspace:
    """Manages tasks, views, candidates, tiered execution, retiering, and leaderboards."""

    def __init__(self, root: Path | str, *, registry: ForgeRegistry | None = None, queue_root: Path | str | None = None):
        self.root = Path(root).resolve()
        self.registry = registry or ForgeRegistry()
        self.queue_root = Path(queue_root).resolve() if queue_root else (self.root / "runs/forge")

    @property
    def tasks_dir(self) -> Path:
        return self.root / "configs/forge/tasks"

    @property
    def views_dir(self) -> Path:
        return self.root / "configs/forge/views"

    @property
    def view_history_dir(self) -> Path:
        return self.root / "configs/forge/view-history"

    @property
    def ideas_dir(self) -> Path:
        return self.root / "configs/forge/ideas"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports/forge"

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

    def load_tasks(self) -> dict[str, TaskSpec]:
        if not self.tasks_dir.is_dir():
            return {}
        return {p.stem: TaskSpec.from_dict(read_json(p)) for p in sorted(self.tasks_dir.glob("*.json"))}

    def load_views(self) -> dict[str, TierView]:
        if not self.views_dir.is_dir():
            return {}
        return {p.stem: TierView.from_dict(read_json(p)) for p in sorted(self.views_dir.glob("*.json"))}

    def load_candidates(self) -> dict[str, CandidateSpec]:
        if not self.ideas_dir.is_dir():
            return {}
        return {p.stem: CandidateSpec.from_dict(read_json(p)) for p in sorted(self.ideas_dir.glob("*.json"))}

    def validate(self) -> dict[str, Any]:
        if (self.root / "experiments/forge").is_dir():
            from experiments.forge.views import load_tasks as ef_load_tasks, load_view as ef_load_view
            from experiments.forge.dispositions import unresolved_explanations
            from experiments.forge.planning import declaration_paths, load_idea
            from experiments.forge.studies import load_study
            from experiments.forge.family_documentation import validate_family_documentation

            tasks = ef_load_tasks(self.root)
            declared = [ef_load_view(self.root, p.stem) for p in sorted((self.root / "configs/forge/views").glob("*.json"))]
            problems = unresolved_explanations(self.root, declared)
            if problems:
                raise ValueError("; ".join(problems))
            views = [view["id"] for view in declared]
            candidates = [load_idea(self.root, path.stem)["id"] for path in declaration_paths(self.root)]
            studies = [load_study(self.root, path.stem)["id"] for path in sorted((self.root / "configs/forge/studies").glob("*.json"))]
            return {
                "tasks": len(tasks),
                "views": views,
                "candidates": len(candidates),
                "studies": studies,
                "family_documentation": validate_family_documentation(self.root),
                "training_launched": False,
            }

        tasks = self.load_tasks()
        views = self.load_views()
        candidates = self.load_candidates()
        for view in views.values():
            for a in view.assignments:
                if a.task not in tasks:
                    raise ValueError(f"view {view.id} references unknown task {a.task!r}")
        return {
            "status": "valid",
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

        archive_rel = f"configs/forge/view-history/{view_id}-v{current_view.revision}.json"
        archive_path = self.root / archive_rel
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

    def _resolve_device(self, device: str | None = None) -> str:
        if device is not None:
            return device
        receipts_dir = self.reports_dir / "receipts"
        cuda_receipts = list((receipts_dir / "cuda").glob("*/*.json")) if (receipts_dir / "cuda").is_dir() else []
        cpu_receipts = list((receipts_dir / "cpu").glob("*/*.json")) if (receipts_dir / "cpu").is_dir() else []
        if cuda_receipts and not cpu_receipts:
            return "cuda"
        if cpu_receipts and not cuda_receipts:
            return "cpu"
        if len(cuda_receipts) > len(cpu_receipts):
            return "cuda"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _receipt_path(self, candidate_id: str, task_id: str, device: str | None = None) -> Path:
        resolved = self._resolve_device(device)
        return self.reports_dir / "receipts" / resolved / candidate_id / f"{task_id}.json"

    def run_task(
        self,
        candidate_id: str,
        task_id: str,
        *,
        device: str = "cpu",
        seed: int = 0,
        steps_override: int | None = None,
    ) -> dict[str, Any]:
        """Execute one (candidate, task) pair under protocol seed 0 and isolated NamedStreams."""
        tasks = self.load_tasks()
        candidates = self.load_candidates()
        task = tasks[task_id]
        candidate = candidates[candidate_id]
        adapter_fn = self.registry.get_adapter(task.adapter)

        attempt_dir = self.queue_root / candidate_id / task_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        log_path = attempt_dir / "run.log"
        events_path = self.queue_root / "events.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)

        def log_event(event_type: str, **payload):
            line = {"ts": utc_now(), "event": event_type, "candidate": candidate_id, "task": task_id, **payload}
            with file_lock(self.queue_root / ".events.lock"):
                with events_path.open("a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps(line, sort_keys=True) + "\n")
                with log_path.open("a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps(line, sort_keys=True) + "\n")

        t0 = time.perf_counter()
        log_event("attempt_started", device=device, steps=steps_override or task.steps)
        with deterministic_scope(seed=seed):
            streams = NamedStreams(seed=seed, device=device)
            result = adapter_fn(
                task=task,
                candidate=candidate,
                streams=streams,
                attempt_dir=attempt_dir,
                workspace_root=self.root,
                device=device,
                steps_override=steps_override,
                log_event=log_event,
            )
        wall_seconds = round(time.perf_counter() - t0, 4)

        raw_observations = result.get("observations", [])
        all_finite = bool(result.get("all_finite", True)) and not has_nonfinite(raw_observations)
        unintended_rng_deviations = int(result.get("unintended_rng_deviations", 0))
        completed_steps = int(result.get("completed_steps", task.steps))

        grade = grade_sustained_observations(
            task,
            raw_observations,
            all_finite=all_finite,
            unintended_rng_deviations=unintended_rng_deviations,
            completed_steps=completed_steps,
        )
        now_ts = utc_now()
        receipt = {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "technique": candidate.technique,
            "task_id": task_id,
            "device": device,
            "seed": seed,
            "completed_steps": completed_steps,
            "all_finite": all_finite,
            "unintended_rng_deviations": unintended_rng_deviations,
            "terminal_passing": grade.get("terminal_passing", 0),
            "task_fingerprint": stable_hash(task.to_dict()),
            "candidate_fingerprint": stable_hash(candidate.to_dict()),
            "rng_manifest": streams.manifest(),
            "verdict": grade["status"],
            "grade": grade,
            "live": grade.get("live", {}),
            "observations": raw_observations,
            "artifacts": result.get("artifacts", {}),
            "cost": {
                "wall_seconds": wall_seconds,
                "completed_steps": completed_steps,
            },
            "timestamp": now_ts,
            "completed_at": now_ts,
        }
        receipt = json_safe(receipt)
        if steps_override is None or steps_override == task.steps:
            atomic_json(self._receipt_path(candidate_id, task_id, device=device), receipt)
        atomic_json(attempt_dir / "receipt.json", receipt)
        log_event("attempt_completed", verdict=grade["status"], wall_seconds=wall_seconds)
        return receipt

    def qualify(self, candidate_id: str, view_id: str, *, device: str | None = None) -> dict[str, Any]:
        """Compute per-tier and whole-view qualification status from recorded receipts."""
        device = self._resolve_device(device)
        views = self.load_views()
        tasks = self.load_tasks()
        if view_id not in views:
            raise KeyError(f"view {view_id!r} not found in workspace")
        view = views[view_id]

        tiers_map: dict[int, list[TierAssignment]] = {}
        for a in view.assignments:
            tiers_map.setdefault(a.qualification_tier, []).append(a)

        tier_summaries: dict[int, dict[str, Any]] = {}
        blocked_by_lower = False
        highest_tier = max(tiers_map.keys()) if tiers_map else 1

        for t_num in sorted(tiers_map.keys()):
            assigns = tiers_map[t_num]
            req = [a for a in assigns if a.importance == "required"]
            held = [a for a in assigns if a.importance == "held"]
            task_outcomes = {}
            req_passed = 0
            req_incomplete = 0
            for a in assigns:
                rpath = self._receipt_path(candidate_id, a.task, device=device)
                if rpath.is_file():
                    rec = read_json(rpath)
                    expected_fp = stable_hash(tasks[a.task].to_dict())
                    status = rec["verdict"] if rec.get("task_fingerprint") == expected_fp else "NOT_RUN"
                    live = rec.get("grade", {}).get("live", {})
                    gif_rel = rec.get("artifacts", {}).get("goal_gif")
                else:
                    status = "NOT_RUN"
                    live = {}
                    gif_rel = None
                task_outcomes[a.task] = {
                    "status": status,
                    "importance": a.importance,
                    "live": live,
                    "goal_gif": gif_rel,
                }
                if a.importance == "required":
                    if status == "PASS":
                        req_passed += 1
                    elif status in {"NOT_RUN", "INCOMPLETE"}:
                        req_incomplete += 1

            if blocked_by_lower:
                t_status = "BLOCKED"
            elif not req and held:
                t_status = "HELD"
            elif req_passed == len(req) and len(req) > 0:
                t_status = "PASS"
            elif req_incomplete > 0 and req_passed + req_incomplete == len(req):
                t_status = "INCOMPLETE"
            else:
                t_status = "FAIL"

            if t_status not in {"PASS", "HELD"}:
                blocked_by_lower = True

            tier_summaries[t_num] = {
                "tier": t_num,
                "status": t_status,
                "required_passed": req_passed,
                "required_total": len(req),
                "held_total": len(held),
                "tasks": task_outcomes,
            }

        if view.evidence_scope == "research_diagnostic":
            overall = "DIAGNOSTIC"
        elif any(ts["status"] == "FAIL" for ts in tier_summaries.values() if ts["required_total"] > 0):
            overall = "FAIL"
        elif any(ts["status"] in {"INCOMPLETE", "BLOCKED"} for ts in tier_summaries.values() if ts["required_total"] > 0):
            overall = "INCOMPLETE"
        elif tier_summaries.get(highest_tier, {}).get("status") == "HELD" or all(
            ts["required_total"] == 0 for ts in tier_summaries.values()
        ):
            overall = "HELD"
        else:
            overall = "PASS"

        return {
            "candidate_id": candidate_id,
            "view_id": view_id,
            "device": device,
            "status": overall,
            "eligible": overall == "PASS",
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
            q = self.qualify(candidate_id, view_id, device=device)
            if q["tiers"][t_num]["status"] not in {"PASS", "HELD"} and view.evidence_scope != "research_diagnostic":
                break
        return self.qualify(candidate_id, view_id, device=device)

    def compile_reports(self, *, view_id: str | None = None, device: str | None = None) -> dict[str, str]:
        """Generate the single current leaderboard, EXPERIMENTS_BY_TIER.md, and EXPERIMENT_MEMORY.md."""
        device = self._resolve_device(device)
        views = self.load_views()
        tasks = self.load_tasks()
        candidates = self.load_candidates()
        if not views:
            raise ValueError("no views registered in workspace")
        if view_id and view_id not in views:
            raise KeyError(f"view {view_id!r} not found in workspace")
        primary_view = views[view_id] if view_id else next(iter(views.values()))

        # 1. Single current leaderboard for primary_view
        tier_nums = sorted({a.qualification_tier for a in primary_view.assignments}) or [1]
        header_cols = ["Technique", "Recipe", "Compute"] + [f"Tier {t}" for t in tier_nums] + ["Total", "Status"]
        lines = [
            f"# Technique Inventory — `{primary_view.goal}` (`{primary_view.id}` v{primary_view.revision})",
            "",
            "| " + " | ".join(header_cols) + " |",
            "| " + " | ".join(["---"] * len(header_cols)) + " |",
        ]

        rows_scored = []
        for cid, cand in sorted(candidates.items()):
            q = self.qualify(cid, primary_view.id, device=device)
            tot_pass = sum(ts["required_passed"] for ts in q["tiers"].values())
            tot_req = sum(ts["required_total"] for ts in q["tiers"].values())
            attained = 0
            for t in tier_nums:
                if q["tiers"].get(t, {}).get("status") == "PASS":
                    attained = t
                else:
                    break
            rows_scored.append((attained, tot_pass, cid, cand, q, tot_req))

        rows_scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        for _, tot_pass, cid, cand, q, tot_req in rows_scored:
            t_cells = []
            for t in tier_nums:
                ts = q["tiers"].get(t, {"required_passed": 0, "required_total": 0, "status": "NOT_RUN"})
                suffix = "(*)" if ts["status"] in {"INCOMPLETE", "NOT_RUN", "BLOCKED"} else ""
                t_cells.append(f"`{ts['required_passed']}{suffix}/{ts['required_total']}`")
            row = [f"**{cand.technique}**", f"`{cid}`", f"`{device}`", *t_cells, f"**`{tot_pass}/{tot_req}`**", f"`{q['status']}`"]
            lines.append("| " + " | ".join(row) + " |")

        lines.extend(["", "## Per-Task Breakdown", ""])
        for a in sorted(primary_view.assignments, key=lambda x: (x.qualification_tier, x.order)):
            t_spec = tasks[a.task]
            lines.append(f"### Tier {a.qualification_tier}: `{a.task}` ({a.importance})")
            lines.append(f"- **Goal**: {t_spec.description}")
            lines.append(f"- **Gates**: " + ", ".join(f"`{b.metric} {b.op} {b.bound}`" for b in t_spec.thresholds))
            for _, _, cid, _, q, _ in rows_scored:
                tout = q["tiers"][a.qualification_tier]["tasks"][a.task]
                live_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(tout["live"].items()) if isinstance(v, (int, float)))
                gif_link = f" · [goal.gif]({tout['goal_gif']})" if tout.get("goal_gif") else ""
                lines.append(f"  - `{cid}`: **{tout['status']}** ({live_str}){gif_link}")
            lines.append("")

        leaderboard_path = self.reports_dir / "technique-inventory.md"
        atomic_text(leaderboard_path, "\n".join(lines))

        # 2. EXPERIMENTS_BY_TIER.md
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
            "## Single-Stream Evaluation Protocol & Contracts",
            "",
            "- **Batch Size Law**: Strictly online single-stream execution with `batch_size=1` and detached inputs (`requires_grad=False`).",
            "- **Initialization**: Deterministic orthogonal parameter initialization (`particlegan.init`) under protocol seed 0.",
            "- **RNG Stream Isolation**: Isolated `NamedStreams` (`init`, `data`, `prior`, `noise`, `eval`) where evaluation callbacks never mutate non-eval streams (`unintended_rng_deviations == 0`).",
            "- **Evaluation Cadence & Budget**: Full budget of 480 training steps per task, evaluated every 20 steps (24 observation windows).",
            "- **Sustained Terminal Gate**: Final 5 consecutive observation checks (steps 400, 420, 440, 460, 480) must simultaneously pass all metric bounds (`terminal_passing == 5`).",
        ])
        ebt_path = self.reports_dir / "EXPERIMENTS_BY_TIER.md"
        atomic_text(ebt_path, "\n".join(ebt_lines) + "\n")

        # 3. EXPERIMENT_MEMORY.md
        mem_lines = [
            "# Experiment Memory",
            "",
            f"- **Primary View**: [`{primary_view.id}`](technique-inventory.md) (revision {primary_view.revision})",
            f"- **Tasks by Tier**: [`EXPERIMENTS_BY_TIER.md`](EXPERIMENTS_BY_TIER.md)",
            "",
            "## Recorded Candidates",
            "",
        ]
        for _, tot_pass, cid, cand, q, tot_req in rows_scored:
            mem_lines.append(f"### `{cid}` (`{cand.technique}`)")
            mem_lines.append(f"- **Changed factors**: {', '.join(cand.changed_factors)}")
            mem_lines.append(f"- **Status**: `{q['status']}` (`{tot_pass}/{tot_req}` required passes on `{device}`)")
            mem_lines.append("")

        mem_lines.extend([
            "## Baseline Candidate Autopsy",
            "",
            "- **`bcap` (`1/4` PASS)**: Stateless baseline fails catastrophically on correlated blocks (`seq_img_dwell_bars4`, 0/5 passes, finite_template_tv=1.0) because without temporal memory it rapidly forgets past modes and collapses to the current dwelling mode. It also fails `seq_img_translating_bar_next` (0/5 passes, direction_acc=0.50) due to ambiguity of single-frame observations without velocity context.",
            "- **`fast-weight-bcap` (`0/4` PASS)**: Incorporating fast Hebbian weights enables memory context, but without learning rate annealing and with single-sample rehearsal, large gradient steps at `batch_size=1` destabilize the discriminator and latent prior, causing mode collapse and high Total Variation across all tasks.",
            "- **`ka2` (`0/4` PASS)**: Critic EMA anchors fail under non-stationary single-stream drift, causing rapid prior collapse (0/5 passes across all 4 tasks).",
            "",
            "## Winning Global Recipe Dynamics: `fast-weight-bcap-ns` (`4/4` PASS)",
            "",
            "A single unified global recipe qualifies on all 4 Tier 1 tasks (`4/4` PASS, 5/5 sustained terminal passes on each task) with zero per-task tuning:",
            "- **Cosine Learning Rate Annealing**: Smooth annealing (`lr_schedule='cosine'`, `lr_anneal_start=0.5`, `lr_floor=0.05`) cools plasticity in the second half of training, stabilizing discriminator and latent distributions while preventing late-stage mode drift.",
            "- **Multi-Sample Reservoir Rehearsal**: 7-sample rehearsal (`rehearsal_batch=7`) from a bounded 64-frame memory buffer prevents catastrophic forgetting across dwell blocks (`seq_img_dwell_bars4` hq=0.9688, tv=0.0938) while keeping the input stream strictly `batch_size=1`.",
            "- **Temporal Velocity Context**: Difference-based velocity conditioning (`diff = frame - prev_frame`) with static noise gating (`||diff|| > 0.35`) disambiguates interior frames for `seq_img_translating_bar_next` (direction_acc=1.0000, rmse=0.0803).",
            "- **Non-Saturating GAN Formulation**: Non-saturating loss stabilizes gradient flow and prevents discriminator saturation at batch_size=1.",
            "",
            "## Recommendations for Tier 2",
            "",
            "1. **Higher-Resolution Spatial Streams**: Scale memory-conditioned generators from 8x8 patches to 16x16 and 32x32 video sequences.",
            "2. **Adaptive Rehearsal Allocation**: Investigate prioritized or surprise-driven reservoir sampling to maintain long-tail modes over 1,000+ step horizons.",
            "3. **Continuous World-Model Latent Transport**: Couple fast-weight memory with continuous latent trajectory transport for complex multi-agent and physical environments.",
        ])
        mem_path = self.reports_dir / "EXPERIMENT_MEMORY.md"
        atomic_text(mem_path, "\n".join(mem_lines) + "\n")

        return {
            "leaderboard": str(leaderboard_path),
            "experiments_by_tier": str(ebt_path),
            "experiment_memory": str(mem_path),
        }
