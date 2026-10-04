"""One source-bound Gaussian C6 observation through the existing public runner.

Declaration/source preparation is model-free. ``run_bound_case`` requires the
root supervisor's real import and admitted-resource guards before construction.
It creates no supervisor, queue, retry, sampler override or additional gate.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch

from particlegan import GANTrainer
from experiments.forge.policy_adapters import (
    DIGEST_KIND, PolicyLifecycleAudit, controls_receipt, finite_policy_state,
    typed_state_digest,
)


COHORT = "api_gaussian2d_c6_gpu_v1"
CASE_ID = "api-gaussian2d"
BASE_COMMIT = "fb7acc775b3a1a6184d36b55e035b9da04531492"
OVERRIDES = {"lr": .0053125, "prior_lr_mult": 1.5}
EXTRA_SOURCE_PATHS = ("configs/forge/tasks/ring16_acquisition.json", "reports/toy_audit/catalog.json")
THRESHOLDS = [["sample_count", ">=", 1024], ["mean_error_sigma", "<=", .10],
              ["min_cov_eigen", ">=", .85], ["max_cov_eigen", "<=", 1.15],
              ["radial_ks", "<=", .075], ["max_projection_ks", "<=", .06]]
OWNERS = {"continuous_controller", "stationarity_lr", "row_evidence", "birth_death",
          "learned_output_noise", "selected_averaging", "optimizer_surprise", "reopen_guard"}


def declaration():
    """Resolve existing case/public Recipe only; construct no trainer/model."""
    from particlegan import get_recipe
    from . import api_contract, api_vectors
    case = next(record for record in api_vectors.list_cases() if record["id"] == CASE_ID)
    required = {"id": CASE_ID, "legacy_ids": ["source-family-16"], "kind": "gaussian",
                "default_steps": 1000, "batch_size": 2048, "eval_samples": 4096,
                "particles": 20000, "z_dim": 2, "thresholds": THRESHOLDS,
                "law": {"kind": "normal2d", "mean": [1., 1.], "covariance": [[.04, 0.], [0., .04]]},
                "profile": {"kind": "batch_distance", "width": 96, "layers": 3,
                            "scales": [.1, .25, .5, 1.], "init_std": 1.}}
    if any(case.get(key) != value for key, value in required.items()):
        raise ValueError("existing Gaussian question/gates/resources changed")
    if api_contract.metric_observations(case) != 24 or case.get("terminal_observations", 5) != 5:
        raise ValueError("existing Gaussian metric cadence changed")
    case = dict(case, provider="api_vectors", evaluation_observations=24)
    recipe = recipe_identity(get_recipe("atlas", num_particles=20000, z_dim=2,
                                       batch_size=2048, **OVERRIDES).to_dict())
    if recipe["total_steps"] is not None:
        raise ValueError("Atlas Gaussian schedule must remain externally capped")
    return {"schema": "pg_gaussian2d_additional_question_proposal_v1", "status": "PROPOSED_NOT_EXECUTED",
            "case": case, "catalog_id": "source-family-16", "requested_recipe": "atlas",
            "requested_recipe_overrides": deepcopy(OVERRIDES), "resolved_recipe": recipe,
            "execution": {"updates": 1000, "eval_samples": 4096, "protocol_seed": 24002,
                          "heldout_seed": 34002, "metric_steps": api_contract.evaluation_steps(1000, 25),
                          "media_steps": api_contract.evaluation_steps(1000, 9), "terminal_observations": 5},
            "budget": {"physical_gpu": 0, "logical_device": "cuda:0", "cpu_threads": 1,
                       "cuda_memory_fraction": .2, "inclusive_paid_seconds": 180,
                       "attempts": 1, "export_grace_seconds": 0},
            "qualification": {"new_catalog_question": False, "old_cpu_credit": False,
                              "ordinary_current_26_slot_credit": False, "speed_or_default_credit": False}}


def source_manifest(root, *, supervisor_source_paths=()):
    """Existing snapshot API plus discovery/config/optional driver closure."""
    from experiments.forge.sources import inspect_source
    root = Path(root).resolve()
    # source_identity also binds native examples loaded by runpy. Including
    # all example Python sources preserves that original closure, even though
    # the direct Gaussian factory invokes none of those deferred hosts.
    examples = tuple(path.relative_to(root).as_posix() for path in sorted((root / "examples").rglob("*.py")))
    return inspect_source(root, extra_paths=EXTRA_SOURCE_PATHS + examples + tuple(supervisor_source_paths))


def load_proposal(path):
    proposal = json.loads(Path(path).read_text())
    if proposal["schema"] != "pg_gaussian2d_additional_question_proposal_v1":
        raise ValueError("wrong Gaussian proposal schema")
    if proposal["status"] != "PROPOSED_NOT_EXECUTED" or proposal["case"]["id"] != "api-gaussian2d":
        raise ValueError("proposal question/status differs")
    if proposal["requested_recipe_overrides"] != {"lr": .0053125, "prior_lr_mult": 1.5}:
        raise ValueError("Gaussian C6 tuple differs")
    return proposal


def global_state():
    np_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (np_state[0], tuple(int(value) for value in np_state[1]),
                      int(np_state[2]), int(np_state[3]), float(np_state[4])),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
                          if torch.cuda.is_initialized() else []}


def _case_identity(case):
    # These two annotations are inserted by metadata discovery, and are absent
    # when the unchanged provider reconstructs its own canonical case.
    return {key: deepcopy(value) for key, value in case.items()
            if key not in {"provider", "evaluation_observations"}}


def recipe_identity(value):
    """Normalize JSON's tuple/list boundary only; retain every Recipe value."""
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _module_modes(trainer):
    names = ("G", "D", "prior", "ema_G", "ema_prior", "ema_D")
    return {owner: {name: module.training for name, module in getattr(trainer, owner).named_modules()}
            for owner in names}


def _counts(optimizer, parameters):
    values, missing = [], 0
    for parameter in parameters:
        value = optimizer.state.get(parameter, {}).get("step")
        if value is None:
            values.append(0); missing += 1
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError("invalid optimizer step tensor")
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 or value != int(value):
            raise ValueError("invalid optimizer update clock")
        values.append(int(value))
    if not values:
        raise ValueError("an actual optimizer role has no parameters")
    return {"minimum": min(values), "maximum": max(values), "parameters": len(values),
            "uninitialized_parameters": missing}


class ObservedGaussianFixture:
    """Preserve fixture API and append source/owner/purity evidence to reads."""
    def __init__(self, fixture, proposal, *, source_guard):
        if not callable(source_guard):
            raise TypeError("a frozen imported-source guard is required")
        self.fixture, self.proposal, self.source_guard = fixture, deepcopy(proposal), source_guard
        self.observations = []
        self.source_guard()
        if not isinstance(fixture.trainer, GANTrainer):
            raise TypeError("Gaussian execution requires the actual public GANTrainer")
        if _case_identity(fixture.metadata) != _case_identity(proposal["case"]):
            raise ValueError("Gaussian canonical case/gates/resources differ")
        if recipe_identity(fixture.recipe.to_dict()) != proposal["resolved_recipe"]:
            raise ValueError("complete Gaussian Recipe differs")
        if fixture.seed != 24002 or fixture.execution_steps != 1000 or fixture.trainer.max_steps != 1000:
            raise ValueError("Gaussian seed/full execution horizon differs")
        if str(fixture.device) != "cuda:0" or tuple(fixture.trainer.prior.z.shape) != (20000, 2):
            raise ValueError("Gaussian original device/prior population differs")
        if fixture.completed_steps != 0 or fixture.trainer.completed_steps != 0 or fixture.trainer.policy.completed_steps != 0:
            raise ValueError("fresh Gaussian owner must start at zero updates")
        if fixture.trainer.policy.row_semantics != "independent":
            raise ValueError("Gaussian must retain independent cloud rows")
        if fixture.trainer.policy.prior is not fixture.trainer.prior or fixture.trainer.policy.table is not fixture.trainer.prior.z:
            raise ValueError("Gaussian policy must own the actual prior table")
        if (fixture.trainer.policy.G is not fixture.trainer.G or fixture.trainer.policy.D is not fixture.trainer.D
                or fixture.trainer.policy.table_optimizer is not fixture.trainer.opt_g):
            raise ValueError("Gaussian actual model/table optimizer owners differ")
        self.audit = PolicyLifecycleAudit(fixture.trainer.policy)

    def __getattr__(self, name):
        return getattr(self.fixture, name)

    def state_dict(self):
        return self.fixture.state_dict()

    def _state(self):
        state = self.fixture.state_dict()
        if not finite_policy_state(state):
            raise FloatingPointError("Gaussian public state is unhealthy")
        return {**state, "observed_module_modes": _module_modes(self.fixture.trainer)}

    def _updates(self):
        trainer = self.fixture.trainer
        result = {"generator": _counts(trainer.opt_g, trainer.G.parameters()),
                  "prior": _counts(trainer.opt_g, [trainer.prior.z]),
                  "discriminator": _counts(trainer.opt_d, trainer.D.parameters())}
        step = self.fixture.completed_steps
        if any(row["minimum"] != step or row["maximum"] != step for row in result.values()):
            raise ValueError("Gaussian actual optimizer clocks differ from completed updates")
        return result

    def _controls(self):
        result = controls_receipt(self.fixture.trainer.policy, self.fixture.completed_steps)
        if set(result["requested"]) != OWNERS or set(result["enabled"]) != OWNERS:
            raise ValueError("Gaussian policy owner manifest is incomplete")
        if not all(result["requested"].values()) or not all(result["enabled"].values()):
            raise ValueError("Gaussian omitted an actual requested Atlas owner")
        if not result["requested_owners_bound"] or not result["implementation_observed"]:
            raise ValueError("Gaussian ordered public lifecycle was not actually observed")
        if result["completed_steps"] != self.fixture.completed_steps:
            raise ValueError("Gaussian control cursor differs")
        if result["row_evidence_observations"] != self.fixture.completed_steps:
            raise ValueError("Gaussian actual row-evidence clock differs")
        if self.fixture.trainer.policy.controller.variant != "dv12":
            raise ValueError("Gaussian actual continuous controller is not DV12")
        execution = result["execution"]
        if (execution["model_devices"] != ["cuda:0"] or execution["floating_dtypes"] != ["torch.float32"]
                or execution["autocast_enabled"] is not False):
            raise ValueError("Gaussian actual device/dtype/autocast differs")
        result = deepcopy(result)
        result["cohort"] = COHORT
        return result

    def step(self):
        before = self.fixture.completed_steps
        value = self.fixture.step()
        if (self.fixture.completed_steps, self.fixture.trainer.completed_steps,
                self.fixture.trainer.policy.completed_steps) != (before + 1,) * 3:
            raise ValueError("one delegated Gaussian update did not advance all actual clocks")
        return value

    def observe(self, n=4096, seed=34002):
        if n != 4096 or seed != 34002:
            raise ValueError("Gaussian original held-out count/seed differs")
        self.source_guard()
        before = typed_state_digest(self._state())
        global_before = typed_state_digest(global_state())
        selected = self.fixture.trainer.served_snapshot()
        selected_before = typed_state_digest(selected)
        record = self.fixture.observe(n=n, seed=seed)
        controls = self._controls()
        updates = self._updates()
        global_after = typed_state_digest(global_state())
        after = typed_state_digest(self._state())
        selected_after = typed_state_digest(self.fixture.trainer.served_snapshot())
        self.source_guard()
        pure = before == after and global_before == global_after and selected_before == selected_after
        if not pure:
            raise RuntimeError("Gaussian observation changed complete training/selected/global RNG state")
        if controls["served_source"] != selected["source"]:
            raise ValueError("Gaussian actual selected source disagrees with control receipt")
        observation = {"cohort": COHORT, "owner": "particlegan.UpdatePolicy",
                       "completed_steps": self.fixture.completed_steps,
                       "selected_source": selected["source"],
                       "snapshot_sha256": selected_before, "digest_kind": DIGEST_KIND,
                       "sampler": "particlegan.GANTrainer.sample", "output_noise": False,
                       "latent_policy": "actual_selected_public_policy",
                       "controller": controls["diagnostics"]["controller"],
                       "backend_selection": controls["diagnostics"]["backend_selection"],
                       "requested_owners": controls["requested"], "enabled_owners": controls["enabled"],
                       "actual_optimizer_updates": updates,
                       "execution": controls["execution"],
                       "purity": {"before_sha256": before, "after_sha256": after,
                                  "global_before_sha256": global_before, "global_after_sha256": global_after,
                                  "snapshot_before_sha256": selected_before, "snapshot_after_sha256": selected_after,
                                  "allowed_changes": "none", "pure": pure}}
        self.observations.append(deepcopy(observation))
        # validate_observation/run_case preserve additional receipt fields.
        return {**record, "policy_observation": observation}

    def finalize(self, receipt):
        self.source_guard()
        state_before = typed_state_digest(self._state())
        rng_before = typed_state_digest(global_state())
        calls_before = len(self.observations)
        if receipt["case"]["id"] != self.case_id or receipt["recipe"] != recipe_identity(self.recipe.to_dict()):
            raise ValueError("Gaussian numerical receipt identity differs")
        if receipt["completed_updates"] != self.completed_steps:
            raise ValueError("Gaussian numerical receipt completion differs")
        reads = receipt["observations"]
        if len(reads) != len(self.observations) or any(
                row.get("policy_observation") != observation or row["step"] != observation["completed_steps"]
                for row, observation in zip(reads, self.observations)):
            raise ValueError("Gaussian numerical receipt is missing/forging actual observation evidence")
        complete = receipt["status"] == "COMPLETE"
        if complete and (self.completed_steps != 1000 or [row["step"] for row in reads] != self.proposal["execution"]["metric_steps"]):
            raise ValueError("Gaussian full protocol clock/cadence differs")
        controls = self._controls()
        updates = self._updates()
        if reads and controls["served_source"] != self.observations[-1]["selected_source"]:
            raise ValueError("Gaussian final controls differ from last actual selected observation")
        checkpoint_digest = typed_state_digest(self.state_dict())
        state_after = typed_state_digest(self._state())
        rng_after = typed_state_digest(global_state())
        if state_before != state_after or rng_before != rng_after or calls_before != len(self.observations):
            raise RuntimeError("Gaussian final attestation changed complete state/RNG/observation count")
        return {"schema": "pg_gaussian2d_policy_observer_sidecar_v1", "cohort": COHORT,
                "case_id": self.case_id, "pre_export_numerical_status": receipt["status"],
                "pre_export_numerical_verdict": receipt["verdict"], "completed_updates": self.completed_steps,
                "policy_protocol_complete": complete, "controls": controls,
                "actual_optimizer_updates": updates,
                "observations": deepcopy(self.observations),
                "checkpoint_state_sha256": checkpoint_digest,
                "checkpoint_digest_kind": DIGEST_KIND,
                "finalizer_purity": {"state_before_sha256": state_before, "state_after_sha256": state_after,
                                     "global_before_sha256": rng_before, "global_after_sha256": rng_after,
                                     "pure": True, "extra_observations": 0},
                "quality_from_owner_evidence": False, "training_or_rescoring_added": False,
                "observer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def build_factory(proposal, *, source_guard):
    """Explicit future runner hook; original contract.build owns construction."""
    from benchmarks.toy_audit import api_contract

    def build(case, **options):
        expected = dict(device="cuda:0", seed=24002, recipe_name="atlas",
                        max_steps=1000, recipe_overrides={"lr": .0053125, "prior_lr_mult": 1.5})
        if options != expected or _case_identity(case) != _case_identity(proposal["case"]):
            raise ValueError("Gaussian factory options/canonical declaration differ")
        source_guard()
        fixture = api_contract.build(case, **options)
        return ObservedGaussianFixture(fixture, proposal, source_guard=source_guard)
    return build


def run_bound_case(output, *, proposal, frozen_source, source_guard, admission_guard):
    """Root-supervised child interface; never admits/retries a physical job.

    Both guards must attest real frozen imports and the inherited live lease.
    The supervisor owns the inclusive180s clock and .2 GPU memory setup; this
    function calls neither a CUDA setup API nor a queue/admission API.
    """
    from . import api_run
    if not callable(source_guard) or not callable(admission_guard):
        raise TypeError("root's source and admitted-resource guards are required")
    source_guard()
    if proposal != declaration():
        raise ValueError("Gaussian root-supervised proposal differs from the existing case/Recipe")
    from experiments.forge.contracts import stable_hash
    if (not isinstance(frozen_source, dict) or not re.fullmatch(r"[a-f0-9]{40}", str(frozen_source.get("origin_commit", "")))
            or frozen_source.get("schema_version") != 1 or not isinstance(frozen_source.get("files"), dict)
            or stable_hash(frozen_source.get("files", {})) != frozen_source.get("digest")
            or any(path not in frozen_source["files"] for path in EXTRA_SOURCE_PATHS)
            or "benchmarks/toy_audit/gaussian2d_observer.py" not in frozen_source["files"]):
        raise ValueError("Gaussian requires a committed complete frozen source/data manifest")
    # Old inspection pins cannot cover the new hook/observer. The root's
    # imported-source guard additionally checks which modules Python loaded;
    # this verifies the complete current public/source/config closure itself.
    current_files = source_manifest(api_run.contract.ROOT)["files"]
    if any(frozen_source["files"].get(path) != digest for path, digest in current_files.items()):
        raise ValueError("Gaussian frozen source does not bind the complete current source closure")
    admission = admission_guard()
    required = {"status": "running", "device": "cuda:0", "physical_gpu": 0, "threads": 1,
                "memory_fraction": .2, "allowance_seconds": 180, "grace_seconds": 0,
                "lease_verified": True, "single_attempt": True}
    if not isinstance(admission, dict) or any(admission.get(key) != value for key, value in required.items()):
        raise ValueError("Gaussian actual admitted GPU/budget/thread resources differ")
    result = api_run.run_case(
        proposal["case"], output, device="cuda:0", recipe_name="atlas", steps=1000,
        eval_samples=4096, frames=9, seed=24002, recipe_overrides=deepcopy(OVERRIDES),
        wall_cap_seconds=180, fixture_factory=build_factory(proposal, source_guard=source_guard),
        receipt_finalizer=lambda fixture, receipt: fixture.finalize(receipt))
    try:
        source_guard()
    except Exception as error:
        result["numerical_before_bound_source_error"] = {
            key: deepcopy(result[key]) for key in ("status", "verdict", "passed", "failed_bounds")}
        result.update(status="ERROR", passed=False, verdict="FAIL", default_protocol_complete=False)
        result["failed_bounds"].append(f"Gaussian final imported-source guard error: {type(error).__name__}: {error}")
    result["gaussian_bound_protocol"] = {"declaration": deepcopy(proposal),
                                         "source": deepcopy(frozen_source), "admission": deepcopy(admission),
                                         "scientific_status": result["status"],
                                         "final_raw_receipt_is_verdict_authority": True}
    api_run.write_json(Path(output) / "receipt.json", result)
    return result


def result_exit_code(receipt):
    if receipt.get("status") != "COMPLETE":
        return 2
    return 0 if receipt.get("passed") is True and receipt.get("verdict") == "PASS" else 1
