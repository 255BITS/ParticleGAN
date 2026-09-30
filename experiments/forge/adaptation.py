"""Public-trainer target shift with an own-checkpoint frozen negative control.

The prefix is executed once. Both suffixes start from the same measured public
state, including optimizer moments and named streams. Frozen continuation only
samples; it cannot acquire the shifted target through training or EMA updates.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import platform

import torch

from .adapters import _Run, _context, _event, _models
from .api import CapabilityError
from .artifacts import verify_artifacts
from .contracts import atomic_json, file_hash, read_json, stable_hash
from .sources import runtime_manifest
from .state import state_digest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "experiments/forge/adaptation.py", "experiments/forge/adapters.py",
    "experiments/forge/api.py", "experiments/forge/rng.py", "experiments/forge/state.py",
    "experiments/forge/mechanisms.py", "particlegan/training.py", "particlegan/recipes.py",
    "particlegan/k3p.py", "particlegan/grad_regularizers.py", "particlegan/particle_prior.py",
    "lib/toy_models.py", "benchmarks/locked_shared/mode_hold.py",
    "benchmarks/toy100/continuous_probe.py",
)


def _optimizer_rows(state):
    """Read Adam counters and hashes from actual public optimizer checkpoints."""
    rows = []
    for role, optimizer in zip(("generator_and_prior", "discriminator"), state["trainer"]["optimizers"]):
        counters = [int(value["step"].item()) if isinstance(value["step"], torch.Tensor) else int(value["step"])
                    for value in optimizer["state"].values() if "step" in value]
        if not counters or min(counters) != max(counters):
            raise RuntimeError("optimizer counters are absent or inconsistent")
        rows.append({"role": role, "updates": counters[0], "moment_steps_min": min(counters),
                     "moment_steps_max": max(counters), "parameter_states": len(counters),
                     "state_sha256": state_digest(optimizer)})
    return rows


def _training_state(state):
    # Evaluation is permitted to advance only eval-family streams. Everything
    # else, including EMA, K3P state and the data cursor, remains frozen.
    trainer = {k: v for k, v in state["trainer"].items() if k != "streams"}
    bindings = state["streams"]["manifest"]["bindings"]
    streams = {k: v for k, v in state["streams"]["states"].items() if bindings[k]["family"] != "eval"}
    return {"context": {k: v for k, v in state.items() if k not in {"trainer", "streams"}},
            "trainer": trainer, "training_streams": streams}


def _provenance(request, device):
    sources = {path: file_hash(ROOT / path) for path in SOURCE_FILES}
    declared = request.get("source", {}).get("files", {})
    mismatch = [path for path, digest in sources.items() if path in declared and declared[path] != digest]
    if mismatch:
        raise CapabilityError(["executed source differs from request: " + path for path in mismatch])
    runtime = {**runtime_manifest(), "torch": str(torch.__version__),
               "torch_git_revision": torch.version.git_version,
               "threads": torch.get_num_threads(), "device": str(device),
               "deterministic": torch.are_deterministic_algorithms_enabled(),
               "cpu_capability": torch.backends.cpu.get_cpu_capability(),
               "processor": platform.processor()}
    if torch.device(device).type == "cuda":
        runtime["gpu_model"] = torch.cuda.get_device_name(torch.device(device))
    return sources, runtime


def _summary(points, execution):
    from benchmarks.toy100.continuous_probe import _window, RECOVERY_DEADLINE
    horizon, shift = execution["original_schedule_horizon"], execution["shift_step"]
    stationary = _window([p for p in points if p["step"] in range(horizon - 200, horizon + 1, 50)])
    continued = _window([p for p in points if horizon < p["step"] <= shift])
    late = _window([p for p in points if p["step"] >= shift + RECOVERY_DEADLINE])
    return {"stationary": stationary, "continued_hold": continued,
            "shift_recovery": {"deadline_step": shift + RECOVERY_DEADLINE,
                               "deadline_window": late,
                               "deadline_pass": late["checks"] >= 5 and late["pass_all"]}}


def run_adaptation(request: dict, task: dict, output_dir: Path, device: str) -> dict:
    """Run the declared pair; short task fixtures remain non-qualifying evidence."""
    from benchmarks.locked_shared.mode_hold import ring_means, sample_ring, diversity, SIGMA
    from benchmarks.toy100.continuous_probe import match_frozen_control

    execution = task["execution"]
    steps, shift_step, cadence = (execution[k] for k in ("steps", "shift_step", "diagnostic_every"))
    if (any(type(v) is not int for v in (steps, shift_step, cadence)) or cadence < 1
            or not 0 < shift_step < steps or steps % cadence or shift_step % cadence
            or execution.get("control_steps") != steps or execution.get("frozen_control") is not True):
        raise CapabilityError(["paired adaptation needs aligned active/control budgets and a frozen suffix"])
    if execution["original_schedule_horizon"] != 1200 or execution.get("shift") != [1.0, 0.0]:
        raise CapabilityError(["paired adaptation preserves the 1200-update horizon and target shift [1, 0]"])
    if task["evaluation"].get("scoring_weights") != "live":
        raise CapabilityError(["paired adaptation measures live public sampling only"])
    spec = execution["host_definition"]
    resources = {"num_particles": spec["particles"], "z_dim": spec["z_dim"], "batch_size": spec["batch"]}
    sources, runtime = _provenance(request, device)
    output = Path(output_dir)
    artifact = output / "paired-adaptation"
    artifact.mkdir(parents=True, exist_ok=True)
    context = _context(request, task, device, resources)
    trainer = context.build_trainer(*_models(context, spec), max_steps=steps)
    active_run = _Run(context, trainer, output, task)
    data = context.streams.generator("data", component="target", purpose="training", device="cpu")
    means = ring_means()
    original_means = means.clone()
    prefix = []
    shift_samples = None
    for step in range(1, shift_step + 1):
        active_run.step(sample_ring(means, context.recipe.batch_size, SIGMA, data).to(device))
        if step % cadence == 0:
            samples = active_run.evaluate(lambda: active_run.sample(spec["eval_samples"]).detach().cpu())
            point = {"step": step, **diversity(samples, means, detailed=True)}
            prefix.append(point)
            _event("observation", task=task["id"], arm="shared_prefix", step=step,
                   metrics={key: point[key] for key in ("modes", "hq", "effective_modes")})
            if step == shift_step:
                shift_samples = samples
    before = deepcopy(prefix[-1])
    means.add_(torch.tensor(execution["shift"], dtype=means.dtype))
    after = {"step": shift_step, **diversity(shift_samples, means, detailed=True)}
    checkpoint = context.state_dict()
    torch.save(checkpoint, artifact / "active-at-shift.pt")
    frozen_context = _context(request, task, device, resources)
    frozen_trainer = frozen_context.build_trainer(*_models(frozen_context, spec), max_steps=steps)
    frozen_context.load_state_dict(checkpoint)
    frozen_checkpoint = frozen_context.state_dict()
    if state_digest(checkpoint) != state_digest(frozen_checkpoint):
        raise RuntimeError("public checkpoint restore changed the paired prefix state")
    torch.save(frozen_checkpoint, artifact / "frozen-at-shift.pt")
    torch.save({"before": original_means, "after": means, "shift_samples": shift_samples}, artifact / "shift-target.pt")
    shift_pair = {"before": before, "after": after, "optimizer_at_shift": _optimizer_rows(checkpoint),
                  "public_state_sha256": state_digest(checkpoint),
                  "training_state_sha256": state_digest(_training_state(checkpoint)),
                  "named_rng_sha256": state_digest(checkpoint["streams"])}
    frozen_run = _Run(frozen_context, frozen_trainer, output / "frozen", task)
    active_points, frozen_points = deepcopy(prefix), deepcopy(prefix)
    for step in range(shift_step + 1, steps + 1):
        active_run.step(sample_ring(means, context.recipe.batch_size, SIGMA, data).to(device))
        if step % cadence == 0:
            for arm, run, points in (("active", active_run, active_points), ("frozen", frozen_run, frozen_points)):
                metrics = run.evaluate(lambda run=run: diversity(run.sample(spec["eval_samples"]).detach().cpu(), means, detailed=True))
                point = {"step": step, **metrics}
                points.append(point)
                _event("observation", task=task["id"], arm=arm, step=step,
                       metrics={key: metrics[key] for key in ("modes", "hq", "effective_modes")})
    active_final, frozen_final = context.state_dict(), frozen_context.state_dict()
    if state_digest(_training_state(checkpoint)) != state_digest(_training_state(frozen_final)):
        raise RuntimeError("frozen negative control changed training state")
    torch.save(active_final, artifact / "active-final.pt")
    torch.save(frozen_final, artifact / "frozen-final.pt")
    config = {"recipe": context.recipe.to_dict(), "prior": context.prior_config,
              "initializer": context.initializer, "execution": execution, "host_definition": spec,
              "protocol": request["protocol"], "candidate_revision": request.get("candidate_revision")}
    atomic_json(artifact / "config.json", config)
    atomic_json(artifact / "provenance.json", {"source_sha256": sources, "runtime": runtime,
                "request_source_digest": request.get("source", {}).get("digest")})
    common = {"mode": "scheduled" if context.recipe.lr_anneal_start or context.recipe.lr_floor != 1 else "constant",
              "config_sha256": stable_hash(config), "source_sha256": sources, "runtime": runtime,
              "steps": steps, "noise_horizon": context.recipe.total_steps, "diagnostic_every": cadence,
              "dense_after": None, "dense_until": None, "shift_step": shift_step,
              "shift": execution["shift"], "shift_pair": shift_pair,
              "sampling_law": "public_prior_without_output_noise", "eval_output_noise": "clean"}
    active = {**deepcopy(common), "freeze_after_shift": False, "diagnostic": active_points,
              "optimizer_final": _optimizer_rows(active_final), **_summary(active_points, execution)}
    frozen = {**deepcopy(common), "freeze_after_shift": True, "diagnostic": frozen_points,
              "optimizer_final": _optimizer_rows(frozen_final), **_summary(frozen_points, execution)}
    confirmed = match_frozen_control(active, frozen)
    # Window decisions are diagnostics; the independent grader recomputes them.
    atomic_json(artifact / "active.json", active)
    atomic_json(artifact / "frozen.json", frozen)
    proof = {"schema_version": 1, "active_at_shift": "active-at-shift.pt", "frozen_at_shift": "frozen-at-shift.pt",
             "active_final": "active-final.pt", "frozen_final": "frozen-final.pt", "target": "shift-target.pt",
             "config": "config.json", "provenance": "provenance.json",
             "shared_prefix_state_sha256": state_digest(checkpoint),
             "frozen_training_state_sha256": state_digest(_training_state(frozen_final)),
             "active_final_state_sha256": state_digest(active_final)}
    atomic_json(artifact / "pair-proof.json", proof)
    evidence = {"active": active, "frozen": frozen, "artifact_root": str(artifact.resolve()),
                "artifact_schema": "public_paired_adaptation_v1", "pair_proof": proof,
                "scoring_weights": "live", "diagnostics": {"matched_control": confirmed["matched_control"],
                    "recorded_status": confirmed["status"], "quality_evidence_only": True},
                "frozen_rng_audits": frozen_run.rng_audits}
    raw = active_run.receipt(evidence)
    raw["evidence"]["guards"]["unintended_rng_deviations"] += sum(
        audit["unintended_rng_deviations"] for audit in frozen_run.rng_audits)
    verify_pair_artifacts(raw["evidence"])
    atomic_json(output / "adapter-receipt.json", raw)
    return raw


def verify_pair_artifacts(evidence: dict) -> dict:
    """Check actual saved state/optimizer/sample bytes, not declared PASS hashes."""
    from benchmarks.locked_shared.mode_hold import diversity, ring_means

    root = Path(evidence["artifact_root"])
    verify_artifacts(root, evidence["artifact_manifest"])
    if evidence.get("artifact_schema") != "public_paired_adaptation_v1":
        raise ValueError("unsupported paired adaptation artifact schema")
    proof = read_json(root / "pair-proof.json")
    if proof != evidence.get("pair_proof"):
        raise ValueError("pair proof differs from saved artifact")
    names = {"active_at_shift": "active-at-shift.pt", "frozen_at_shift": "frozen-at-shift.pt",
             "active_final": "active-final.pt", "frozen_final": "frozen-final.pt", "target": "shift-target.pt",
             "config": "config.json", "provenance": "provenance.json"}
    if any(proof.get(k) != name for k, name in names.items()):
        raise ValueError("pair proof uses unexpected artifact paths")
    states = {key: torch.load(root / names[key], map_location="cpu", weights_only=True)
              for key in ("active_at_shift", "frozen_at_shift", "active_final", "frozen_final")}
    prefix = states["active_at_shift"]
    if not (state_digest(prefix) == state_digest(states["frozen_at_shift"]) == proof["shared_prefix_state_sha256"]):
        raise ValueError("active and frozen public prefix checkpoints differ")
    if not (state_digest(_training_state(prefix)) == state_digest(_training_state(states["frozen_final"]))
            == proof["frozen_training_state_sha256"]):
        raise ValueError("frozen training state changed after shift")
    if state_digest(states["active_final"]) != proof["active_final_state_sha256"]:
        raise ValueError("active final state digest mismatch")
    config, provenance = read_json(root / names["config"]), read_json(root / names["provenance"])
    target = torch.load(root / names["target"], map_location="cpu", weights_only=True)
    execution = config["execution"]
    if prefix["trainer"]["completed_steps"] != execution["shift_step"]:
        raise ValueError("public checkpoint is not at the declared shift")
    if (target["shift_samples"].shape != (config["host_definition"]["eval_samples"], 2)
            or not bool(torch.isfinite(target["shift_samples"]).all())):
        raise ValueError("saved shift samples are incomplete or nonfinite")
    if (not torch.equal(target["before"], ring_means()) or not torch.equal(
            target["before"] + torch.tensor(execution["shift"]), target["after"])):
        raise ValueError("saved target shift differs from declared ring")
    for name, final in (("active", "active_final"), ("frozen", "frozen_final")):
        run = evidence[name]
        if run != read_json(root / (name + ".json")):
            raise ValueError("paired curve differs from saved artifact")
        if (run["config_sha256"] != stable_hash(config) or run["source_sha256"] != provenance["source_sha256"]
                or run["runtime"] != provenance["runtime"] or stable_hash(states[final]["recipe"]) != stable_hash(config["recipe"])
                or states[final]["prior"] != config["prior"]
                or run["noise_horizon"] != config["recipe"]["total_steps"]
                or run["noise_horizon"] != execution["original_schedule_horizon"]):
            raise ValueError("paired provenance/config disagrees with measured checkpoint")
        expected = execution["shift_step"] if name == "frozen" else execution["steps"]
        if (states[final]["trainer"]["completed_steps"] != expected
                or run["optimizer_final"] != _optimizer_rows(states[final])
                or any(row["updates"] != expected for row in run["optimizer_final"])):
            raise ValueError("measured final optimizer counters differ from the declared branch")
        pair = run["shift_pair"]
        if (next((p for p in run["diagnostic"] if p["step"] == execution["shift_step"]), None) != pair["before"]
                or [p["step"] for p in run["diagnostic"]] != list(range(
                    execution["diagnostic_every"], execution["steps"] + 1, execution["diagnostic_every"]))):
            raise ValueError("paired diagnostic curve does not preserve the measured shift boundary or cadence")
        if (pair["optimizer_at_shift"] != _optimizer_rows(prefix)
                or any(row["updates"] != execution["shift_step"] for row in pair["optimizer_at_shift"])
                or pair["public_state_sha256"] != state_digest(prefix)
                or pair["training_state_sha256"] != state_digest(_training_state(prefix))
                or pair["named_rng_sha256"] != state_digest(prefix["streams"])):
            raise ValueError("recorded prefix counters/state do not match the public checkpoint")
        for side in ("before", "after"):
            actual = {"step": execution["shift_step"], **diversity(target["shift_samples"], target[side], detailed=True)}
            if pair[side] != actual:
                raise ValueError("shift diagnostic disagrees with saved live samples")
    return {"verified": True, "prefix_step": prefix["trainer"]["completed_steps"],
            "frozen_final_step": states["frozen_final"]["trainer"]["completed_steps"],
            "active_final_step": states["active_final"]["trainer"]["completed_steps"],
            "shared_prefix_state_sha256": state_digest(prefix)}
