"""Restore analytic smoke witnesses under each legacy family's actual API law.

Only constructed parameter states are reused. Optimizers and named streams are
new and task-owned. No optimizer step, fitting, or ordinary qualification occurs.
The input archive must be the source-bound, zero-update BCap capacity receipt.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from particlegan import init
from benchmarks.locked_shared import two_pole
from benchmarks.locked_shared.hosts import unused_token_hold, ae_gan_hold
from benchmarks.locked_shared.baseline import score_metrics
from benchmarks.transfer_suite.legacy_noise_adapters import wrap_output
from experiments.forge.api import resolve_public_recipe
from experiments.forge.behavior_adapters import BehaviorComponents
from experiments.forge.configuration_search import _declarations, _load_spec
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.sources import runtime_manifest
from experiments.forge.state import state_digest

STUDIES = (
    "r1r2-modern-family-round1-v1", "bcap-family-defaults-round1-v1",
    "k3p-family-defaults-round1-v1", "ka2-family-defaults-round1-v1",
    "release07-gan-v3-mog-family-defaults-round1-v1",
)
SMOKE = ("two_pole", "unused_token_hold", "ae_gan_hold")
# Optimizer rates/penalties cannot constrain a constructed parameter state.
# Every field read by these serving/AE paths stays invariant within a family.
LAW_FIELDS = ("continuous_policy", "row_policy", "serve_average", "output_noise_mode",
              "output_noise_std", "output_noise_warmup", "prior_kind", "sigma_rel",
              "standardize", "routing_temperature", "distance_reduction", "encoder_mode")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checks(task, metrics):
    values = score_metrics(metrics, task["evaluation"]["thresholds"])
    return {"metrics": {name: metrics[name] for name, _, _ in task["evaluation"]["thresholds"]},
            "passed": all(c["status"] == "PASS" for c in values), "bounds": values}


def load_witnesses(directory):
    directory = Path(directory).resolve()
    receipt = read_json(directory / "receipt.json")
    if (receipt.get("training_updates") != 0 or receipt.get("fitting_updates") != 0
            or receipt.get("qualification_credit") is not False
            or receipt.get("global_torch_rng_unchanged") is not True):
        raise ValueError("input is not the zero-update capacity cohort")
    # These executable bindings are needed to interpret the saved parameters.
    relevant = {name: digest for name, digest in receipt["source_files_sha256"].items()
                if name.startswith("particlegan/") or name.startswith("benchmarks/locked_shared/")
                or name in {"experiments/forge/behavior_adapters.py", "experiments/forge/api.py",
                            "experiments/forge/rng.py", "experiments/forge/taskrecipes.py",
                            "benchmarks/transfer_suite/legacy_noise_adapters.py"}}
    if not relevant:
        raise ValueError("source witness has no executable source bindings")
    for name, digest in relevant.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"capacity source changed: {name}; regenerate the zero-update witness")
    records = {r["task_id"]: r for r in receipt["records"]}
    result = {}
    for name in SMOKE:
        record = records[name]
        if record["representation_status"] != "SUPPORTED" or record["optimizer_updates"] != 0:
            raise ValueError(f"missing supported constructed state: {name}")
        if stable_hash(record["task_binding"]) != stable_hash(read_json(ROOT / f"configs/forge/tasks/{name}.json")):
            raise ValueError(f"task changed: {name}")
        state_path, metrics_path = directory / name / "state.pt", directory / name / "metrics.json"
        if sha(state_path) != record["state_file_sha256"] or sha(metrics_path) != record["raw_metrics_sha256"]:
            raise ValueError(f"capacity artifact changed: {name}")
        result[name] = {"models": torch.load(state_path, map_location="cpu", weights_only=True)["models"],
                        "state_sha256": record["state_file_sha256"],
                        "metrics_sha256": record["raw_metrics_sha256"]}
    return receipt, result, relevant


def replay(candidate, name, witness):
    task = read_json(ROOT / f"configs/forge/tasks/{name}.json")
    components = BehaviorComponents({"candidate": candidate, "protocol": {"seed": 0}}, task)
    if name == "two_pole":
        critic = two_pole.HostCritic()
        particles = nn.Parameter(torch.zeros_like(witness["models"][1]))
        components.bind(generator=None, critic=critic, direct_particles=[particles], opt_g=None, opt_d=None)
        critic.load_state_dict(witness["models"][0], strict=True)
        with torch.no_grad():
            particles.copy_(witness["models"][1])
        models = [critic, particles]
        def measure():
            return {"mean_abs": float(particles.detach().abs().mean()),
                    "grad_med": two_pole._grad_median(critic, two_pole.real_batch(len(particles)), particles)}
        control = checks(task, {"mean_abs": 0., "grad_med": 0.})
    elif name == "unused_token_hold":
        # The actual run_behavior installs this public KEEP declaration too.
        init.register(unused_token_hold.SharedSlotStudent,
                      lambda model: {n: init.KEEP for n, _ in model.named_parameters(recurse=False)})
        student, critic = unused_token_hold.SharedSlotStudent(), unused_token_hold.SlotCritic()
        components.bind(generator=student, critic=critic, opt_g=None, opt_d=None)
        student.load_state_dict(witness["models"][0], strict=True)
        critic.load_state_dict(witness["models"][1], strict=True)
        models = [student, critic]
        measure = lambda: unused_token_hold.score_student(student)
        original = student.slot.detach().clone()
        with torch.no_grad():
            student.slot.zero_()
        control = checks(task, measure())
        with torch.no_grad():
            student.slot.copy_(original)
    else:
        cfg = ae_gan_hold.HoldConfig(name="capacity-only")
        recipe = components.encoder_recipe(cfg)
        prior = components.make_prior(recipe)
        encoder, decoder, critic = ae_gan_hold.MLP(2, 4), ae_gan_hold.MLP(2, 2), ae_gan_hold.MLP(2, 1)
        served = wrap_output(decoder, components.noise)
        components.bind(generator=served, encoder=encoder, critic=critic,
                        priors=[prior], opt_g=None, opt_d=None)
        models = [encoder, decoder, critic, prior]
        for model, state in zip(models, witness["models"], strict=True):
            model.load_state_dict(state, strict=True)
        measure = lambda: ae_gan_hold.evaluate(encoder, served, prior, recipe)
        original = prior.z.detach().clone()
        with torch.no_grad():
            prior.z.zero_()
        with components.noise.evaluation(task["execution"]["steps"]):
            control = checks(task, measure())
        with torch.no_grad():
            prior.z.copy_(original)
    def capture():
        return {"models": [m.state_dict() if isinstance(m, nn.Module) else m.detach().clone() for m in models],
                "optimizers": {role: opt.state_dict() for role, opt in components.optimizers.items()}}
    before = state_digest(capture())
    stream_before = components.context.streams.audit()
    rows = []
    for label in sorted({math.ceil(i * task["execution"]["steps"] / 24) for i in range(1, 25)}):
        with components.noise.evaluation(label):
            row = checks(task, measure())
        rows.append({"schedule_label": label, **row})
    if state_digest(capture()) != before:
        raise ValueError("capacity observation changed model or optimizer state")
    audit = components.context.streams.compare(stream_before, components.context.streams.audit())
    if audit["unintended_rng_deviations"] or control["passed"]:
        raise ValueError("capacity stream audit or negative control failed")
    for opt in components.optimizers.values():
        for owned in getattr(opt, "optimizers", [opt]):
            if owned.state:
                raise ValueError("optimizer state has updates in a capacity observation")
    return {"task_id": name, "representation_status": "SUPPORTED" if all(r["passed"] for r in rows) else "UNRESOLVED",
        "gate_scope": "Constructed public parameters meet the declared numerical tolerance at all schedule labels; optimizer acquisition is untested.",
        "recipe": asdict(components.recipe), "prior": components.context.prior_config,
        "sampling": task["evaluation"], "observations_are_training_checkpoints": False,
        "optimizer_updates": 0, "fitting_updates": 0, "ordinary_qualification": False,
        "evaluated_schedule_labels": [r["schedule_label"] for r in rows],
        "all_capacity_observations_passed": all(r["passed"] for r in rows),
        "minimum_metrics": {k: min(r["metrics"][k] for r in rows) for k in rows[0]["metrics"]},
        "maximum_metrics": {k: max(r["metrics"][k] for r in rows) for k in rows[0]["metrics"]},
        "negative_control": control, "constructed_state_sha256": before,
        "model_optimizer_state_unchanged": True, "unintended_rng_deviations": audit["unintended_rng_deviations"],
        "input_state_sha256": witness["state_sha256"], "input_metrics_sha256": witness["metrics_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witnesses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt, witnesses, source_bindings = load_witnesses(args.witnesses)
    previous_threads = torch.get_num_threads()
    records = []
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.set_num_threads(1)
            for study in STUDIES:
                spec = _load_spec(ROOT, study)
                candidates = _declarations(ROOT, spec)
                laws = [{key: getattr(resolve_public_recipe(c), key) for key in LAW_FIELDS} for c, _ in candidates]
                if len({stable_hash(law) for law in laws}) != 1:
                    raise ValueError("family configs have different serving laws; evaluate separately")
                representative = candidates[0][0]
                for name in SMOKE:
                    with torch.random.fork_rng(devices=[]):
                        record = replay(representative, name, witnesses[name])
                    record.update(family=spec["trainer_family"], study_id=study,
                        actually_observed_candidate_id=representative["id"],
                        compatible_configuration_ids=[c["configuration_id"] for c, _ in candidates],
                        family_law= laws[0], all_family_configuration_laws_verified_equal=True)
                    records.append(record)
    finally:
        torch.set_num_threads(previous_threads)
    result = {"schema_version": 1, "kind": "zero_update_cross_family_smoke_capacity_admission",
        "training_updates": 0, "fitting_updates": 0, "ordinary_qualification": False,
        "representation_scope": "Exact unchanged smoke task numeric gates and public serving laws; optimizer learning, mechanisms during training, and full target distribution quality are untested.",
        "records": records, "input_receipt": str((args.witnesses / "receipt.json").resolve()),
        "input_receipt_sha256": sha(args.witnesses / "receipt.json"),
        "source_bindings": source_bindings, "runtime": runtime_manifest(),
        "reproducer_sha256": sha(__file__), "complete_shared_suite_representation": "UNRESOLVED"}
    atomic_json(args.output, result)
    print(json.dumps({"capacity_cells": len(records), "supported": sum(r["representation_status"] == "SUPPORTED" for r in records),
                      "training_updates": 0, "ordinary_qualification": False}), flush=True)
    return 0 if all(r["representation_status"] == "SUPPORTED" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
