"""Verify retained zero-update image states under current protected API bytes.

No fitting, training, new parameters or scientific configuration search occurs.
Original raw files remain byte-identical; the sidecar preserves both bindings.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from benchmarks.toy_audit import api_contract
from benchmarks.toy_audit.api_family_search import proof_bindings
from benchmarks.toy_audit.api_run import json_value
from policy_image_capacity import same


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reverify(previous_receipt, output):
    previous_receipt, output = Path(previous_receipt).resolve(), Path(output).resolve()
    previous_hash = sha256(previous_receipt)
    previous = json.loads(previous_receipt.read_text())
    if (previous["declaration"]["ordinary_training_updates"] != 0
            or previous["declaration"]["new_fitting_updates"] != 0
            or previous["declaration"]["same_mode_neighbors"] is not True
            or previous["ordinary_qualification_credit"] is not False):
        raise ValueError("compatibility requires the retained zero-update parameter construction")
    if len(previous["records"]) != 4:
        raise ValueError("all four declared image/family cells must be retained")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    cases = api_contract.discover()
    records = []
    for original in previous["records"]:
        start = time.monotonic()
        case, family = cases[original["case_id"]], original["family"]
        bindings = proof_bindings(case, family)
        for name in ("case_sha256", "preset_sha256", "sampling_sha256"):
            if bindings[name] != original["bindings"][name]:
                raise ValueError("compatibility cannot adapt the original host/recipe/sampling contract")
        changed = {path for path in set(bindings["source_files_sha256"]) | set(original["bindings"]["source_files_sha256"])
                   if bindings["source_files_sha256"].get(path) != original["bindings"]["source_files_sha256"].get(path)}
        if changed - {"benchmarks/toy_audit/api_contract.py", "benchmarks/toy_audit/api_images.py"}:
            raise ValueError("compatibility covers the declared override guard and WordFixture-only provider publication; other protected source changes need separate review")
        for artifact in original["artifacts"].values():
            if sha256(artifact["path"]) != artifact["sha256"]:
                raise ValueError("retained raw state/samples changed before compatibility verification")
        state = torch.load(original["artifacts"]["state"]["path"], map_location="cpu", weights_only=True)
        if (state["max_steps"] != case["default_steps"]
                or state["api_state"]["completed_steps"] != 0
                or json_value(state["recipe"]) != original["resolved_recipe"]):
            raise ValueError("retained public fixture/horizon/zero-clock/recipe mismatch")
        fixture = api_contract.build(case, device="cpu", seed=state["seed"], recipe_name=family,
                                     max_steps=state["max_steps"])
        if fixture.recipe.to_dict() != state["recipe"] or fixture.case != state["case"]:
            raise ValueError("current public fixture resolves a different Recipe/host descriptor")
        fixture.trainer.load_state_dict(state["api_state"])
        fixture.policy = fixture.trainer.policy
        fixture.data_generator.set_state(state["data_generator"])
        if not same(state, fixture.state_dict()):
            raise ValueError("public restore did not preserve the complete retained fixture")
        before = deepcopy(fixture.state_dict())
        rng_before = torch.random.get_rng_state().clone()
        observation = api_contract.validate_observation(fixture.observe(n=case["eval_samples"], seed=34002))
        with np.load(original["artifacts"]["samples"]["path"], allow_pickle=False) as retained:
            exact = np.array_equal(retained["samples"], api_contract.array(observation["views"][0]["samples"]))
            exact_targets = np.array_equal(retained["target"], api_contract.array(observation["views"][0]["target"]))
        metrics_equal = observation["metrics"] == original["observations"][0]["metrics"]
        pure = same(before, fixture.state_dict()) and torch.equal(rng_before, torch.random.get_rng_state())
        if (not exact or not exact_targets or not metrics_equal or not pure
                or observation["passed"] is not True or observation["failed_bounds"]):
            raise ValueError("retained-state public sampling or original numerical gate changed")
        if bindings != proof_bindings(case, family):
            raise ValueError("current protected source changed during compatibility verification")
        for artifact in original["artifacts"].values():
            if sha256(artifact["path"]) != artifact["sha256"]:
                raise ValueError("compatibility verification changed an original raw artifact")
        record = deepcopy(original)
        record.update(bindings=bindings, original_bindings=deepcopy(original["bindings"]),
                      compatibility_only=True, compatibility_source_changes=sorted(changed),
                      observations=[dict(metrics=observation["metrics"], passed=observation["passed"],
                                         failed_bounds=observation["failed_bounds"], completed_steps=0,
                                         samples_key="samples")],
                      retained_samples_bitwise_equal=exact, retained_targets_bitwise_equal=exact_targets,
                      metrics_unchanged=metrics_equal, observer_state_rng_unchanged=pure,
                      compatibility_elapsed_seconds=time.monotonic() - start)
        records.append(record)
        print(json.dumps({"family": family, "case_id": case["id"], "status": record["status"],
                          "bitwise_samples_equal": exact, "metrics_unchanged": metrics_equal}), flush=True)
    if sha256(previous_receipt) != previous_hash:
        raise ValueError("compatibility verification altered the original receipt")
    receipt = dict(schema_version=1, kind="retained_public_sampler_compatibility",
                   previous_receipt=dict(path=str(previous_receipt), sha256=previous_hash),
                   ordinary_training_updates=0, fitting_updates=0, new_parameter_candidates=0,
                   ordinary_qualification_credit=False, original_raw_files_unchanged=True,
                   reproducer_sha256=sha256(__file__), records=records)
    payload = json.dumps(json_value(receipt), sort_keys=True, indent=2, allow_nan=False) + "\n"
    (output / "receipt.json").write_text(payload)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reverify(args.previous_receipt, args.output)


if __name__ == "__main__":
    main()
