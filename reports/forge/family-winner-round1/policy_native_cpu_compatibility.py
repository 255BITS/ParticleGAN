"""Bounded CPU admission captures for current public native policy capacity.

Reuse the declared analytic construction, not a CUDA RNG realization. These
zero-update CPU witnesses support exact CPU replay, never learned qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from benchmarks.toy_audit import api_contract, api_family_search, api_run
from policy_native_representation import construct, scalar_observation, state_hash


CASES = ("api-grid100", "api-rotated100", "api-staggered100")
FAMILIES = ("atlas", "e22")


def run(output, *, wall_cap=120):
    started = time.monotonic()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    declaration = dict(kind="cpu_backend_capacity_admission", families=list(FAMILIES), cases=list(CASES),
                       wall_cap_seconds=wall_cap, device="cpu", cpu_threads=1,
                       seed=24002, evaluation_seed=34002, evaluation_samples_per_case=20000,
                       captures_per_case=1, ordinary_training_updates=0, fitting_updates=0,
                       source="Current public fixture and root's declared analytic construction",
                       scope="CPU public selected/served-law capacity at original gate tolerance; no CUDA RNG replay, training or robustness qualification",
                       reproducer_sha256=api_run.file_hash(__file__),
                       construction_reproducer_sha256=api_run.file_hash(Path(__file__).with_name("policy_native_representation.py")))
    api_run.write_json(output / "declaration.json", declaration)
    records, fixtures = [], []
    cases = api_contract.discover()

    def save(status):
        receipt = dict(schema_version=1, kind=declaration["kind"], declaration=declaration,
                       status=status, elapsed_seconds=time.monotonic() - started,
                       ordinary_qualification_credit=False, records=records)
        api_run.write_json(output / "receipt.json", receipt)
        return receipt

    for family in FAMILIES:
        for name in CASES:
            if time.monotonic() - started >= wall_cap:
                return save("INCOMPLETE_WALL_CAP")
            case = cases[name]
            if case["eval_samples"] != 20000:
                raise ValueError("CPU admission must retain the original 20000-sample native gate")
            geometry = "axis_unique" if family == "atlas" and name in ("api-grid100", "api-staggered100") else "lattice"
            bindings = api_family_search.proof_bindings(case, family)
            case_start = time.monotonic()
            fixture = construct(case, family, unique_support=True, device="cpu", geometry=geometry)
            state = deepcopy(fixture.state_dict())
            before_hash = state_hash(state)
            rng_before = torch.random.get_rng_state().clone()
            observation, samples = scalar_observation(fixture, count=case["eval_samples"], seed=34002, return_samples=True)
            pure = state_hash(fixture.state_dict()) == before_hash
            global_pure = torch.equal(rng_before, torch.random.get_rng_state())
            if not pure or not global_pure or fixture.completed_steps != 0:
                raise ValueError("CPU capacity observer changed fixture, optimizer or training RNG state")
            if bindings != api_family_search.proof_bindings(case, family):
                raise ValueError("physical native capacity source changed during capture")
            directory = output / family / name
            directory.mkdir(parents=True)
            state_path, sample_path = directory / "state.pt", directory / "samples.npz"
            torch.save(state, state_path)
            np.savez_compressed(sample_path, **{"34002": samples})
            record = dict(family=family, case_id=name,
                          status="SUPPORTED" if observation["passed"] else "UNRESOLVED",
                          claim_scope=declaration["scope"], bindings=bindings, observations=[observation],
                          resolved_recipe=fixture.recipe.to_dict(),
                          construction=dict(geometry=geometry, prior="Balanced200 near-unique rows per mode",
                                            generator="Identity affine host", learned_output_sigma=fixture.trainer.output_sigma(),
                                            device="cpu", max_steps=case["default_steps"], public_preludes=1,
                                            completed_steps=0, ordinary_updates=0, reference_fit_updates=0,
                                            backend_selection=fixture.trainer.policy.served_snapshot().get("backend_selection")),
                          observer_purity=dict(state_sha256=before_hash, state_unchanged=pure,
                                               global_rng_unchanged=global_pure,
                                               complete_training_rng_in_fixture_state=True),
                          collapsed_control=dict(status="NOT_CAPTURED", reason="Positive six-case admission captures take priority within120 seconds"),
                          artifacts={key: dict(path=str(path), sha256=api_run.file_hash(path))
                                     for key, path in (("state", state_path), ("samples", sample_path))},
                          ordinary_qualification_credit=False, elapsed_seconds=time.monotonic() - case_start)
            records.append(record)
            fixtures.append(fixture)
            save("CAPTURING")
            print(json.dumps(dict(family=family, case=name, status=record["status"],
                                  elapsed_seconds=record["elapsed_seconds"], metrics=observation["metrics"])), flush=True)

    # This control is optional under the total cap. It does not produce a
    # second positive capture or choose between favorable RNG realizations.
    for record, fixture in zip(records, fixtures):
        elapsed = time.monotonic() - started
        if elapsed + max(2., record["elapsed_seconds"] * 1.3) >= wall_cap - 2.:
            record["collapsed_control"]["reason"] = "Remaining whole-run allowance cannot fit conservative control estimate"
            continue
        initial_state_hash = state_hash(fixture.state_dict())
        global_rng = torch.random.get_rng_state().clone()
        with torch.no_grad():
            prior = fixture.trainer.prior.z.clone()
            fixture.trainer.prior.z.zero_()
        control = scalar_observation(fixture, count=cases[record["case_id"]]["eval_samples"], seed=34002)
        with torch.no_grad():
            fixture.trainer.prior.z.copy_(prior)
        if control["passed"] or state_hash(fixture.state_dict()) != initial_state_hash or not torch.equal(global_rng, torch.random.get_rng_state()):
            raise ValueError("collapsed native control passed or altered the restored positive fixture/RNG")
        record["collapsed_control"] = dict(status="EXPECTED_FAIL", observation=control)
        save("CAPTURING_CONTROLS")
        print(json.dumps(dict(family=record["family"], case=record["case_id"], collapsed_control="EXPECTED_FAIL")), flush=True)
    receipt = save("COMPLETE")
    (output / "reproducer.py").write_bytes(Path(__file__).read_bytes())
    (output / "construction-reproducer.py").write_bytes(Path(__file__).with_name("policy_native_representation.py").read_bytes())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
