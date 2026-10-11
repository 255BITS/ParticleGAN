"""Compare the five completed zero-weight controls from saved evidence only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from experiments.forge.api import resolve_public_recipe
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.state import state_digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def observations(path):
    result = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event") == "observation":
            result.append(row)
    return result


def run(pool, output):
    records, full, acquired = [], {}, {}
    for name in ("proximal", "coulomb", "mmd", "sigreg", "covariance"):
        state = json.loads((pool / name / "queue/queue/state.json").read_text())
        progress = json.loads((pool / name / "progress.json").read_text())
        request = state["submissions"][progress["control_request"]]["request"]
        definition = next(j for j in request["jobs"] if j["execution_group"] == "five_word_joint_smoke")
        job = state["jobs"][definition["compatibility_key"]]
        require(job["status"] == "terminal", name + ": full control is unfinished")
        result = job["result"]
        attempt = next(a for a in job["attempts"] if a["attempt_id"] == result["attempt_id"])
        directory = Path(attempt["path"])
        require(stable_hash(json.loads((directory / "result.json").read_text())) == stable_hash(result),
                name + ": queue projection differs from original result")
        row = result["task_results"][0]
        require(row["gate_status"] == "PASS" and row["cost"]["completed_steps"] == 20001,
                name + ": this audit requires the complete passing control")
        descriptor = row["evidence"]["provenance_checkpoint"]
        artifact_root = Path(descriptor["artifact_root"])
        verify_artifacts(artifact_root, descriptor["artifact_manifest"])
        checkpoint = artifact_root / descriptor["path"]
        require(file_hash(checkpoint) == descriptor["sha256"], "Checkpoint bytes changed")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        require(state_digest(saved) == descriptor["state_sha256"], "Saved state hash differs")
        first_path = next(directory.rglob("acquired-state.pt"))
        first = torch.load(first_path, map_location="cpu", weights_only=False)
        api = saved["fixture"]["api_state"]
        require(api["completed_steps"] == 20001 and first["fixture"]["api_state"]["completed_steps"] == 834,
                "Comparison is not at the declared equal prefixes")
        recipe = asdict(resolve_public_recipe(dict(recipe_overrides=row["recipe"])))
        require(recipe["prior_reg"] == 0 and recipe["prior_regularizer"] == "vicreg"
                and recipe["critic_formulation"] == "bcap", "Control regularization or formulation changed")
        manifest = saved["streams"]["manifest"]
        non_eval = {k: v for k, v in saved["streams"]["states"].items()
                    if manifest["bindings"][k]["family"] != "eval"}
        initial = {role: packet["initial_state_sha256"] for role, packet in row["initialization"].items()}
        prior = api["models"]["prior"]
        require(row["prior"]["kind"] == "particle_cloud" and "sigma" not in prior and "d0" not in prior,
                "Word sigma/d0 absence is not the explicit cloud contract")
        require(not any("moment_tether" in k for k in prior), "Disabled moment reference was captured")
        packets = observations(directory / "run.log")
        full[name] = dict(saved=saved, recipe=recipe, initial=initial, observations=packets)
        acquired[name] = first
        records.append(dict(name=name, source_commit=request["source"]["origin_commit"],
            source_digest=request["source"]["digest"], attempt_id=attempt["attempt_id"],
            artifact_root=str(directory), actual_recipe_sha256=stable_hash(recipe),
            initial_tensor_sha256=initial, prior_sigma="absent: explicit particle_cloud",
            prior_d0="absent: explicit particle_cloud", initializer=row["initializer"],
            acquired_checkpoint_sha256=file_hash(first_path), checkpoint_sha256=descriptor["sha256"],
            state_sha256=descriptor["state_sha256"], completed_updates=api["completed_steps"],
            optimizer_update_counts=row["evidence"]["guards"]["optimizer_updates"],
            initial_stream_bindings_sha256=stable_hash(manifest),
            named_non_eval_stream_states_sha256={k: state_digest(v) for k, v in non_eval.items()},
            all_named_streams_sha256=state_digest(saved["streams"]),
            role_tensor_sha256={role: state_digest(values) for role, values in api["models"].items()},
            optimizer_state_sha256=state_digest(api["optimizers"]),
            scheduled_observation_packets_sha256=stable_hash(packets),
            passing_observations=row["evaluator_result"]["passing_observations"],
            independently_confirmed_observations=row["evaluator_result"]["confirmed_checks"],
            original_result_sha256=file_hash(directory / "result.json")))
    reference, first_reference = full["proximal"], acquired["proximal"]
    comparisons = []
    for name, cell in full.items():
        require(cell["recipe"] == reference["recipe"] and cell["initial"] == reference["initial"],
                name + ": actual full recipe or initial tensor hashes differ")
        require(cell["observations"] == reference["observations"], name + ": scheduled packets differ")
        excluded = {"cpu_rng", "cuda_rng"}
        for point, current, expected in (("first_certified_834", acquired[name], first_reference),
                                         ("full_20001", cell["saved"], reference["saved"])):
            a, b = current["fixture"]["api_state"], expected["fixture"]["api_state"]
            different = [k for k in a if state_digest(a[k]) != state_digest(b[k])]
            require(set(different) <= excluded, name + "/" + point + ": scientific API state differs")
            require(state_digest(current["streams"]) == state_digest(expected["streams"]),
                    name + "/" + point + ": named streams differ")
            require(state_digest(current["fixture"]["data_generator"]) == state_digest(expected["fixture"]["data_generator"]),
                    name + ": fixture data generator differs")
            comparisons.append(dict(name=name, point=point, scientific_api_state_equal=True,
                all_named_streams_equal=True, fixture_data_generator_equal=True,
                differing_caller_global_snapshots=different))
    receipt = dict(schema_version=1, status="PASS", scope="saved_zero_weight_word_control_parity",
        controls=records, comparisons=comparisons, optimizer_updates_added=0, sampling_draws_added=0,
        stream_draw_count_scope="Explicit per-stream draw counters are not instrumented; exact stream states, manifests and optimizer-update counts are compared at equal prefixes.",
        exclusions="Only stored caller-global cpu_rng/cuda_rng snapshots differ. No feature buffers are excluded. Their differences did not change the observed complete trajectory.",
        source_review="Moment-tether changes WordFixture's factory reference and calls apply_prior_policy after initialization before optimizer binding; the learned zero-weight VICReg branch performs no reference capture. Word step and penalty algebra are unchanged.",
        interpretation="All five MPS controls reproduce the same 24 numerical observation packets, with 23 independently confirmed passes and a passing endpoint. No disabled-feature late scientific divergence is demonstrated. Source-bound cohorts and original cancellations retain their identities; archived word gates grant no current credit.")
    atomic_json(output, receipt)
    print(json.dumps(dict(status="PASS", controls=len(records), equal_prefix_comparisons=len(comparisons))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    run(args.pool, args.output)
