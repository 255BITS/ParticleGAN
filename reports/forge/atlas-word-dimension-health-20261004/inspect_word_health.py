"""Portable retained-data inspection only: no model, sampler, scorer or run."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import struct

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch

TASK = "five_word_joint_acquisition_word_joint_policy_min11_v1"
SOURCE_COMMIT = "fb7acc775b3a1a6184d36b55e035b9da04531492"
SOURCE_DIGEST = "f380eed990931bacb205e6537beaf387fdbb676ffc97b6f32f19ff903ae1cfed"
SOURCE_PINS = {
    "experiments/forge/policy_adapters.py": "95f7bf03c878d484d01a0fdf0f6b5c9ad34b8df94106ff7a50b02f7f1dfe67cf",
    "experiments/forge/word_joint_policy_adapters.py": "52405305a374e9631b76d50bd35a462adc2cb212f08f7c6dbdc7045b4d3dbf08",
    "experiments/forge/views.py": "d0bafd90cb5bc727bb21b554678245e3d028bdd4c04fcd5cae8af40a513727d1",
    "particlegan/birth_death.py": "b14c50c611a8cf188347e391739fca50a5171200eb6be90e4e3d1509d64e47ed",
}
RAW_PINS = {
    "raw-result.json": "e26ec506c81e8cdd74447c466d516ef30785de501da28dd50e698c115ebe7515",
    "graded-result.json": "5eb697cf6678ef764204ca5cd9e755a9d2f6f1fe0530c58a608b18ee9e3d766a",
    "resolved.json": "52a639183e9ed3c541b90311a16d0968ed85cd7cb039c64d0df693f4cc43bbc1",
    "run.log": "22cb009f52291036a500322f556802fe4cad99adc2c3120184044084a89f0db1",
    "word-joint-policy/state.pt": "16388c4e59a657a154a4730e285191acfd18fbc23d41102d0c3ba82227efb8a7",
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_health_functions(path):
    # Compile only the two pinned, read-only functions. No module import,
    # constructor, public loader, forward or official numerical scorer runs.
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"finite_policy_state", "typed_state_digest"}]
    if len(functions) != 2:
        raise ValueError("pinned health function definitions are missing")
    scope = dict(torch=torch, math=math, hashlib=hashlib, struct=struct)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    return scope["finite_policy_state"], scope["typed_state_digest"]


def inspect(run, source):
    torch.set_num_threads(1)
    manifest = json.loads((source / "forge-source.json").read_text())
    if manifest["origin_commit"] != SOURCE_COMMIT or manifest["digest"] != SOURCE_DIGEST:
        raise ValueError("wrong frozen source identity")
    manifest_path = source / "forge-source.json"
    inputs = [dict(role="source", relative_path="forge-source.json",
                   sha256=file_hash(manifest_path), bytes=manifest_path.stat().st_size)]
    for role, root, pins in (("source", source, SOURCE_PINS), ("raw", run, RAW_PINS)):
        for relative, expected in pins.items():
            path = root / relative
            if path.is_symlink() or file_hash(path) != expected:
                raise ValueError("changed retained input: " + relative)
            if role == "source" and manifest["files"][relative] != expected:
                raise ValueError("source does not match frozen manifest")
            inputs.append(dict(role=role, relative_path=relative, sha256=expected, bytes=path.stat().st_size))
    raw = json.loads((run / "raw-result.json").read_text())
    grade = json.loads((run / "graded-result.json").read_text())
    resolved = json.loads((run / "resolved.json").read_text())
    study_path = run.parents[1] / "study.json"
    study = json.loads(study_path.read_text())
    inputs.append(dict(role="study", relative_path="study.json", sha256=file_hash(study_path), bytes=study_path.stat().st_size))
    evidence = raw["evidence"]
    artifact = evidence["artifact_manifest"]["files"]["state.pt"]
    if artifact["sha256"] != RAW_PINS["word-joint-policy/state.pt"]:
        raise ValueError("raw checkpoint attestation differs")
    state = torch.load(run / "word-joint-policy/state.pt", map_location="cpu", weights_only=True)
    health, digest = original_health_functions(source / "experiments/forge/policy_adapters.py")
    state_sha = digest(state)
    if state_sha != evidence["checkpoint"]["state_sha256"]:
        raise ValueError("complete retained typed state differs")
    nonfinite = []
    tensor_count = 0

    def visit(value, path=()):
        nonlocal tensor_count
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            if value.is_floating_point() or value.is_complex():
                mask = ~torch.isfinite(value)
                if bool(mask.any()):
                    nonfinite.append(dict(path=list(path), kind="tensor", dtype=str(value.dtype),
                        shape=list(value.shape), count=int(mask.sum()),
                        nan_count=int(torch.isnan(value).sum()), inf_count=int(torch.isinf(value).sum()),
                        frozen_health_accepts=health(value, path)))
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + (key,))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, path + (index,))
        elif isinstance(value, float) and not math.isfinite(value):
            nonfinite.append(dict(path=list(path), kind="float", value=repr(value),
                                  frozen_health_accepts=health(value, path)))

    visit(state)
    birth = state["policy"]["birth_death"]
    unique, counts = torch.unique(birth["reservoir"], dim=0, return_counts=True)
    observations = evidence["observations"]
    task = resolved["request"]["tasks"][TASK]
    if len(observations) != 24 or state["caller_cursor"] != 20001 or state["policy"]["completed_steps"] != 20001:
        raise ValueError("original full observation/update counts differ")
    all_finite = all(isinstance(value, (int, float)) and math.isfinite(value)
                     for row in observations for value in row.values())
    if not all_finite or not all(row["pure"] for row in evidence["policy_purity"]):
        raise ValueError("retained metric/purity facts differ")
    rejected = [row for row in nonfinite if not row["frozen_health_accepts"]]
    if len(rejected) != 1 or rejected[0]["path"] != ["policy", "birth_death", "last", "d_R"]:
        raise ValueError("failure attribution differs")
    last = {key: repr(value) if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in birth["last"].items()}
    report = dict(
        schema="pg_word_undefined_dimension_retained_diagnosis_v1",
        scope="retained_data_health_diagnosis_no_numeric_regrading",
        task_id=TASK, family="atlas_word_joint_min11", source_commit=SOURCE_COMMIT,
        source_digest=SOURCE_DIGEST, checkpoint_state_sha256=state_sha,
        input_roots=dict(run=str(run), frozen_source=str(source)), inputs=inputs,
        exact_recipe=state["recipe"], initialization=state["initialization"],
        original_task_execution=task["execution"],
        preserved_original=dict(family_status=study["status"], gate=grade["grades"][TASK],
            paid_seconds=study["jobs"][0]["paid_wall_seconds"], reserve_seconds=study["jobs"][0]["unmeasured_interrupt_reserved_seconds"],
            numerical_grade_changed=False, checkpoint_changed=False),
        checkpoint_health=dict(tensor_leaves=tensor_count, nonfinite_leaves=nonfinite,
            rejected_leaves=rejected, frozen_health_accepts=health(state),
            learned_parameter_or_optimizer_nonfinite_found=False),
        source_defined_skip=dict(last=last, counters=birth["counters"], moved_rows=None if birth["moved_rows"] is None else birth["moved_rows"].tolist(),
            reservoir_shape=list(birth["reservoir"].shape), unique_raw_reference_rows=len(unique),
            reference_multiplicities=sorted(counts.tolist()),
            duplicate_nearest_neighbor_reason="Every raw reference row has an exact duplicate; the source's deterministic critic features preserve equality.",
            source_statistic_lines=[206, 215], source_skip_lines=[511, 524],
            earliest_exact_observable_step=20001, first_occurrence_not_retained=True,
            dimension_skip_counter_is_not_a_per_update_trace=True),
        original_observation=dict(updates=20001, count=24, thresholds=task["evaluation"]["thresholds"],
            steps=[row["step"] for row in observations], all_metrics_finite=all_finite,
            final_metrics=evidence["live"],
            recorded_reconstruction_exact_values=sorted({row["reconstruction_exact"] for row in observations}),
            policy_purity_count=len(evidence["policy_purity"]), all_policy_reads_pure=True,
            all_named_deviations_zero=all(row["unintended_rng_deviations"] == 0 for row in evidence["rng_audits"]),
            optimizer_updates=evidence["guards"]["optimizer_updates"],
            selected_snapshots=[dict(step=row["completed_steps"], selected_source=row["selected_source"],
                snapshot_sha256=row["snapshot_sha256"]) for row in evidence["policy_observations"]],
            no_new_numeric_verdict_computed=True),
        causal_limit="The sentinel explains the audit rejection, not the observed loss of word coverage or paired reconstruction. Missing per-update dimension/model traces prevent earlier timing or optimizer-mechanism attribution.",
        minimal_repair="Audit health only: recognize the exact public clocked no-move undefined-dimension branch; preserve NaN bytes and reject nonfinite learned/unknown state. No package algorithm or numerical-gate change.",
        future_source_requirement="Commit the audit patch and bind new implementation-source/task pins before any future execution; never substitute it into the old raw guard/grade. No unchanged C6 retry is proposed.",
        actions=dict(models_constructed=0, training_restores=0, sampler_calls=0,
            official_scoring_calls=0, optimizer_updates=0, cuda_initialized=torch.cuda.is_initialized(),
            raw_writes=0, numeric_regrading=False))
    for row in inputs:
        root = source if row["role"] == "source" else run.parents[1] if row["role"] == "study" else run
        if file_hash(root / row["relative_path"]) != row["sha256"]:
            raise ValueError("input changed during read-only inspection")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = inspect(args.run.resolve(), args.source_root.resolve())
    if args.output.exists():
        raise FileExistsError("retain existing diagnosis; choose a new report path")
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="READ_ONLY_DIAGNOSIS", report=str(args.output),
                         sha256=file_hash(args.output), rejected_leaves=len(report["checkpoint_health"]["rejected_leaves"]),
                         numeric_regrading=False), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
