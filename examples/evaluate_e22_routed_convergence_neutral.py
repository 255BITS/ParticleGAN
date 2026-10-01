"""Recover only the fixed neutral evaluation, preserving the failed receipt.

No training runs here. The archived training source must differ from the
repaired helper by exactly the fresh carrier used before checkpoint scoring.
"""

import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import time
import torch

if __package__:
    from . import e22_routed_convergence_neutral as neutral
    # The held parent runner imports its core by its direct-script name.
    import sys
    sys.modules.setdefault("e22_routed_convergence", neutral.baseline)
    from . import run_e22_routed_convergence as common
else:
    import e22_routed_convergence_neutral as neutral
    import run_e22_routed_convergence as common

baseline = neutral.baseline


def observational_source_repair(original, repaired):
    old, new = ast.parse(original), ast.parse(repaired)
    insertion = ast.dump(ast.parse("loop = make_neutral_loop(data, bindings=bindings)").body[0])
    for index, node in enumerate(ast.walk(new)):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for position, statement in enumerate(value):
                    if isinstance(statement, ast.Assign) and ast.dump(statement) == insertion:
                        candidate = deepcopy(new)
                        owner = list(ast.walk(candidate))[index]
                        getattr(owner, field).pop(position)
                        if ast.dump(candidate) == ast.dump(old):
                            return {"sole_change": "fresh neutral loop before fixed checkpoint evaluation",
                                    "training_ast_unchanged": True}
    raise ValueError("repair changed more than the declared observational carrier")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=common.ROOT / "runs/routed-convergence-v1")
    args = parser.parse_args()
    output = args.run / "recovered-evaluation-receipt.json"
    curves_path = args.run / "recovered-common-judge-curves.jsonl"
    if output.exists() or curves_path.exists():
        parser.error("recovery artifacts exist; preserve them")
    torch.set_num_threads(1)
    start = time.monotonic()
    failed_path = args.run / "training-receipt-error.json"
    failed = json.loads(failed_path.read_text())
    parent = json.loads((args.parent / "receipt.json").read_text())
    archive = args.run / "training-source-504df385.py"
    helper = Path(neutral.__file__).resolve()
    helper_key = str(helper.relative_to(common.ROOT))
    bindings = failed["bindings"]
    old_hashes = bindings["source_hashes"]
    if failed["status"] != "error" or failed["error"] != "ValueError: checkpoint output shape differs from the resolved policy":
        parser.error("expected the preserved post-training carrier error")
    if common.sha(archive) != old_hashes[helper_key]:
        parser.error("archived training source does not match saved training bindings")
    proof = observational_source_repair(archive.read_text(), helper.read_text())
    for name, sha in old_hashes.items():
        if name != helper_key and common.sha(common.ROOT / name) != sha:
            parser.error("frozen scientific source or card changed")
    if common.native_source_hash() != bindings["native_source_hash"]:
        parser.error("native source changed")
    if common.sha(args.parent / "receipt.json") != bindings["parent_receipt_sha256"]:
        parser.error("parent receipt changed")
    artifact_hashes = {path.name: common.sha(path) for path in (archive, failed_path, args.run / "trace.jsonl")}
    for step in (*range(0, 6401, 200), 802):
        path = args.run / f"step-{step:04d}.pt"
        artifact_hashes[path.name] = common.sha(path)
    for entry in failed["checkpoints"]:
        if artifact_hashes[entry["file"]] != entry["sha256"]:
            parser.error("trained checkpoint changed after the failed evaluation")
    trace_count = 0
    with (args.run / "trace.jsonl").open() as trace, (args.parent / neutral.ARM / "trace.jsonl").open() as reference:
        for trace_count, (line, expected_line) in enumerate(zip(trace, reference, strict=True), 1):
            row, expected = json.loads(line), json.loads(expected_line)
            if row["step"] != trace_count or row["batch_indices"] != expected["batch_indices"] or row["paired_base_digest"] != expected["paired_base_digest"]:
                raise AssertionError("saved input/paired streams differ from the parent")
    if trace_count != 6400 or not failed["recovery_witness"]["state_exact"]:
        parser.error("incomplete saved training/recovery evidence")
    repaired_hash = common.sha(helper)
    recovery_hash = common.sha(__file__)

    def budget():
        if (common.sha(helper) != repaired_hash or common.sha(__file__) != recovery_hash
                or common.native_source_hash() != bindings["native_source_hash"]):
            raise RuntimeError("recovery source changed")
        if failed["wall_seconds"] + time.monotonic() - start > failed["contract"]["execution"]["timeout_seconds"]:
            raise TimeoutError("original training plus recovery exhausted its declared total budget")

    original = baseline.make_data()
    data = neutral.make_neutral_data(original)
    if data["digest"] != failed["data_digest"]:
        parser.error("frozen data identity changed")
    panels = baseline.evaluation_panels(data)
    if baseline.digest(panels) != parent["private_panel_digest"]:
        parser.error("private evaluation panel changed")
    judges = {}
    for name in failed["contract"]["evaluation"]["common_judges"]:
        arm, step = name.split("@")
        state = torch.load(args.parent / arm / f"step-{int(step):04d}.pt", weights_only=False)
        with torch.random.fork_rng(devices=[]):
            judge = baseline.ConditionalCritic(data["scale"])
        judge.load_state_dict(state["training"]["models"]["critic"], strict=True)
        judge.eval().requires_grad_(False)
        if baseline.digest(judge.state_dict()) != parent["judges"][name]:
            parser.error("mandatory parent critic changed")
        judges[name] = judge
    loop = neutral.make_neutral_loop(data, bindings=bindings)
    scores_at = {}
    with curves_path.open("w", buffering=1) as curves:
        for step in range(0, 6401, 200):
            budget()
            baseline.restore(loop, torch.load(args.run / f"step-{step:04d}.pt", weights_only=False))
            scores = {name: {pool: baseline.evaluate(loop, judge, pool, panels) for pool in baseline.SPLITS}
                      for name, judge in judges.items()}
            curves.write(json.dumps({"step": step, "scores": scores}, allow_nan=False) + "\n")
            if step in (0, 1600, 6400):
                scores_at[str(step)] = {name: {pool: common.compact(value) for pool, value in pools.items()}
                                       for name, pools in scores.items()}
            print(json.dumps({"event": "recovered_score", "step": step,
                              "test_game": {name: pools["test"]["paired_game"] for name, pools in scores.items()}}), flush=True)
    ablation = {name: {pool: common.compact(baseline.evaluate(loop, judge, pool, panels, code_ablation=True))
                       for pool in baseline.SPLITS} for name, judge in judges.items()}
    witness = deepcopy(failed["final_particle_witness"])
    witness["zero_code_minus_live_test_game"] = {name: ablation[name]["test"]["paired_game"]
                                                - scores_at["6400"][name]["test"]["paired_game"] for name in judges}
    improvements, reductions = {}, {}
    for name in judges:
        particle = parent["endpoint_scores"][neutral.ARM + "@6400"][name]["test"]["paired_game"]
        improvements[name] = particle - scores_at["6400"][name]["test"]["paired_game"]
        if name.endswith("@6400"):
            ordinary = parent["endpoint_scores"]["ordinary_native_game@6400"][name]["test"]["paired_game"]
            reductions[name] = improvements[name] / (particle - ordinary)
    coverage = failed["particle_coverage"]
    retained = (witness["bridge_still_trainable"] and all(value > 0 for value in witness["C_norms"].values())
                and coverage["live_bank_updates"] > 0 and coverage["live_query_updates"] > 0
                and all(abs(value) > 1e-6 for value in witness["zero_code_minus_live_test_game"].values()))
    for name, sha in artifact_hashes.items():
        if common.sha(args.run / name) != sha:
            raise AssertionError("read-only recovery changed an original training artifact")
    budget()
    receipt = {**deepcopy(failed), "status": "complete_recovered_evaluation", "error": None,
               "original_training_status": "error_after_complete_training", "original_error_receipt_sha256": common.sha(failed_path),
               "evaluation_repair": {**proof, "archived_training_source_sha256": common.sha(archive),
                                     "repaired_helper_sha256": repaired_hash, "recovery_cli_sha256": recovery_hash,
                                     "saved_training_bindings_unchanged": True},
               "original_artifact_sha256": artifact_hashes, "original_training_artifacts_unchanged": True,
               "endpoint_scores": scores_at, "code_ablation": ablation, "final_particle_witness": witness,
               "paired_game_improvements": improvements, "endpoint_gap_reduction": reductions,
               "retained_particle_gate": retained,
               "mechanism_supported": retained and all(value > 1e-4 for value in improvements.values())
                                              and all(value >= .5 for value in reductions.values()),
               "recovery_evaluation_seconds": time.monotonic() - start,
               "total_training_and_recovery_seconds": failed["wall_seconds"] + time.monotonic() - start,
               "new_training_updates": 0, "qualification_credit": "none"}
    common.write_json(output, receipt)
    print(json.dumps({"event": "recovered_complete", "receipt": str(output),
                      "mechanism_supported": receipt["mechanism_supported"]}), flush=True)


if __name__ == "__main__":
    main()
