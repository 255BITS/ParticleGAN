"""Reclassify retained rotated/guided endpoint evidence, without training.

Original source/card/review identities stay immutable. This verifies the entire
original artifact manifest, then strengthens only the signed code-use decision.
It does not rerun tensor scoring or grant qualification to the repaired source.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from . import run_e22_routed_convergence_rotated_teacher as runner
    from . import review_e22_routed_convergence_rotated_teacher as reviewer
else:
    import run_e22_routed_convergence_rotated_teacher as runner
    import review_e22_routed_convergence_rotated_teacher as reviewer


ROOT = Path(__file__).resolve().parents[1]
RUNNER = "examples/run_e22_routed_convergence_rotated_teacher.py"
REVIEWER = "examples/review_e22_routed_convergence_rotated_teacher.py"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def read_bound(path, expected):
    if sha(path) != expected:
        raise ValueError("bound artifact changed: " + str(path))
    return json.loads(Path(path).read_text())


def signed_source_proof(original, repaired):
    """Allow one pure gate correction and reject any other source AST change."""
    old, new = ast.parse(original), ast.parse(repaired)
    signed = ast.parse('all(value > 1e-6 for value in witness["zero_code_minus_live_test_game"].values())').body[0].value
    unsigned = ast.parse('all(abs(value) > 1e-6 for value in witness["zero_code_minus_live_test_game"].values())').body[0].value

    class RestoreUnsigned(ast.NodeTransformer):
        replacements = 0

        def visit_Call(self, node):
            if ast.dump(node) == ast.dump(signed):
                self.replacements += 1
                return deepcopy(unsigned)
            return self.generic_visit(node)

    correction = RestoreUnsigned()
    restored = correction.visit(new)
    if correction.replacements != 1 or ast.dump(restored) != ast.dump(old):
        raise ValueError("source repair must change exactly one signed evaluation gate")
    return {"changes": ["absolute code effect to positive beneficial code effect"],
            "other_source_ast_unchanged": True,
            "archived_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "repaired_sha256": hashlib.sha256(repaired.encode()).hexdigest()}


def classify_scores(scores, witnesses):
    first = runner.final_gates(scores, witnesses)
    second = reviewer.independent_gates(scores, witnesses)
    if first != second:
        raise AssertionError("runner and independent signed classifications disagree")
    return first


def metadata_source_proof(original, repaired):
    """Keep the stable factory fingerprint repair distinct from trained law."""
    old, new = ast.parse(original), ast.parse(repaired)
    def held_body(module):
        return ast.Module(body=[node for node in module.body
            if not (isinstance(node, ast.FunctionDef) and node.name == "factory_binding_manifest")
            and not (isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == "marshal")],
            type_ignores=[])
    if (sum(isinstance(node, ast.FunctionDef) and node.name == "factory_binding_manifest" for node in old.body) != 1
            or sum(isinstance(node, ast.FunctionDef) and node.name == "factory_binding_manifest" for node in new.body) != 1
            or "marshal.dumps" not in original or "marshal.dumps" in repaired
            or ast.dump(held_body(old)) != ast.dump(held_body(new))):
        raise ValueError("metadata repair changed more than factory identity encoding")
    return {"changes": ["marshal code bytes to canonical code fields", "bind actual factory defaults"],
            "other_source_ast_unchanged": True,
            "archived_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "repaired_sha256": hashlib.sha256(repaired.encode()).hexdigest(),
            "qualification": "New metadata identity only; existing cards, data and checkpoints retain original identities"}


def reclassify_run(run, family):
    run = Path(run).resolve()
    summary_path = ROOT / f"docs/e22_routed_convergence_{family}_results.json"
    summary = json.loads(summary_path.read_text())
    review = read_bound(run / "independent-review.json", summary["review_sha256"])
    receipt = read_bound(run / "receipt.json", review["execution_receipt_sha256"])
    if review.get("qualified") is not True or review.get("status") != "qualified" or receipt["status"] != "complete":
        raise ValueError("complete independently reviewed original cohort required")
    task = f"routed_convergence_{family}_v1"
    if receipt["task"] != task or receipt["contract"]["task_id"] != task:
        raise ValueError("wrong original task law")
    for key in ("endpoint_scores", "particle_witnesses", "source_hashes", "gates"):
        if summary[key] != review[key]:
            raise ValueError("published summary differs from bound original review: " + key)
    if (review["source_hashes"] != receipt["bindings"]["source_hashes"]
            or review["native_source_digest"] != receipt["bindings"]["native_source_hash"]
            or reviewer.native_hash() != review["native_source_digest"]
            or review["data_digest"] != receipt["data_digest"]):
        raise ValueError("original source, native package or data binding differs")

    artifacts = {"receipt.json": review["execution_receipt_sha256"]}
    for relative, expected in receipt["source_archive"].items():
        artifacts["source/" + relative] = expected
    artifacts["data.pt"] = receipt["data_file_sha256"]
    for arm in runner.law.ARMS:
        record = receipt["arms"][arm]
        steps = {entry["step"] for entry in record["checkpoints"]}
        if (record["status"] != "complete" or len(record["checkpoints"]) != 35
                or steps != set(runner.checkpoint_steps())
                or review["state_counts"][arm] != 35 or review["curve_counts"][arm] != 34):
            raise ValueError("original fixed state/curve denominator differs")
        for entry in record["checkpoints"]:
            artifacts[arm + "/" + entry["file"]] = entry["sha256"]
        artifacts[arm + "/trace.jsonl"] = sha(run / arm / "trace.jsonl")
    artifacts["common-judge-curves.jsonl"] = sha(run / "common-judge-curves.jsonl")
    for relative, expected in artifacts.items():
        if sha(run / relative) != expected:
            raise ValueError("original artifact changed: " + relative)
    if runner.base.digest(artifacts) != review["artifact_manifest_digest"]:
        raise ValueError("original independent artifact manifest differs")
    proofs = {RUNNER: signed_source_proof((run / "source" / RUNNER).read_text(), (ROOT / RUNNER).read_text())}
    if proofs[RUNNER]["archived_sha256"] != review["source_hashes"][RUNNER]:
        raise ValueError("gate source is not the original qualified runner")

    witnesses = deepcopy(review["particle_witnesses"])
    for arm, witness in witnesses.items():
        if arm not in runner.law.ARMS[1:] or set(witness["zero_code_minus_live_test_game"]) != set(runner.JUDGES):
            raise ValueError("complete original arm/judge particle witness required")
        for judge in runner.JUDGES:
            ablated = receipt["code_ablation"][arm][judge]["test"]["paired_game"]
            live = receipt["endpoint_scores"][arm + "@6400"][judge]["test"]["paired_game"]
            value = witness["zero_code_minus_live_test_game"][judge]
            if not all(math.isfinite(item) for item in (ablated, live, value)) or not math.isclose(ablated - live, value, abs_tol=2e-7, rel_tol=2e-7):
                raise ValueError("signed witness disagrees with actual saved endpoint scores")
    gates = classify_scores(review["endpoint_scores"], witnesses)
    inputs = {"published_summary": (summary_path, sha(summary_path)),
              "independent_review": (run / "independent-review.json", summary["review_sha256"])}
    if family == "guided_pair":
        completion_path = run / "independent-review.json.completion.json"
        completion = read_bound(completion_path, summary["review_completion_sha256"])
        if completion.get("complete") is not True or completion["review_sha256"] != summary["review_sha256"]:
            raise ValueError("original guided review completion differs")
        inputs["review_completion"] = (completion_path, summary["review_completion_sha256"])
        inputs["execution_completion"] = (run / "execution-completion.json", review["execution_completion_sha256"])
    for relative, expected in artifacts.items():
        if sha(run / relative) != expected:
            raise AssertionError("read-only classification changed original evidence")
    for path, expected in inputs.values():
        if sha(path) != expected:
            raise AssertionError("original summary or review changed")
    return {"task": task, "original_run": str(run),
            "original_input_sha256": {name: expected for name, (_, expected) in inputs.items()},
            "original_execution_receipt_sha256": review["execution_receipt_sha256"],
            "original_artifact_manifest_digest": review["artifact_manifest_digest"],
            "verified_artifact_files": len(artifacts), "verified_saved_states": 105,
            "original_review_checks": review["checks"], "original_source_hashes": review["source_hashes"],
            "evaluation_repair_proofs": proofs, "old_gates": review["gates"], "signed_gates": gates,
            "gate_decisions_unchanged": gates == review["gates"],
            "signed_code_deltas": {arm: witness["zero_code_minus_live_test_game"] for arm, witness in witnesses.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotated", type=Path, required=True)
    parser.add_argument("--guided", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or any(args.out.resolve().is_relative_to(path.resolve()) for path in (args.rotated, args.guided)):
        parser.error("preserve original evidence; use a fresh separate output path")
    rows = [reclassify_run(args.rotated, "rotated_teacher"), reclassify_run(args.guided, "guided_pair")]
    archived_reviewer = args.guided / "source" / REVIEWER
    proof = signed_source_proof(archived_reviewer.read_text(), (ROOT / REVIEWER).read_text())
    original_rotated_review = json.loads((args.rotated / "independent-review.json").read_text())
    if proof["archived_sha256"] != original_rotated_review["reviewer_sha256"]:
        raise ValueError("archived independent gate source differs from qualified rotated reviewer")
    factory_path = "examples/e22_routed_convergence_guided_pair.py"
    metadata = metadata_source_proof((args.guided / "source" / factory_path).read_text(), (ROOT / factory_path).read_text())
    if metadata["archived_sha256"] != rows[1]["original_source_hashes"][factory_path]:
        raise ValueError("metadata source is not the original qualified guided factory")
    report = {"schema": "routed_endpoint_beneficial_signed_reclassification_v1",
              "trained_particle_gate": "unsigned_effect_v1", "evaluated_particle_gate": "beneficial_signed_v2",
              "scope": "Hash-bound original saved endpoint metrics; no new tensor scoring or trained-source qualification",
              "quality_training_updates": 0, "software_replay_updates": 0, "model_or_optimizer_updates": 0,
              "original_evidence_unchanged": True, "qualification_credit": "none",
              "classification_source_sha256": sha(__file__), "independent_gate_repair_proof": proof,
              "separate_guided_metadata_repair_proof": metadata, "records": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"out": str(args.out), "records": len(rows), "saved_states_bound": 210,
                      "model_or_optimizer_updates": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
