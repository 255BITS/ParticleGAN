"""Bound one zero-update stationarity observation to four qualified tiny critics."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import torch
import particlegan

if __package__:
    from . import e22_routed_game_stationarity as law
    from . import e22_routed_convergence_rotated_teacher as parent
else:
    import e22_routed_game_stationarity as law
    import e22_routed_convergence_rotated_teacher as parent

held = law.held
ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "docs/e22_routed_game_stationarity_v1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def native_sources():
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / "particlegan").rglob("*.py"))}


def preflight(card):
    if (card["schema"] != "routed_game_stationarity_card_v1" or card["task"] != law.TASK
            or card["device"] != "cpu" or card["threads"] != 1 or card["budget_seconds"] != 120
            or card["cases"] != [case[0] for case in law.CASES] or card["optimizer_updates"] != 0
            or card["gradient_absolute_norm_tolerance"] != law.GRADIENT_ATOL):
        raise ValueError("card differs from the single declared zero-update law")
    for name, digest in card["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("frozen source changed: " + name)
    if held.digest(native_sources()) != card["native_source_digest"]:
        raise ValueError("native source differs from qualified rotated task")
    run = Path(card["parent_run"])
    review_path, receipt_path = run / "independent-review.json", run / "receipt.json"
    if sha(review_path) != card["qualified_review_sha256"] or sha(receipt_path) != card["parent_receipt_sha256"]:
        raise ValueError("qualified parent review/receipt identity changed")
    review, receipt = json.loads(review_path.read_text()), json.loads(receipt_path.read_text())
    if (review["schema"] != "routed_convergence_rotated_independent_review_v1"
            or review.get("qualified") is not True or review.get("status") != "qualified"
            or receipt["status"] != "complete" or review["execution_receipt_sha256"] != sha(receipt_path)
            or review["native_source_digest"] != card["native_source_digest"]
            or review["data_digest"] != card["data_digest"] or review["judges"] != card["judge_tensor_digests"]):
        raise ValueError("not the qualified four-critic parent cohort")
    reviewer = ROOT / "examples/review_e22_routed_convergence_rotated_teacher.py"
    if sha(reviewer) != review["reviewer_sha256"]:
        raise ValueError("actual qualified reviewer source changed")
    if review["source_hashes"] != receipt["bindings"]["source_hashes"]:
        raise ValueError("parent source identities disagree")
    for name, digest in review["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("held parent source changed: " + name)
    inputs = {"data": run / "data.pt", "review": review_path, "receipt": receipt_path}
    expected = {"data": receipt["data_file_sha256"], "review": card["qualified_review_sha256"],
                "receipt": card["parent_receipt_sha256"]}
    for judge_name in card["judges"]:
        arm, step = judge_name.split("@")
        entries = [entry for entry in receipt["arms"][arm]["checkpoints"] if entry["step"] == int(step)]
        if len(entries) != 1:
            raise ValueError("qualified tiny checkpoint missing")
        inputs[judge_name] = run / arm / entries[0]["file"]
        expected[judge_name] = entries[0]["sha256"]
        if expected[judge_name] != card["checkpoint_sha256"][judge_name] or inputs[judge_name].stat().st_size > 12 << 20:
            raise ValueError("wrong checkpoint or non-tiny input")
    actual = {name: sha(path) for name, path in inputs.items()}
    if actual != expected or actual["data"] != card["data_file_sha256"]:
        raise ValueError("qualified input file changed")
    return inputs, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/routed-game-stationarity-v1")
    args = parser.parse_args()
    card = json.loads(CARD.read_text())
    started = time.monotonic()
    def budget():
        if time.monotonic() - started > 120:
            raise TimeoutError("fixed 120-second CPU observation budget exhausted")
    if Path(particlegan.__file__).resolve().parent != ROOT / "particlegan":
        parser.error("use PYTHONPATH for this exact checkout")
    inputs, expected = preflight(card)
    budget()
    if not args.execute:
        print(json.dumps(dict(preflight=True, zero_updates=True, source_only=True, seconds=time.monotonic()-started)))
        return
    if not card["execution_authorized"] or args.output.exists():
        parser.error("fixed execution must be authorized and output must be fresh")
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only witness refuses initialized CUDA")
    args.output.mkdir(parents=True)
    source_hashes = {**card["source_sha256"], **native_sources()}
    for relative, digest in source_hashes.items():
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        if sha(target) != digest:
            raise AssertionError("source archive changed")
    plan = dict(card=card, card_sha256=sha(CARD), inputs={name: str(path) for name, path in inputs.items()},
        input_sha256=expected, source_sha256=source_hashes, torch_version=torch.__version__,
        python=platform.python_version(), CPU=platform.processor(), software_cost_scope="separate from this 120-second observation")
    write(args.output / "plan.json", plan)
    results = []
    try:
        data = torch.load(inputs["data"], map_location="cpu", weights_only=False)
        if data["digest"] != card["data_digest"] or held.digest({k:v for k,v in data.items() if k != "digest"}) != data["digest"]:
            raise ValueError("actual qualified data contents changed")
        indices, selection_stream = law.fit_contexts(data)
        context = data["fit"]["context"][indices].contiguous()
        raw_panels = law.private_panels()
        if (indices != card["fit_indices"] or held.digest(context) != card["context_digest"]
                or held.digest(selection_stream) != card["selection_stream_digest"]
                or held.digest(raw_panels) != card["private_panel_digest"]):
            raise ValueError("declared context/panel law changed")
        global_rng = held.digest(torch.get_rng_state())
        with torch.random.fork_rng(devices=[]):
            loop = parent.make_rotated_loop("particle_native_game", data)
            fresh = law.fresh_witness(loop)
            judges, owners = {}, {}
            for name in card["judges"]:
                budget()
                state = torch.load(inputs[name], map_location="cpu", weights_only=False)
                if state["step"] != int(name.split("@")[1]) or state["training"]["completed_steps"] != state["step"]:
                    raise ValueError("critic clock mislabeled")
                current = held.ConditionalCritic(data["scale"])
                current.load_state_dict(state["training"]["models"]["critic"], strict=True)
                current.eval().requires_grad_(False)
                even = law.ReflectionEvenCritic(data["scale"])
                even.load_state_dict(current.state_dict(), strict=True)
                even.eval().requires_grad_(False)
                if held.digest(current.state_dict()) != card["judge_tensor_digests"][name] or held.digest(even.state_dict()) != held.digest(current.state_dict()):
                    raise ValueError("actual learned critic tensors changed")
                judges[name] = (current, even)
                owners.update({name + "/current": current, name + "/even": even})
                del state
            before = law.owner_identity(loop, owners)
            initial_state_digest = before["native_checkpoint"]
            for start in range(0, 12, 4):
                budget()
                batch_results = law.observe_batch(loop, judges, context[start:start+4], raw_panels[:,start:start+4])
                for result in batch_results:
                    result["fit_indices"] = indices[start:start+4]
                    for row, record in enumerate(result["per_context"]):
                        index = indices[start+row]
                        record.update(fit_index=index, subject=int(data["fit"]["subjects"][index]),
                                      flow_time=float(data["fit"]["times"][index]))
                    print(json.dumps(dict(event="stationarity_case", batch=start//4, judge=result["judge"],
                        case=result["case"], up_norm=result["batch_mean"]["norms"]["generator_Up"],
                        residual_norm=result["batch_mean"]["norms"]["normalized_residual"],
                        stationary=result["batch_mean"]["local_stationarity_verified"])), flush=True)
                results.extend(batch_results)
            if law.owner_identity(loop, owners) != before:
                raise AssertionError("native owners/RNG/flags/modes/.grad/diagnostics changed")
        if held.digest(torch.get_rng_state()) != global_rng or torch.cuda.is_initialized():
            raise AssertionError("global RNG/CUDA changed")
        if {name:sha(path) for name,path in inputs.items()} != expected:
            raise AssertionError("qualified actual input files changed")
        if any(sha(ROOT/name) != digest for name,digest in source_hashes.items()) or sha(CARD) != plan["card_sha256"]:
            raise AssertionError("held/new source/card changed")
        budget()
        report = dict(schema="routed_game_stationarity_observation_v1", complete=True,
            plan_sha256=sha(args.output/"plan.json"), results=results, fresh_particle_witness=fresh,
            initial_native_state_digest=initial_state_digest, optimizer_updates=0, model_updates=0,
            actual_generator_forward_calls=3, native_policy_or_diagnostic_calls=0,
            native_owners_unchanged=True, input_files_unchanged=True, global_rng_unchanged=True,
            full_Supra_forward_calls=0, quality_training=False, qualification_credit="none",
            limits=card["limits"], seconds_before_report_write=time.monotonic()-started)
        write(args.output/"report.json", report)
        budget()
        report_sha = sha(args.output/"report.json")
        budget()
        write(args.output/"completion.json", dict(complete=True, report_sha256=report_sha,
            wall_seconds=time.monotonic()-started, qualification_credit="none"))
        budget()
        print(json.dumps(dict(event="stationarity_complete", wall_seconds=time.monotonic()-started,
            model_updates=0, qualification_credit="none")), flush=True)
    except Exception as error:
        write(args.output/"incomplete.json", dict(complete=False, error=type(error).__name__+": "+str(error),
            results=results, wall_seconds=time.monotonic()-started, model_updates=0))
        write(args.output/"completion.json", dict(complete=False, qualification_credit="none"))
        raise


if __name__ == "__main__":
    main()
